"""LMCIndexCacheLlamaModel — runs cache gen (dense attn) and prefill
(lean_attn) using vLLM model weights for the IndexCache integration.

This is completely separate from LMCLlamaModel (CacheBlend).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmcache.logging import init_logger
from lmcache.v1.compute.indexcache.config import IndexCacheConfig
from lmcache.v1.compute.indexcache.metadata_utils import (
    find_important_pages_fast,
    get_hot_tile_code_fast,
)

logger = init_logger(__name__)

# Lazy import — only needed at prefill time
_lean_attn_prime_cache_func = None


def _get_lean_attn():
    global _lean_attn_prime_cache_func
    if _lean_attn_prime_cache_func is None:
        from lean_attn import lean_attn_prime_cache_func
        _lean_attn_prime_cache_func = lean_attn_prime_cache_func
    return _lean_attn_prime_cache_func


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads: (B, num_kv_heads, S, D) -> (B, num_heads, S, D)."""
    if n_rep == 1:
        return hidden_states
    B, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        B, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(B, num_kv_heads * n_rep, slen, head_dim)


class LMCIndexCacheLlamaModel(nn.Module):
    """Runs IndexCache cache gen and prefill using vLLM model weights.

    cache_gen_forward():  dense matmul attention → metadata extraction
    compute_layer():      lean_attn prefill → writes KV to paged buffer
    """

    def __init__(self, vllm_model, config: IndexCacheConfig):
        super().__init__()
        self.vllm_model = vllm_model
        self.config = config
        self.num_layers = len(vllm_model.model.layers)

        # Extract head geometry from the first layer's attention
        attn0 = vllm_model.model.layers[0].self_attn
        self.num_heads = attn0.attn.num_heads
        self.num_kv_heads = attn0.attn.num_kv_heads
        self.head_dim = attn0.attn.head_size
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    # ------------------------------------------------------------------
    # Cache Gen: dense attention for metadata extraction
    # ------------------------------------------------------------------
    def cache_gen_forward(self, input_ids: torch.Tensor):
        """Generator — yields (kvcolidx, LA_hot_tile_code) per layer.

        Runs the full model forward on chunk tokens with dense matmul
        attention to materialise the attention matrix, then extracts
        IndexCache metadata via find_important_pages_fast /
        get_hot_tile_code_fast.

        Args:
            input_ids: 1-D token ID tensor (num_tokens,)
        """
        seq_len = input_ids.shape[0]
        num_pages = seq_len // self.config.page_size
        positions = torch.arange(seq_len, device=input_ids.device)
        mh_set = self.config.misbehaving_heads_set

        hidden_states = self.vllm_model.get_input_embeddings(input_ids.cuda())
        residual = None

        for layer_idx in range(self.num_layers):
            layer = self.vllm_model.model.layers[layer_idx]

            # --- Pre-attention layernorm ---
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual
                )

            # --- QKV projection ---
            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

            # --- RoPE ---
            q, k = layer.self_attn.rotary_emb(positions, q, k)

            # --- Reshape for matmul attention ---
            # q: (seq_len, q_size) → (1, num_heads, seq_len, head_dim)
            q_4d = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_4d = k.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_4d = v.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Expand KV heads for attention matmul
            k_expanded = repeat_kv(k_4d, self.num_kv_groups)
            v_expanded = repeat_kv(v_4d, self.num_kv_groups)

            # --- Dense matmul attention ---
            attn_weights = torch.matmul(
                q_4d, k_expanded.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            # Causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=q.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                q_4d.dtype
            )

            # --- Metadata extraction ---
            kvcolidx = find_important_pages_fast(
                attn_scores=attn_scores,
                page_size=self.config.page_size,
                num_topk_kvcols=num_pages,
                layer_idx=layer_idx,
                misbehaving_heads_set=mh_set,
                imp_threshold=self.config.ca_threshold,
                strong_consistency=self.config.ca_consistency,
            )
            la_hot_tile = get_hot_tile_code_fast(
                attn_scores=attn_scores,
                page_size=self.config.page_size,
                layer_idx=layer_idx,
                misbehaving_heads_set=mh_set,
                if_thresh=self.config.if_thresh,
                threshold=self.config.threshold,
                local_pct=self.config.local_pct,
            )

            # --- Complete attention → hidden states for next layer ---
            attn_output = torch.matmul(attn_scores, v_expanded)
            # (1, num_heads, seq_len, head_dim) → (seq_len, num_heads * head_dim)
            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(seq_len, -1)
            )
            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # --- Post-attention layernorm + MLP ---
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield (kvcolidx, la_hot_tile)

    # ------------------------------------------------------------------
    # Prefill: lean_attn sparse attention + write KV to paged buffer
    # ------------------------------------------------------------------
    def compute_layer(
        self,
        input_ids: torch.Tensor,
        kvcolidx_caches: list,
        la_hot_tile_caches: list,
        offset_list: list,
        page_size: int,
        kvcaches: list,
        slot_mapping: torch.Tensor,
    ):
        """Generator — runs lean_attn prefill and writes KV to paged buffer.

        For each layer:
        1. layernorm → qkv_proj → RoPE
        2. repeat_kv (expand KV heads for lean_attn)
        3. lean_attn_prime_cache_func(Q, K, V, kvcolidx, hot_tile, offsets, page_size)
        4. Write unexpanded K, V to vLLM paged buffer
        5. o_proj → post_layernorm → MLP
        6. yield

        Args:
            input_ids: 1-D token IDs for context tokens (num_tokens,)
            kvcolidx_caches: per-layer kvcolidx tensors (merged across chunks)
            la_hot_tile_caches: per-layer LA_hot_tile_code tensors (merged)
            offset_list: chunk boundary offsets for lean_attn
            page_size: page size for lean_attn
            kvcaches: list of (K_paged, V_paged) per layer from vLLM
            slot_mapping: maps token positions to paged buffer slots
        """
        import lmcache.c_ops as lmc_ops

        lean_attn_fn = _get_lean_attn()
        seq_len = input_ids.shape[0]

        hidden_states = self.vllm_model.get_input_embeddings(input_ids.cuda())
        positions = torch.arange(seq_len, device=hidden_states.device)
        residual = None

        for layer_idx in range(self.num_layers):
            layer = self.vllm_model.model.layers[layer_idx]

            # --- Pre-attention layernorm ---
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual
                )

            # --- QKV projection ---
            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

            # --- RoPE ---
            q, k = layer.self_attn.rotary_emb(positions, q, k)

            # --- Reshape for lean_attn ---
            # lean_attn expects (bsz, num_heads, seq_len, head_dim)
            q_4d = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_4d = k.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_4d = v.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Expand KV heads for lean_attn
            k_expanded = repeat_kv(k_4d, self.num_kv_groups)
            v_expanded = repeat_kv(v_4d, self.num_kv_groups)

            # --- Sparse attention via lean_attn ---
            kvcolidx = kvcolidx_caches[layer_idx].contiguous()
            la_hot_tile = la_hot_tile_caches[layer_idx].contiguous()
            q_cont = q_4d.contiguous()
            k_cont = k_expanded.contiguous()
            v_cont = v_expanded.contiguous()

            attn_output = lean_attn_fn(
                q_cont,
                k_cont,
                v_cont,
                kvcolidx_cache=kvcolidx,
                LA_hot_tile_code_cache=la_hot_tile,
                offset_list=offset_list,
                page_size=page_size,
                causal=True,
            )
            # (1, num_heads, seq_len, head_dim) → (seq_len, num_heads * head_dim)
            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(seq_len, -1)
            )

            # --- Write KV to vLLM paged buffer ---
            # Use UNEXPANDED K, V (num_kv_heads) for the paged cache
            k_for_cache = k.contiguous()   # (seq_len, kv_size)
            v_for_cache = v.contiguous()   # (seq_len, kv_size) — v is pre-RoPE, that's correct

            kv_buf = torch.stack(
                [k_for_cache, v_for_cache], dim=0
            )  # (2, seq_len, kv_size)

            lmc_ops.single_layer_kv_transfer(
                kv_buf,
                kvcaches[layer_idx][0],   # K paged cache
                kvcaches[layer_idx][1],   # V paged cache
                slot_mapping,
                False,
                False,
            )

            # --- o_proj → post-attention layernorm → MLP ---
            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield
