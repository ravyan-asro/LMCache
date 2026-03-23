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

# Lazy imports — only needed at prefill time
_lean_attn_prime_cache_func = None
_flash_attn_forward_func = None
_build_indexcache_metadata_func = None


def _get_lean_attn():
    global _lean_attn_prime_cache_func
    if _lean_attn_prime_cache_func is None:
        from lean_attn import lean_attn_prime_cache_func
        _lean_attn_prime_cache_func = lean_attn_prime_cache_func
    return _lean_attn_prime_cache_func


def _get_fa3():
    global _flash_attn_forward_func, _build_indexcache_metadata_func
    if _flash_attn_forward_func is None:
        import sys
        fa3_path = "/var/tmp/rsanovar3/flash-attention/hopper"
        if fa3_path not in sys.path:
            sys.path.insert(0, fa3_path)
        from flash_attn_interface import _flash_attn_forward
        from sparse_indexcache_utils import build_indexcache_metadata
        _flash_attn_forward_func = _flash_attn_forward
        _build_indexcache_metadata_func = build_indexcache_metadata
    return _flash_attn_forward_func, _build_indexcache_metadata_func


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

        # Compat: vLLM 0.18+ uses embed_input_ids(), older uses get_input_embeddings()
        if hasattr(vllm_model, "embed_input_ids"):
            self._embed = vllm_model.embed_input_ids
        elif hasattr(vllm_model, "get_input_embeddings"):
            self._embed = vllm_model.get_input_embeddings
        else:
            self._embed = vllm_model.model.embed_tokens

        self.num_layers = len(vllm_model.model.layers)

        # Extract head geometry from the first layer's attention
        attn0 = vllm_model.model.layers[0].self_attn
        self.num_heads = attn0.attn.num_heads
        self.num_kv_heads = attn0.attn.num_kv_heads
        self.head_dim = attn0.attn.head_size
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Detect QK-norm (e.g. Qwen3 MoE, MiniMax M2.5)
        self.has_qk_norm = hasattr(attn0, "q_norm") and hasattr(attn0, "k_norm")

        # Detect MiniMax-style QK-norm (uses static forward_qk method)
        self.minimax_qk_norm = False
        if self.has_qk_norm:
            try:
                from vllm.model_executor.layers.mamba.linear_attn import (
                    MiniMaxText01RMSNormTP,
                )
                if isinstance(attn0.q_norm, MiniMaxText01RMSNormTP):
                    self.minimax_qk_norm = True
            except ImportError:
                pass

        # Detect MLP attribute name
        layer0 = vllm_model.model.layers[0]
        if hasattr(layer0, "mlp"):
            self.mlp_attr = "mlp"
        elif hasattr(layer0, "block_sparse_moe"):
            self.mlp_attr = "block_sparse_moe"
        else:
            raise ValueError("Cannot find MLP/MoE attribute on model layer")

    def _apply_qk_norm(self, layer, q, k):
        """Apply QK-norm with compat for Qwen3 and MiniMax styles."""
        if not self.has_qk_norm:
            return q, k
        if self.minimax_qk_norm:
            from vllm.model_executor.layers.mamba.linear_attn import (
                MiniMaxText01RMSNormTP,
            )
            return MiniMaxText01RMSNormTP.forward_qk(
                layer.self_attn.q_norm, layer.self_attn.k_norm, q, k
            )
        else:
            q = layer.self_attn.q_norm(
                q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            ).view(q.shape)
            k = layer.self_attn.k_norm(
                k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
            ).view(k.shape)
            return q, k

    def _run_mlp(self, layer, hidden_states):
        """Run MLP/MoE on a layer."""
        return getattr(layer, self.mlp_attr)(hidden_states)

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

        hidden_states = self._embed(input_ids.cuda())
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

            # --- QK-norm (Qwen3 MoE, MiniMax M2.5, and similar models) ---
            q, k = self._apply_qk_norm(layer, q, k)

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
            hidden_states = self._run_mlp(layer, hidden_states)

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
        attn_backend: str = "lean_attn",
        question_len: int = 0,
        sparse_metadata_caches: list = None,
    ):
        """Generator — runs sparse attention prefill and writes KV to paged buffer.

        Supports two backends:
        - "lean_attn": original lean_attn (expanded KV, [B,H,S,D] layout)
        - "fa3": sparse FA3 Hopper kernel (GQA native, [B,S,H,D] layout)

        Args:
            input_ids: 1-D token IDs (context + question tokens)
            kvcolidx_caches: per-layer kvcolidx tensors (merged across chunks)
            la_hot_tile_caches: per-layer LA_hot_tile_code tensors (merged)
            offset_list: chunk boundary offsets
            page_size: page size for attention
            kvcaches: list of (K_paged, V_paged) per layer from vLLM
            slot_mapping: maps token positions to paged buffer slots
            attn_backend: "lean_attn" or "fa3"
            question_len: number of question tokens (needed for FA3 metadata)
            sparse_metadata_caches: pre-built FA3 sparse metadata per layer
                (list of (sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts)
                 tuples). If provided, skips build_indexcache_metadata() in the
                 per-layer loop.
        """
        import lmcache.c_ops as lmc_ops

        seq_len = input_ids.shape[0]

        hidden_states = self._embed(input_ids.cuda())
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

            # --- QK-norm (Qwen3 MoE, MiniMax M2.5, and similar models) ---
            q, k = self._apply_qk_norm(layer, q, k)

            # --- RoPE ---
            q, k = layer.self_attn.rotary_emb(positions, q, k)

            # --- Sparse attention (backend-dependent) ---
            if attn_backend == "fa3":
                # FA3 expects Q: [B, seqlen, H, D], K/V: [B, seqlen, H_kv, D]
                # No KV expansion needed — FA3 handles GQA natively
                fa3_fwd, _ = _get_fa3()

                q_fa3 = q.view(1, seq_len, self.num_heads, self.head_dim).contiguous()
                k_fa3 = k.view(1, seq_len, self.num_kv_heads, self.head_dim).contiguous()
                v_fa3 = v.view(1, seq_len, self.num_kv_heads, self.head_dim).contiguous()

                # Use pre-built sparse metadata (converted in blend() before prefill)
                if sparse_metadata_caches is not None:
                    sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts = \
                        sparse_metadata_caches[layer_idx]
                else:
                    # Fallback: build on the fly (slow — ~850ms per layer)
                    _, build_meta = _get_fa3()
                    kvcolidx = kvcolidx_caches[layer_idx].contiguous()
                    la_hot_tile = la_hot_tile_caches[layer_idx].contiguous()
                    sparse_n_indices, sparse_n_offsets, sparse_n_mask_counts = build_meta(
                        seqlen_total=seq_len,
                        offset_list=offset_list,
                        question_len=question_len,
                        kvcolidx_cache=kvcolidx,
                        la_hot_tile_code_cache=la_hot_tile,
                        num_heads_kv=self.num_kv_heads,
                        kBlockM=128,
                        kBlockN=128,
                        page_size=page_size,
                        device=str(q.device),
                    )

                softmax_scale = self.head_dim ** (-0.5)
                out, _, _, _ = fa3_fwd(
                    q_fa3, k_fa3, v_fa3,
                    softmax_scale=softmax_scale,
                    causal=True,
                    sparse_n_indices=sparse_n_indices,
                    sparse_n_offsets=sparse_n_offsets,
                    sparse_n_mask_counts=sparse_n_mask_counts,
                )
                # out: [1, seqlen, H, D] → [seqlen, H * D]
                attn_output = out.view(seq_len, -1)

            else:
                # lean_attn expects (bsz, num_heads, seq_len, head_dim)
                lean_attn_fn = _get_lean_attn()

                q_4d = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k_4d = k.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_4d = v.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

                k_expanded = repeat_kv(k_4d, self.num_kv_groups)
                v_expanded = repeat_kv(v_4d, self.num_kv_groups)

                kvcolidx = kvcolidx_caches[layer_idx].contiguous()
                la_hot_tile = la_hot_tile_caches[layer_idx].contiguous()

                attn_output = lean_attn_fn(
                    q_4d.contiguous(),
                    k_expanded.contiguous(),
                    v_expanded.contiguous(),
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
            v_for_cache = v.contiguous()   # (seq_len, kv_size)

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
            hidden_states = self._run_mlp(layer, hidden_states)

            yield
