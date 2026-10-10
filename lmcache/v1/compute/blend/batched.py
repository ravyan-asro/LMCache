# Batched CacheBlend for the SIFT MLSys Exp 3 (FusionRAG-style serving).
#
# Same algorithm as LMCBlender.process_qkv + LMCLlamaModel.compute_layer, the
# per-request path behind the paper's CacheBlend numbers, for several requests
# in one forward pass:
#   layers < check_layer   every context token, fresh K/V, causal attention;
#                          the paged cache keeps the loaded K/V (not written),
#                          as in the per-request path
#   check_layer (1)        every token; per request the top `ratio` tokens by
#                          sum((K_fresh - K_loaded)^2) are kept; their K/V are
#                          written into the paged cache, and the kept queries
#                          attend to the request's full K/V (bottom-right
#                          causal alignment, as the per-request path)
#   later layers           kept tokens only, same write + attention
# The loaded KV must already be in vLLM's paged cache with its positions
# re-encoded for the request (the async loader does this).  Where the
# per-request path reads a contiguous loaded-K buffer, this reads the paged
# cache: same values, and no gather.  The question tokens are computed
# afterwards by vLLM's own forward, as in the per-request path.

# Standard
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.v1.compute.models.utils import infer_model_from_vllm

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    from vllm.third_party.flashinfer.flashattn_hopper import flash_attn_varlen_func


def _cumsum0(lengths: List[int], device) -> torch.Tensor:
    out = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    out[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return out


class BatchedBlender:
    """CacheBlend blend step for a batch of requests whose loaded KV is in
    vLLM's paged cache."""

    def __init__(self, vllm_model, ratio: float, check_layer: int = 1):
        self.model = infer_model_from_vllm(vllm_model, self)   # LMCLlamaModel
        if self.model.minimax_qk_norm:
            raise NotImplementedError("batched blend: MiniMax QK-norm not supported")
        self.ratio = ratio
        self.check_layer = check_layer

    # -- paged cache access ----------------------------------------------------
    @staticmethod
    def _slots(kv: torch.Tensor, slot_mapping: torch.Tensor):
        block_size = kv.shape[2]                  # [2, num_blocks, block_size, H, D]
        return slot_mapping // block_size, slot_mapping % block_size

    def _read_k(self, kv, slot_mapping):
        blk, off = self._slots(kv, slot_mapping)
        return kv[0][blk, off]                    # [T, H_kv, D]

    def _write_kv(self, kv, slot_mapping, k, v):
        blk, off = self._slots(kv, slot_mapping)
        kv[0][blk, off] = k
        kv[1][blk, off] = v

    def _attn(self, idx, q, k, v, out, cu_q, max_q, max_k, cu_k=None,
              seqused_k=None, block_table=None):
        """flash_attn_varlen_func with the arguments LMCFlashAttnBackend uses;
        paged when block_table is given (as vLLM's FlashAttentionImpl)."""
        vllm_attn = self.model.vllm_attn_layers[idx]
        impl = vllm_attn.impl
        descale_shape = (cu_q.shape[0] - 1, k.shape[-2])
        flash_attn_varlen_func(
            q=q, k=k, v=v, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=max_q,
            cu_seqlens_k=cu_k, seqused_k=seqused_k, max_seqlen_k=max_k,
            softmax_scale=impl.scale, causal=True,
            alibi_slopes=impl.alibi_slopes, window_size=impl.sliding_window,
            block_table=block_table, softcap=impl.logits_soft_cap,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=vllm_attn._q_scale.expand(descale_shape),
            k_descale=vllm_attn._k_scale.expand(descale_shape),
            v_descale=vllm_attn._v_scale.expand(descale_shape),
        )
        return out

    # -- blend -------------------------------------------------------------------
    @torch.no_grad()
    def blend(
        self,
        token_ids: List[torch.Tensor],
        block_table: torch.Tensor,
        kv_caches: List[torch.Tensor],
        return_hidden: bool = False,
        force_sel: Optional[List[torch.Tensor]] = None,
    ) -> Optional[dict]:
        """
        token_ids:   per request, the context tokens (1-D, int64) whose KV was
                     loaded; positions 0..len-1 of the request.
        block_table: (B, max_blocks) int32, each request's vLLM KV blocks.
        kv_caches:   per layer, vLLM's paged cache [2, num_blocks, block_size, H, D].
        force_sel:   tests only: per request, the kept token indices to use
                     instead of the top-k (isolates rounding from logic).
        """
        m = self.model
        dev = block_table.device
        lens = [int(t.numel()) for t in token_ids]
        B = len(lens)
        cu_k = _cumsum0(lens, dev)
        seqused_k = torch.tensor(lens, dtype=torch.int32, device=dev)
        max_k = max(lens)

        input_ids = torch.cat([t.to(dev) for t in token_ids])
        positions = torch.cat([torch.arange(n, device=dev) for n in lens])
        block_size = kv_caches[0].shape[2]
        row = torch.repeat_interleave(torch.arange(B, device=dev), seqused_k.long())
        slot_mapping = (block_table[row, positions // block_size].long() * block_size
                        + positions % block_size)

        hidden_states = m._embed(input_ids)
        residual = None
        sel = None                       # kept rows (global), after check_layer
        cu_q, max_q = cu_k, max_k

        layers = m.vllm_model.model.layers[
            m.vllm_model.model.start_layer: m.vllm_model.model.end_layer]
        for idx, layer in enumerate(layers):
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(hidden_states)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            num_heads = m.vllm_attn_layers[idx].num_heads
            num_kv_heads = m.vllm_attn_layers[idx].num_kv_heads
            head_size = m.vllm_attn_layers[idx].head_size
            if m.has_qk_norm:
                q = attn.q_norm(q.view(*q.shape[:-1], num_heads, head_size)).view(q.shape)
                k = attn.k_norm(k.view(*k.shape[:-1], num_kv_heads, head_size)).view(k.shape)
            q, k = attn.rotary_emb(positions if sel is None else positions[sel], q, k)
            kv = kv_caches[idx]

            if idx == self.check_layer:
                # CacheBlend token selection (LMCBlender.process_qkv)
                old_k = self._read_k(kv, slot_mapping).reshape(k.shape)
                diff_k = torch.sum((k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1])
                picks = []
                for b in range(B):
                    lo, hi = int(cu_k[b]), int(cu_k[b + 1])
                    if force_sel is not None:
                        picks.append(force_sel[b].to(dev) + lo)
                        continue
                    top = torch.topk(diff_k[lo:hi], k=int((hi - lo) * self.ratio)).indices
                    picks.append(torch.sort(top).values + lo)
                sel = torch.cat(picks)
                q_lens = [int(p.numel()) for p in picks]
                cu_q, max_q = _cumsum0(q_lens, dev), max(q_lens)
                q, k, v, residual = q[sel], k[sel], v[sel], residual[sel]

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            out = torch.empty_like(q)
            if idx < self.check_layer:
                self._attn(idx, q, k, v, out, cu_q, max_q, max_k, cu_k=cu_k)
            else:
                self._write_kv(kv, slot_mapping[sel], k, v)
                self._attn(idx, q, kv[0], kv[1], out, cu_q, max_q, max_k,
                           seqused_k=seqused_k, block_table=block_table)

            hidden_states, _ = attn.o_proj(out.view(-1, num_heads * head_size))
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = getattr(layer, m.mlp_attr)(hidden_states)

        if return_hidden:
            return {"hidden": hidden_states, "residual": residual, "sel": sel,
                    "cu_q": cu_q, "cu_k": cu_k}
        return None
