# InfoFlow's recompute-pass attention with the kernels of InfoFlow's own code
# (tx467/InfoFlow-KV, ~/baselines/InfoFlow-KV, commit 5cad12d), SIFT MLSys
# [RYA 10-20]: baselines run the mask and kernel their authors use.  CacheBlend's
# kept tokens use it too: InfoFlow's re-implementation of CacheBlend
# (recompute_cacheblend) attends with the same _compute_attention, and
# CacheBlend's own position-correct mask (_make_partial_bias_gqa) is commented
# out in its code.
#
# The selected tokens (sorted request positions) attend to the request's whole
# context; query i sees key j iff j <= pos[i].  InfoFlow builds that boolean
# mask on every call (every layer) and runs:
#   llama    llm/models/llama/kv_cache/recomputer.py, _compute_attention:
#            torch SDPA with attn_mask, GQA by repeat_interleave of K / V
#   qwen     llm/models/qwen/kv_cache/recomputer.py, _compute_attention_flashinfer:
#            FlashInfer single_prefill_with_kv_cache(custom_mask=...), as its
#            configs set (recompute_attention_mode: flashinfer,
#            num_recompute_chunks: 1)
#   MiniMax has no InfoFlow path; it follows Qwen's (as its chunk prefix).
# LMCACHE_INFOFLOW_ATTN=sdpa|flashinfer|flex overrides (flex = our earlier
# FlexAttention port, position_masked.py), for comparisons only.

# Standard
import os

# Third Party
import torch
import torch.nn.functional as F


def kernel_for(vllm_model) -> str:
    override = os.getenv("LMCACHE_INFOFLOW_ATTN")
    if override:
        return override
    return "sdpa" if "Llama" in type(vllm_model).__name__ else "flashinfer"


def infoflow_attention(q, k, v, q_pos, scale, kernel):
    """q [Lq, H, D] at sorted positions q_pos; k, v [Lk, Hk, D] (the request's
    context, positions 0..Lk-1).  Returns [Lq, H, D]."""
    if kernel == "flex":
        # First Party
        from lmcache.v1.compute.attention.position_masked import (
            position_block_mask,
            position_masked_attention,
        )

        return position_masked_attention(q, k, v, q_pos, scale,
                                         block_mask=position_block_mask(q_pos, k.shape[0]))
    mask = q_pos.unsqueeze(1) >= torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    if kernel == "flashinfer":
        # Third Party
        import flashinfer

        return flashinfer.single_prefill_with_kv_cache(
            q.contiguous(), k.contiguous(), v.contiguous(), custom_mask=mask,
            causal=False, kv_layout="NHD", sm_scale=scale)
    if kernel != "sdpa":
        raise ValueError(f"unknown InfoFlow attention kernel {kernel}")
    groups = q.shape[1] // k.shape[1]
    qh = q.permute(1, 0, 2).unsqueeze(0)                                    # [1, H, Lq, D]
    kh = k.permute(1, 0, 2).repeat_interleave(groups, dim=0).unsqueeze(0)  # [1, H, Lk, D]
    vh = v.permute(1, 0, 2).repeat_interleave(groups, dim=0).unsqueeze(0)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None],
                                         is_causal=False, scale=scale)
    return out.squeeze(0).permute(1, 0, 2)
