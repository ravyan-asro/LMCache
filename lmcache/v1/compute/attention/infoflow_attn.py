# Attention of CacheBlend's kept tokens and InfoFlow's recompute pass: the
# selected tokens (sorted request positions) attend to the request's whole
# context; query i sees key j iff j <= pos[i].
#
# Rya 10-20 (later): FlexAttention with a block mask (position_masked.py) for
# both methods, so that at their minimum ratios (CacheBlend 15-20%, InfoFlow
# 20%) the baselines stay bound by the SSD read, not by the attention kernel.
# [SRC] H200, per layer at 15 / 20% (MuSiQue): CacheBlend recompute with
# InfoFlow's kernels 77.5 / 102.8 ms vs SSD read 48.0 ms (Llama 64K), 40.0 /
# 50.2 vs 18.1 (MiniMax 64K), 30.4 / 31.9 vs 12.7 (Qwen3 64K); with FlexAttention
# 21.1 / 26.3, 11.4 / 13.1 (SSD-bound).  Same mask and outputs: every kernel
# within ~2e-3 of fp32 and ~1.5e-3 of each other.
#
# InfoFlow's own kernels stay selectable for comparison (LMCACHE_INFOFLOW_ATTN):
#   sdpa        InfoFlow's Llama path (llm/models/llama/kv_cache/recomputer.py
#               _compute_attention): torch SDPA with a boolean mask, GQA by
#               repeat_interleave of K / V
#   flashinfer  InfoFlow's Qwen path (_compute_attention_flashinfer): FlashInfer
#               single_prefill_with_kv_cache(custom_mask=...)
#   flex        default

# Standard
import os

# Third Party
import torch
import torch.nn.functional as F


def kernel_for(vllm_model) -> str:
    """FlexAttention unless LMCACHE_INFOFLOW_ATTN names another kernel."""
    return os.getenv("LMCACHE_INFOFLOW_ATTN") or "flex"


def infoflow_attention(q, k, v, q_pos, scale, kernel, block_mask=None):
    """q [Lq, H, D] at sorted positions q_pos; k, v [Lk, Hk, D] (the request's
    context, positions 0..Lk-1).  Returns [Lq, H, D].  block_mask (flex):
    position_block_mask(q_pos, Lk), built once per request by the caller."""
    if kernel == "flex":
        # First Party
        from lmcache.v1.compute.attention.position_masked import (
            position_block_mask,
            position_masked_attention,
        )

        if block_mask is None:
            block_mask = position_block_mask(q_pos, k.shape[0])
        return position_masked_attention(q, k, v, q_pos, scale, block_mask=block_mask)
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
