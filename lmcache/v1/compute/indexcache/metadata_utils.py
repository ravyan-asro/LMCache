"""Standalone metadata-extraction functions for IndexCache.

Extracted from enrichedcaching/model/llama3_prime_lean.py so they can be
called with vLLM model weights without pulling in the full HF model class.
"""

import torch


def find_important_pages_fast(
    attn_scores: torch.Tensor,
    page_size: int,
    num_topk_kvcols: int,
    layer_idx: int,
    misbehaving_heads_set: set,
    imp_threshold: float = 0.01,
    strong_consistency: float = 0.1,
) -> torch.Tensor:
    """Select important KV pages per head based on attention-score consistency.

    Returns a bit-packed uint8 vector where each bit indicates whether
    a page is selected (1) or not (0).  Bits are packed MSB-first within
    each byte, padded to a byte boundary with trailing zeros.

    Args:
        attn_scores: (B, H, Q, K) post-softmax attention scores
        page_size: tokens per page
        num_topk_kvcols: number of page slots (== num_pages)
        layer_idx: current layer index (for misbehaving-head lookup)
        misbehaving_heads_set: set of (layer, head) tuples
        imp_threshold: per-token importance threshold
        strong_consistency: fraction of queries that must be important

    Returns:
        kvcolidx: (B, H, ceil(num_pages/8)) uint8 — bit-packed page selection
    """
    if num_topk_kvcols == 0:
        B, H = attn_scores.shape[:2]
        return torch.empty((B, H, 0), dtype=torch.uint8, device=attn_scores.device)

    B, H, Q, K = attn_scores.shape
    assert Q == K
    assert K % page_size == 0

    num_pages = K // page_size
    assert num_topk_kvcols == num_pages

    # Lower-triangular causal mask
    causal_mask = torch.tril(
        torch.ones((1, 1, K, K), device=attn_scores.device, dtype=torch.bool)
    )

    # Combine threshold + causal in one mask
    important_mask = (attn_scores >= imp_threshold) & causal_mask
    del causal_mask

    # Per-key (column) importance counts
    num_imp = important_mask.sum(dim=-2, dtype=torch.int16)  # (B, H, K)
    del important_mask

    # Column j in lower-triangular has j+1 non-zero entries
    total_causal = torch.arange(1, K + 1, device=attn_scores.device, dtype=torch.float)
    consistency = num_imp.float() / total_causal

    # Strong tokens per key
    strong_tokens = consistency >= strong_consistency     # (B, H, K)

    # Page-level: if any strong token in page → select page
    strong_tokens_pages = strong_tokens.view(B, H, num_pages, page_size).any(dim=-1)
    # (B, H, num_pages) bool

    # Misbehaving heads → select ALL pages
    mis = torch.zeros((B, H), dtype=torch.bool, device=attn_scores.device)
    for (layer, head) in misbehaving_heads_set:
        if layer == layer_idx:
            mis[:, head] = True
    strong_tokens_pages = strong_tokens_pages | mis.unsqueeze(-1)

    # Bit-pack into uint8, MSB-first, padded to byte boundary
    kvcolidx = _pack_bits_msb(strong_tokens_pages)  # (B, H, num_bytes) uint8

    return kvcolidx


def _pack_bits_msb(bools: torch.Tensor) -> torch.Tensor:
    """Pack a boolean tensor's last dimension into uint8 bytes, MSB-first.

    Args:
        bools: (..., N) bool tensor

    Returns:
        packed: (..., ceil(N/8)) uint8 tensor
    """
    N = bools.shape[-1]
    pad = (8 - N % 8) % 8
    if pad > 0:
        padded = torch.nn.functional.pad(bools, (0, pad), value=False)
    else:
        padded = bools
    # Reshape last dim into groups of 8
    shape = padded.shape[:-1] + (padded.shape[-1] // 8, 8)
    reshaped = padded.view(shape).to(torch.uint8)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                           dtype=torch.uint8, device=bools.device)
    return (reshaped * weights).sum(dim=-1, dtype=torch.uint8)


def get_hot_tile_code_fast(
    attn_scores: torch.Tensor,
    page_size: int,
    layer_idx: int,
    misbehaving_heads_set: set,
    if_thresh: bool = True,
    threshold: float = 0.001,
    local_pct: float = 0.5,
) -> torch.Tensor:
    """Compute bit-packed hot-tile codes for lean_attn sparse attention.

    Args:
        attn_scores: (B, H, L, L) post-softmax attention scores
        page_size: tokens per page
        layer_idx: current layer index
        misbehaving_heads_set: set of (layer, head) tuples
        if_thresh: use threshold mode (True) or top-pct mode (False)
        threshold: per-element threshold (if_thresh=True)
        local_pct: fraction of tiles to select (if_thresh=False)

    Returns:
        result: (1, H, num_bytes) uint8 bit-packed hot-tile codes
    """
    B, H, L, _ = attn_scores.shape
    assert B == 1, "Only B=1 supported"
    device = attn_scores.device

    num_pages = L // page_size
    num_tiles = num_pages * (num_pages + 1) // 2
    num_bytes = (num_tiles + 7) // 8

    # Build tile view: (1, H, P, P, ps*ps)
    tiles = attn_scores.view(1, H, num_pages, page_size, num_pages, page_size)
    tiles = (
        tiles.permute(0, 1, 2, 4, 3, 5)
        .reshape(1, H, num_pages, num_pages, page_size * page_size)
    )

    # Misbehaving heads broadcast mask
    mh_mask = torch.tensor(
        [(layer_idx, h) in misbehaving_heads_set for h in range(H)],
        device=device,
    ).view(1, H, 1, 1, 1)

    if if_thresh:
        thresh_tensor = torch.where(
            mh_mask,
            torch.tensor(0.0, device=device),
            torch.tensor(threshold, device=device),
        )
        hot_mask = (tiles >= thresh_tensor).any(dim=-1)
    else:
        tile_scores = tiles.max(dim=-1).values
        causal_mask = torch.tril(
            torch.ones((num_pages, num_pages), device=device, dtype=torch.bool)
        ).view(1, 1, num_pages, num_pages)

        causal_scores_flat = tile_scores[causal_mask.expand(1, H, -1, -1)].view(
            1, H, -1
        )
        num_causal_tiles = causal_scores_flat.shape[-1]
        k = max(1, min(int(num_causal_tiles * local_pct), num_causal_tiles))

        topk_vals, _ = torch.topk(causal_scores_flat, k, dim=-1)
        thresh_vals = topk_vals[..., -1].view(1, H, 1, 1)
        hot_mask = (tile_scores >= thresh_vals) & causal_mask

    # Force streaming tiles (diagonal + first column)
    diag = torch.arange(num_pages, device=device)
    hot_mask[:, :, diag, diag] = True
    hot_mask[:, :, :, 0] = True

    # Flatten lower-triangular tiles → (H, num_tiles)
    i_idx, j_idx = torch.tril_indices(num_pages, num_pages, device=device)
    hot_flat = hot_mask[0, :, i_idx, j_idx]

    # Bit packing (fully vectorized)
    bit_idx = torch.arange(num_tiles, device=device)
    byte_idx = bit_idx // 8
    bit_offset = 7 - (bit_idx % 8)
    bit_mask = (1 << bit_offset).to(torch.uint8)

    bit_mask = bit_mask.unsqueeze(0).expand(H, -1)
    byte_idx = byte_idx.unsqueeze(0).expand(H, -1)

    bits = hot_flat.to(torch.uint8) * bit_mask

    result = torch.zeros((H, num_bytes), dtype=torch.uint8, device=device)
    result.scatter_add_(1, byte_idx, bits)
    result = result.unsqueeze(0)  # (1, H, num_bytes)

    return result
