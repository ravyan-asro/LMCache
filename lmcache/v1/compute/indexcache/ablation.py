"""SIFT bit-vector ablations and static baselines on the sparse FA3 path (Rya 10-27).

LMCACHE_SIFT_ABLATION changes each layer's FA3 block lists -- per (query head, 128-row
block) the key blocks to compute, built from SIFT's metadata by indexcache_inflate --
before the sparse kernel runs.  Question rows stay dense (causal) in every mode; for
document rows:
  none              SIFT as is
  no_ca             SIFT's local tiles; of earlier documents only key block 0 (the global
                    sink): does cross-attention (CA) matter?
  dense_la          every causal tile of the row's own document, SIFT's CA pages: does
                    sparsifying local attention (LA) cost accuracy?
  la_window         cheap LA: each document's first block + the last w own-document blocks
                    (w per layer, SIFT's local block count); SIFT's CA
  ca_static         cheap CA: SIFT's LA; per row the first b earlier-document blocks in a
                    fixed order -- the global sink, each earlier document's first block
                    (nearest document first), then the blocks just before the document
                    (b per layer, SIFT's cross block count)
  ca_topk           runtime CA: SIFT's LA; per head the earlier-document blocks with the
                    highest pooled q.k over all rows, SIFT's cross block count
  sink_window       generic static mask (HPCA review): key block 0 + the last w blocks,
                    one w per layer so the layer computes as many document-row blocks
                    as SIFT's
  sink_window_topk  key block 0 + the last w blocks (w = SIFT's mean LA blocks per row,
                    rounded down) + top-k earlier blocks by pooled q.k (mean query of the
                    row block . mean key of the key block, the query head's KV head): per
                    head, the highest-scoring candidates over all rows until the layer
                    computes SIFT's block count; the selection's runtime is not modelled
Blocks are 128 x 128 and documents are page (128) aligned, so a row block lies in one
document.  LMCACHE_SIFT_ABLATION_STATS (a file): one JSON line per prefill, blocks kept
summed over layers and query heads.
"""

# Standard
import json
import os

# Third Party
import torch

MODES = ("none", "no_ca", "dense_la", "la_window", "ca_static", "ca_topk", "sink_window", "sink_window_topk")
_BLOCK = 128
_acc = {}


def _cfg():
    """(mode, stats file), read when used: the driver sets them after LMCache is imported."""
    mode = os.environ.get("LMCACHE_SIFT_ABLATION", "none")
    assert mode in MODES, f"LMCACHE_SIFT_ABLATION={mode} not in {MODES}"
    return mode, os.environ.get("LMCACHE_SIFT_ABLATION_STATS", "")


def active():
    mode, stats = _cfg()
    return mode != "none" or bool(stats)


def _to_dense(idx, off, H, M, N):
    counts = (off[1:] - off[:-1]).long()
    rows = torch.repeat_interleave(torch.arange(H * M, device=idx.device), counts)
    dense = torch.zeros(H * M, N, dtype=torch.bool, device=idx.device)
    dense[rows, idx.long()] = True
    return dense.view(H, M, N)


def _from_dense(dense):
    """FA3's lists: per (head, row block), head-major, key blocks descending; mask
    count = 1 when the diagonal block (the only one needing the causal mask when
    seqlen_q == seqlen_k) is kept."""
    H, M, N = dense.shape
    flat = dense.reshape(H * M, N)
    _, c = flat.flip(-1).nonzero(as_tuple=True)
    idx = (N - 1 - c).to(torch.int32)
    off = torch.zeros(H * M + 1, dtype=torch.int32, device=dense.device)
    off[1:] = torch.cumsum(flat.sum(-1), 0)
    m = torch.arange(M, device=dense.device).clamp(max=N - 1).repeat(H)
    masks = flat[torch.arange(H * M, device=dense.device), m].to(torch.int32)
    return idx, off, masks


def _regions(offset_list, M, N, dev):
    """(la, ca, doc_rows, row_start, starts): own-document causal tiles, earlier-document
    tiles (M, N), which row blocks are document rows (M,), each row's document's first
    block (M,), the documents' first blocks."""
    starts = torch.tensor([0] + list(offset_list[:-1]), device=dev) // _BLOCK
    ctx = offset_list[-1] // _BLOCK
    m = torch.arange(M, device=dev)
    n = torch.arange(N, device=dev)
    doc_rows = m < ctx
    row_start = starts[torch.searchsorted(starts, m.clamp(max=ctx - 1), right=True) - 1]
    causal = n[None, :] <= m[:, None]
    la = doc_rows[:, None] & causal & (n[None, :] >= row_start[:, None])
    ca = doc_rows[:, None] & (n[None, :] < row_start[:, None])
    return la, ca, doc_rows, row_start, starts


def _closest(count, target, hi):
    """The b in 1..hi whose count(b) (non-decreasing) is closest to target."""
    best = None
    for b in range(1, hi + 1):
        c = count(b)
        if best is None or abs(c - target) < abs(best[1] - target):
            best = (b, c)
        if c >= target:
            break
    return best[0]


def _topk_over_rows(score, cand, extra, H, M, N):
    """Per head the `extra` highest-scoring candidate blocks over all its rows."""
    score = score.masked_fill(~cand[None], float("-inf"))
    val, top = score.view(H, M * N).topk(min(extra, M * N), dim=-1)
    sel = torch.zeros(H, M * N, dtype=torch.bool, device=score.device)
    sel.scatter_(-1, top, torch.isfinite(val))
    return sel.view(H, M, N)


def _window(M, N, w, doc_rows, dev):
    m = torch.arange(M, device=dev)[:, None]
    n = torch.arange(N, device=dev)[None, :]
    keep = ((n <= m) & (n > m - w)) | (n == 0)
    return keep & doc_rows[:, None]


def _pooled(x, nb):
    """(S, H, D) -> (H, nb, D): mean over each 128-row block (the last one partial)."""
    S, H, D = x.shape
    pad = nb * _BLOCK - S
    xs = torch.nn.functional.pad(x.float(), (0, 0, 0, 0, 0, pad)).view(nb, _BLOCK, H, D).sum(1)
    cnt = torch.full((nb,), float(_BLOCK), device=x.device)
    cnt[-1] = _BLOCK - pad
    return (xs / cnt[:, None, None]).transpose(0, 1)


def apply(layer_idx, num_layers, idx, off, masks, seq_len, offset_list, q, k):
    """FA3 lists (idx, off, masks) -> the mode's lists.  q: (1, S, H, D), k: (1, S, Hkv, D)
    after RoPE."""
    MODE, STATS = _cfg()
    H = q.shape[2]
    M = N = -(-seq_len // _BLOCK)
    dev = q.device
    sift = _to_dense(idx, off, H, M, N)
    la, ca, doc_rows, row_start, starts = _regions(offset_list, M, N, dev)
    q_rows = ~doc_rows
    n = torch.arange(N, device=dev)
    budget = (sift & doc_rows[None, :, None]).sum().item() / H          # per head
    if MODE == "none":
        out = sift
    elif MODE == "no_ca":
        out = (sift & ~ca) | (ca & (n == 0))[None]
    elif MODE == "dense_la":
        out = sift | la[None]
    elif MODE == "la_window":
        m_ = torch.arange(M, device=dev)[:, None]
        band = lambda w: la & ((n[None, :] > m_ - w) | (n[None, :] == row_start[:, None]))
        w = _closest(lambda w: band(w).sum().item(), (sift & la[None]).sum().item() / H, N)
        out = (sift & ~la[None]) | band(w)[None]
    elif MODE == "ca_static":
        blk_doc = torch.searchsorted(starts, n, right=True) - 1
        row_doc = torch.searchsorted(starts, torch.arange(M, device=dev), right=True) - 1
        is_start = torch.zeros(N, dtype=torch.bool, device=dev)
        is_start[starts] = True
        rank = torch.where(n[None, :] == 0, torch.zeros((), device=dev),                 # global sink
               torch.where(is_start[None, :], 1.0 + (row_doc[:, None] - blk_doc[None, :] - 1),  # document sinks
                           1000.0 + (row_start[:, None] - 1 - n[None, :])))              # most recent first
        rank = rank.masked_fill(~ca, float("inf"))
        pos = rank.argsort(dim=-1, stable=True).argsort(dim=-1)
        b = _closest(lambda b: (ca & (pos < b)).sum().item(), (sift & ca[None]).sum().item() / H, N)
        out = (sift & ~ca[None]) | (ca & (pos < b))[None]
    elif MODE == "ca_topk":
        qp = _pooled(q[0], M)
        kp = _pooled(k[0], N).repeat_interleave(H // k.shape[2], 0)
        extra = int(round((sift & ca[None]).sum().item() / H))
        out = sift & ~ca[None]
        if extra > 0:
            out = out | _topk_over_rows(torch.einsum("hmd,hnd->hmn", qp, kp), ca, extra, H, M, N)
    elif MODE == "sink_window":
        best = None
        for w in range(1, N + 1):
            tot = _window(M, N, w, doc_rows, dev).sum().item()
            if best is None or abs(tot - budget) < abs(best[1] - budget):
                best = (w, tot)
            if tot >= budget:
                break
        out = _window(M, N, best[0], doc_rows, dev)[None].expand(H, M, N) | (sift & q_rows[None, :, None])
    else:                                                                # sink_window_topk
        rows_n = max(1, doc_rows.sum().item())
        w = max(1, int((sift & la[None]).sum().item() / H / rows_n))
        win = _window(M, N, w, doc_rows, dev)
        out = win[None].expand(H, M, N).clone()
        extra = int(round(budget - win.sum().item()))                    # per head
        if extra > 0:
            m_ = torch.arange(M, device=dev)[:, None]
            cand = doc_rows[:, None] & (n[None, :] < m_ - w + 1) & (n[None, :] > 0)
            qp = _pooled(q[0], M)                                         # (H, M, D)
            kp = _pooled(k[0], N).repeat_interleave(H // k.shape[2], 0)   # (H, N, D)
            # the head's highest-scoring candidate blocks over all its rows: SIFT's budget exactly
            out |= _topk_over_rows(torch.einsum("hmd,hnd->hmn", qp, kp), cand, extra, H, M, N)
        out |= sift & q_rows[None, :, None]
    if STATS:
        a = _acc.setdefault("cur", {"kept": 0, "la_kept": 0, "ca_kept": 0, "la_total": 0, "ca_total": 0,
                                    "sift_kept": 0, "q_kept": 0})
        d = doc_rows[None, :, None]
        a["kept"] += (out & d).sum().item()
        a["sift_kept"] += (sift & d).sum().item()
        a["la_kept"] += (out & la[None]).sum().item()
        a["ca_kept"] += (out & ca[None]).sum().item()
        a["la_total"] += la.sum().item() * H
        a["ca_total"] += ca.sum().item() * H
        a["q_kept"] += (out & q_rows[None, :, None]).sum().item()
        if layer_idx == num_layers - 1:
            a.update(mode=MODE, seq_len=seq_len, docs=len(offset_list), layers=num_layers, heads=H)
            with open(STATS, "a") as f:
                f.write(json.dumps(a) + "\n")
            _acc.clear()
    if MODE == "none":
        return idx, off, masks
    return _from_dense(out)
