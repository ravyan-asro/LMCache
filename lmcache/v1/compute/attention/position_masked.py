# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention for queries at arbitrary sorted positions over a whole context:
query i sees key k iff k <= q_pos[i].  CacheBlend's kept tokens and InfoFlow's
recompute pass used it until 10-20; both now run InfoFlow's own kernels
(infoflow_attn.py), this only with LMCACHE_CACHEBLEND_VARIANT=lmcache or
LMCACHE_INFOFLOW_ATTN=flex.

FlashAttention's bottom-right causal alignment is only right for a contiguous
query suffix, so this used FlashInfer's dense custom mask: a (Lq, Lk) bool mask
packed every layer and every tile computed.  That took 34 / 77 / 135 ms per
layer at 20 / 50 / 90% of a 32K context (Llama-3.1-8B, H200) and failed on some
shapes (packbits "invalid configuration argument").  FlexAttention with a block
mask built once per request skips fully masked tiles and needs no mask inside
fully visible ones: 5.5 / 11 / 20 ms per layer, same mask.
"""

# Third Party
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_BLOCK = 128

_flex = None


def _compiled_flex():
    # dynamic=True: one compile (~3 s, in the warm-up request), no recompile
    # when the query / context lengths change between requests.
    global _flex
    if _flex is None:
        _flex = torch.compile(flex_attention, dynamic=True)
    return _flex


def position_block_mask(q_pos: torch.Tensor, kv_len: int, block: int = _BLOCK):
    """Block mask for query positions `q_pos` (sorted) over keys 0..kv_len-1;
    None when there are no queries.

    Built from the positions instead of create_block_mask, which evaluates the
    mask densely (Lq x Lk index tensors: 28 GiB at 61K x 61K, OOM in the 64K
    CacheBlend sweep at ratio 1.0, 10-11).  Queries are sorted, so for query
    block i and key block j: every key of j visible to every row iff
    last_key(j) <= min position(i) (full block, no mask applied); some visible
    iff first_key(j) <= max position(i) (partial block, mask_mod applied);
    else skipped.  Same blocks and bit-identical outputs as create_block_mask
    (checked 10-11)."""
    if q_pos.numel() == 0:
        return None
    lq, dev = q_pos.numel(), q_pos.device
    nqb, nkb = -(-lq // block), -(-kv_len // block)
    pad = nqb * block - lq
    q_min = torch.nn.functional.pad(q_pos, (0, pad), value=kv_len).view(nqb, block).amin(1)
    q_max = torch.nn.functional.pad(q_pos, (0, pad), value=-1).view(nqb, block).amax(1)
    k_first = torch.arange(nkb, device=dev) * block
    k_last = torch.clamp(k_first + block, max=kv_len) - 1
    full = k_last[None, :] <= q_min[:, None]
    partial = (k_first[None, :] <= q_max[:, None]) & ~full
    if pad:
        # as create_block_mask: padded rows fail the mask, so no block of the
        # last query block is full (keeps outputs bit-identical to it)
        partial[-1] |= full[-1]
        full[-1] = False

    def to_kv_blocks(m):
        num = m.sum(1, dtype=torch.int32)
        idx = torch.argsort((~m).to(torch.int8), dim=1, stable=True).to(torch.int32)
        return num[None, None], idx[None, None]

    def mask_mod(b, h, q_idx, kv_idx):
        return q_pos[q_idx] >= kv_idx

    kv_num, kv_idx = to_kv_blocks(partial)
    full_num, full_idx = to_kv_blocks(full)
    return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=block,
                                    mask_mod=mask_mod, seq_lengths=(lq, kv_len))


def position_masked_attention(q, k, v, q_pos, scale, block_mask=None):
    """q: (Lq, H, D); k, v: (Lk, Hk, D) (NHD, GQA allowed).  Returns (Lq, H, D)."""
    if q.shape[0] == 0:
        # Nothing selected (InfoFlow never selects from the first document, so a
        # one-document context keeps all its loaded KV): FlexAttention cannot
        # compile a zero-length query (2WikiMQA original, 10-11).
        return q.new_empty(q.shape)
    if block_mask is None:
        block_mask = position_block_mask(q_pos, k.shape[0])
    # Contiguous (H, L, D) inputs: the dynamic-shape kernel took 8.7 ms with the
    # strided NHD -> HND views vs 5.4 ms contiguous (20% of 32K, Llama shapes,
    # H200, 10-11); the copies cost ~0.2 ms.
    out = _compiled_flex()(
        q.transpose(0, 1).contiguous()[None], k.transpose(0, 1).contiguous()[None],
        v.transpose(0, 1).contiguous()[None],
        block_mask=block_mask, scale=scale, enable_gqa=True,
    )
    return out[0].transpose(0, 1)
