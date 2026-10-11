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

"""Attention for queries at arbitrary sorted positions over a whole context
(InfoFlow's recompute pass): query i sees key k iff k <= q_pos[i].

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
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_flex = None


def _compiled_flex():
    # dynamic=True: one compile (~3 s, in the warm-up request), no recompile
    # when the query / context lengths change between requests.
    global _flex
    if _flex is None:
        _flex = torch.compile(flex_attention, dynamic=True)
    return _flex


def position_block_mask(q_pos: torch.Tensor, kv_len: int):
    """Block mask for query positions `q_pos` (sorted) over keys 0..kv_len-1."""

    def mask_mod(b, h, q_idx, kv_idx):
        return q_pos[q_idx] >= kv_idx

    return create_block_mask(mask_mod, None, None, q_pos.shape[0], kv_len,
                             device=q_pos.device)


def position_masked_attention(q, k, v, q_pos, scale, block_mask=None):
    """q: (Lq, H, D); k, v: (Lk, Hk, D) (NHD, GQA allowed).  Returns (Lq, H, D)."""
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
