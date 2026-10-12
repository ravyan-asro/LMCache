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
#                          attend to the request's full K/V with a position-
#                          correct causal mask (position-masked FlexAttention,
#                          as the per-request path; Rya 10-11), or FA's
#                          bottom-right alignment with LMCACHE_BLEND_AUTHORS_MASK=1
#   later layers           kept tokens only, same write + attention
# The loaded KV must already be in vLLM's paged cache with its positions
# re-encoded for the request (the async loader does this).  Where the
# per-request path reads a contiguous loaded-K buffer, this reads the paged
# cache: same values, and no gather.  The question tokens are computed
# afterwards by vLLM's own forward, as in the per-request path.
#
# BatchedBlender.infoflow: InfoFlow KV for several requests, the same algorithm
# as LMCBlender._infoflow_blend (no reorder; the per-request path behind the
# paper's InfoFlow numbers):
#   pass 1   the question tokens only, every layer, attending to the request's
#            loaded context K/V followed by their own fresh K/V (bottom-right
#            causal = exact); at the scoring layers every context token gets
#            the L2 norm over question tokens of the head-mean attention
#   select   top int(ctx * ratio) by score, first document excluded
#   pass 2   the selected tokens only, every layer, from the embeddings; their
#            K/V are written into the paged cache and they attend to the
#            request's context K/V with a position-correct causal mask
#            (position-masked FlexAttention, as LMCache's InfoFlow attention)
# Pass 2 starts from the selected tokens' embeddings; the per-request path
# runs layer 0's QKV projection on every context token and then subsets
# (same result, slightly more work).  Question K/V of pass 1 are not kept:
# vLLM computes the question afterwards from the updated context, as in the
# per-request path.

# Standard
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.v1.compute.attention.position_masked import (
    position_block_mask,
    position_masked_attention,
)
from lmcache.v1.compute.blend.blender import BLEND_AUTHORS_MASK
from lmcache.v1.compute.models.utils import infer_model_from_vllm

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    from vllm.third_party.flashinfer.flashattn_hopper import flash_attn_varlen_func


def _offsets(lengths: List[int]) -> List[int]:
    """Python-side cumulative offsets (0, l0, l0+l1, ...).  Slicing with these
    instead of int(cu[b]) on the GPU tensor avoids a device sync per request
    per layer (~124 per request; each drains the GPU queue, 10-12 audit)."""
    out = [0]
    for n in lengths:
        out.append(out[-1] + int(n))
    return out


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

    def _read_kv(self, kv, slot_mapping):
        blk, off = self._slots(kv, slot_mapping)
        return kv[0][blk, off], kv[1][blk, off]

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
        ok = _offsets(lens)
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
        oq = ok

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
                    lo, hi = ok[b], ok[b + 1]
                    if force_sel is not None:
                        picks.append(force_sel[b].to(dev) + lo)
                        continue
                    top = torch.topk(diff_k[lo:hi], k=int((hi - lo) * self.ratio)).indices
                    picks.append(torch.sort(top).values + lo)
                sel = torch.cat(picks)
                # per request: kept positions and their block mask (built once)
                local = [p - ok[b] for b, p in enumerate(picks)]
                masks = [position_block_mask(local[b], lens[b]) for b in range(B)]
                q_lens = [int(p.numel()) for p in picks]
                cu_q, max_q = _cumsum0(q_lens, dev), max(q_lens)
                oq = _offsets(q_lens)
                q, k, v, residual = q[sel], k[sel], v[sel], residual[sel]

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            out = torch.empty_like(q)
            if idx < self.check_layer:
                self._attn(idx, q, k, v, out, cu_q, max_q, max_k, cu_k=cu_k)
            elif BLEND_AUTHORS_MASK:
                self._write_kv(kv, slot_mapping[sel], k, v)
                self._attn(idx, q, kv[0], kv[1], out, cu_q, max_q, max_k,
                           seqused_k=seqused_k, block_table=block_table)
            else:
                self._write_kv(kv, slot_mapping[sel], k, v)
                scale = m.vllm_attn_layers[idx].impl.scale
                for b in range(B):
                    lo, hi = oq[b], oq[b + 1]
                    ck, cv = self._read_kv(kv, slot_mapping[ok[b]:ok[b + 1]])
                    out[lo:hi] = position_masked_attention(q[lo:hi], ck, cv, local[b], scale,
                                                           block_mask=masks[b])

            hidden_states, _ = attn.o_proj(out.view(-1, num_heads * head_size))
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = getattr(layer, m.mlp_attr)(hidden_states)

        if return_hidden:
            return {"hidden": hidden_states, "residual": residual, "sel": sel,
                    "cu_q": cu_q, "cu_k": cu_k}
        return None

    # -- InfoFlow ----------------------------------------------------------------
    def _layers(self):
        m = self.model
        return m.vllm_model.model.layers[m.vllm_model.model.start_layer: m.vllm_model.model.end_layer]

    def _qkv(self, idx, layer, hidden_states, residual, positions):
        m = self.model
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        attn = layer.self_attn
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        H = m.vllm_attn_layers[idx].num_heads
        Hk = m.vllm_attn_layers[idx].num_kv_heads
        D = m.vllm_attn_layers[idx].head_size
        if m.has_qk_norm:
            q = attn.q_norm(q.view(*q.shape[:-1], H, D)).view(q.shape)
            k = attn.k_norm(k.view(*k.shape[:-1], Hk, D)).view(k.shape)
        q, k = attn.rotary_emb(positions, q, k)
        return q.view(-1, H, D), k.view(-1, Hk, D), v.view(-1, Hk, D), residual, H, Hk, D

    def _finish_layer(self, layer, out, residual, H, D):
        attn = layer.self_attn
        hidden_states, _ = attn.o_proj(out.view(-1, H * D))
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        hidden_states = getattr(layer, self.model.mlp_attr)(hidden_states)
        return hidden_states, residual

    @staticmethod
    def _score(q, k_all, q_pos, ctx_len, H, Hk, D):
        """LMCBlender._infoflow_accumulate for one request: head-mean attention,
        L2 norm over query tokens, context part."""
        qh = q.float()
        kh = k_all.float().repeat_interleave(H // Hk, dim=1)
        logits = torch.einsum("qhd,nhd->hqn", qh, kh) * (D ** -0.5)
        key_pos = torch.arange(kh.shape[0], device=q.device)
        logits.masked_fill_(key_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
        attn_mean = torch.softmax(logits, dim=-1).mean(dim=0)       # [Q, N]
        return attn_mean[:, :ctx_len].norm(p=2, dim=0)

    @torch.no_grad()
    def infoflow(
        self,
        token_ids: List[torch.Tensor],
        query_ids: List[torch.Tensor],
        first_lens: List[int],
        block_table: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ratio: float,
        score_layers: List[int],
        force_sel: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """
        token_ids:   per request, the context tokens whose KV was loaded
                     (positions 0..len-1).
        query_ids:   per request, the question tokens (positions len..).
        first_lens:  per request, the first document's length (excluded from
                     selection, as InfoFlow's official code).
        block_table: (B, max_blocks) int32, each request's context KV blocks.
        force_sel:   tests only: per request, the selected indices to use.
        Returns {"sel": per request selected context indices (sorted),
                 "scores": per request scores}.
        """
        m = self.model
        dev = block_table.device
        B = len(token_ids)
        lens = [int(t.numel()) for t in token_ids]
        qlens = [int(t.numel()) for t in query_ids]
        block_size = kv_caches[0].shape[2]
        layers = self._layers()

        def ctx_slots(b, pos):
            return block_table[b, pos // block_size].long() * block_size + pos % block_size

        all_slots = [ctx_slots(b, torch.arange(n, device=dev)) for b, n in enumerate(lens)]

        def gather(kv, b):
            s = all_slots[b]
            return kv[0][s // block_size, s % block_size], kv[1][s // block_size, s % block_size]

        # ---- pass 1: question tokens, every layer, scores at score_layers ----
        q_pos = [torch.arange(n, n + ql, device=dev) for n, ql in zip(lens, qlens)]
        positions = torch.cat(q_pos)
        cu_q = _cumsum0(qlens, dev)
        oq = _offsets(qlens)
        kv_lens = [n + ql for n, ql in zip(lens, qlens)]
        cu_k = _cumsum0(kv_lens, dev)
        scores = [torch.zeros(n, dtype=torch.float32, device=dev) for n in lens]
        hidden_states = m._embed(torch.cat([t.to(dev) for t in query_ids]))
        residual = None
        for idx, layer in enumerate(layers):
            q, k, v, residual, H, Hk, D = self._qkv(idx, layer, hidden_states, residual, positions)
            ks, vs = [], []
            for b in range(B):
                ck, cv = gather(kv_caches[idx], b)
                lo, hi = oq[b], oq[b + 1]
                ks += [ck, k[lo:hi]]
                vs += [cv, v[lo:hi]]
                if idx in score_layers:
                    scores[b] += self._score(q[lo:hi], torch.cat([ck, k[lo:hi]]), q_pos[b],
                                             lens[b], H, Hk, D)
            out = torch.empty_like(q)
            self._attn(idx, q, torch.cat(ks), torch.cat(vs), out, cu_q, max(qlens),
                       max(kv_lens), cu_k=cu_k)
            hidden_states, residual = self._finish_layer(layer, out, residual, H, D)

        # ---- select ----
        sel = []
        for b in range(B):
            if force_sel is not None:
                sel.append(force_sel[b].to(dev).sort().values)
                continue
            cand = torch.arange(first_lens[b], lens[b], device=dev)
            k_ = min(int(lens[b] * ratio), cand.numel())
            sel.append(cand[torch.topk(scores[b][cand], k=k_).indices].sort().values)

        # ---- pass 2: selected tokens, every layer, position-correct mask ----
        sel_lens = [int(s_.numel()) for s_ in sel]
        if sum(sel_lens) == 0:      # one-document contexts: nothing to recompute
            return {"sel": sel, "scores": scores}
        os_ = _offsets(sel_lens)
        positions = torch.cat(sel)
        sel_slots = torch.cat([all_slots[b][sel[b]] for b in range(B)])
        masks = [position_block_mask(sel[b], lens[b]) for b in range(B)]
        hidden_states = m._embed(torch.cat([token_ids[b].to(dev)[sel[b]] for b in range(B)]))
        residual = None
        for idx, layer in enumerate(layers):
            q, k, v, residual, H, Hk, D = self._qkv(idx, layer, hidden_states, residual, positions)
            self._write_kv(kv_caches[idx], sel_slots, k, v)
            impl = m.vllm_attn_layers[idx].impl
            out = torch.empty_like(q)
            for b in range(B):
                lo, hi = os_[b], os_[b + 1]
                ck, cv = gather(kv_caches[idx], b)
                out[lo:hi] = position_masked_attention(q[lo:hi], ck, cv, sel[b], impl.scale,
                                                       block_mask=masks[b])
            hidden_states, residual = self._finish_layer(layer, out, residual, H, D)
        return {"sel": sel, "scores": scores}
