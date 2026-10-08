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

# Standard
from typing import Optional
import os

# Third Party
import torch
import torch.cuda.nvtx as nvtx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector
        self.enable_layer_timing = (
            os.getenv("LMCACHE_ENABLE_LAYER_TIMING", "0").lower() in {"1", "true"}
        )
        # Structured timing data (populated during blend, read by benchmark)
        self._timing_data = None

        self.layerwise_model = infer_model_from_vllm(vllm_model, self)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # EPIC mode: static first-N-tokens-per-chunk recompute
        self.epic_mode = (
            os.getenv("LMCACHE_EPIC_MODE", "0").lower() in {"1", "true"}
        )
        self.epic_tokens_per_chunk = int(
            os.getenv("LMCACHE_EPIC_TOKENS_PER_CHUNK", "64")
        )

        # InfoFlow KV (ICML'26): select recompute tokens by query->context
        # attention mass at mid/late layers, then recompute them at all layers.
        self.infoflow_mode = (
            os.getenv("LMCACHE_INFOFLOW_MODE", "0").lower() in {"1", "true"}
        )
        self.infoflow_layers = [
            int(x) for x in os.getenv("LMCACHE_INFOFLOW_LAYERS", "22,23,24,25").split(",")
        ]
        self.infoflow_ratio = float(os.getenv("BLEND_RECOMPUTE_RATIO", "0.15"))
        self.infoflow_reorder = (
            os.getenv("LMCACHE_INFOFLOW_REORDER", "0").lower() in {"1", "true"}
        )
        self._if_phase = None  # None | "score" | "recompute"
        self.infoflow_stats = None

        if self.infoflow_mode:
            logger.info(
                "InfoFlow mode enabled: ratio %.3f, scoring layers %s",
                self.infoflow_ratio,
                self.infoflow_layers,
            )
            self.common_metadata = LMCBlendCommonMetadata(
                check_layers=[], recomp_ratios=[], thresholds=None
            )
        elif self.epic_mode:
            logger.info(
                "EPIC mode enabled: static %d tokens/chunk recompute, "
                "partial recompute in ALL layers (including 0 and 1)",
                self.epic_tokens_per_chunk,
            )
            # EPIC uses static selection — no dynamic check layers needed
            self.common_metadata = LMCBlendCommonMetadata(
                check_layers=[],
                recomp_ratios=[],
                thresholds=None,
            )
        else:
            blend_ratio = float(os.getenv("BLEND_RECOMPUTE_RATIO", "0.15"))
            logger.info("CacheBlend recompute ratio: %.4f", blend_ratio)
            self.common_metadata = LMCBlendCommonMetadata(
                check_layers=[1],
                recomp_ratios=[blend_ratio],
                thresholds=None,
            )

        # Optional runtime ratio override: LMCACHE_BLEND_RATIO_FILE holds one
        # float, re-read before every blend, so a ratio sweep can run inside one
        # engine (env changes do not reach the vLLM engine process).
        self.ratio_file = os.getenv("LMCACHE_BLEND_RATIO_FILE")

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

    def _maybe_update_ratio(self):
        if not self.ratio_file or not os.path.exists(self.ratio_file):
            return
        with open(self.ratio_file) as f:
            ratio = float(f.read().strip())
        self.infoflow_ratio = ratio
        if not self.infoflow_mode and not self.epic_mode:
            self.common_metadata.recomp_ratios = [ratio]

    def _compute_epic_indices(self, device: torch.device) -> torch.Tensor:
        """Compute static EPIC indices: first N tokens of each cached chunk."""
        chunk_starts = getattr(self.gpu_connector, "chunk_starts", None)
        chunk_ends = getattr(self.gpu_connector, "chunk_ends", None)
        if not chunk_starts or not chunk_ends:
            return torch.tensor([], device=device, dtype=torch.int64)

        indices = []
        for s, e in zip(chunk_starts, chunk_ends):
            n = min(self.epic_tokens_per_chunk, e - s)
            indices.extend(range(s, s + n))

        return torch.tensor(indices, device=device, dtype=torch.int64)

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        if self._if_phase is not None:
            return self._infoflow_process_qkv(
                q, k, v, residual, layer_id, attn_output, attn_metadata
            )
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        # ---- EPIC: static first-N-per-chunk selection at layer 0 ----
        # Sets imp_indices once; layers 0, 1, 2+ all then follow the
        # same imp_indices path that CacheBlend layers ≥2 already use.
        if self.epic_mode and layer_id == 0:
            top_indices = self._compute_epic_indices(q.device)
            topk_num = top_indices.shape[0]
            logger.info(
                "EPIC layer 0: selecting %d static tokens "
                "(%d tokens/chunk × %d chunks)",
                topk_num,
                self.epic_tokens_per_chunk,
                len(self.gpu_connector.chunk_starts),
            )

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.max_query_len = topk_num
            attn_metadata.query_start_loc = torch.tensor(
                [0, topk_num], dtype=torch.int32, device=q.device
            )

        # ---- Original CacheBlend: dynamic top-k at check_layers ----
        elif layer_id in self.common_metadata.check_layers:
            diff_k = torch.sum(
                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
            )
            total_len = diff_k.shape[0]

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])

            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Picking indices: {top_indices}")
            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.max_query_len = topk_num
            attn_metadata.query_start_loc = torch.tensor(
                [0, topk_num], dtype=torch.int32, device=q.device
            )

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        # Per-layer recompute timing storage
        self._per_layer_recompute_ms = {}

        next(layerwise_retriever) # request layer 1 from storage backend
        yield

        prev_start_evt = None
        prev_end_evt = None

        for i in range(self.num_layers):
            next(layerwise_retriever) # request layer i+1 from storage backend
            if self.enable_layer_timing and prev_start_evt is not None:
                # GPU connector's sync inside send() has already waited for GPU compute
                # of layer i-1, so prev layer's CUDA events are safe to read now.
                prev_end_evt.synchronize()  # wait for end event; no-op if already reached
                gpu_elapsed_ms = prev_start_evt.elapsed_time(prev_end_evt)
                self._per_layer_recompute_ms[i - 1] = gpu_elapsed_ms
                logger.info("GPU recompute layer %d: %.3f ms", i - 1, gpu_elapsed_ms)

            if self.enable_layer_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            nvtx.range_push(f"GPURecompute_L{i}")
            next(layerwise_model_executor) # compute layer i (blending + recompute), async
            nvtx.range_pop()
            if self.enable_layer_timing:
                end_evt.record()
                prev_start_evt = start_evt
                prev_end_evt = end_evt
            yield

        # Final next() triggers the GPU connector sync for the last layer,
        # after which the last layer's events are safe to read.
        next(layerwise_retriever)
        if self.enable_layer_timing and prev_start_evt is not None:
            prev_end_evt.synchronize()  # wait for end event; no-op if already reached
            gpu_elapsed_ms = prev_start_evt.elapsed_time(prev_end_evt)
            self._per_layer_recompute_ms[self.num_layers - 1] = gpu_elapsed_ms
            logger.info("GPU recompute layer %d: %.3f ms", self.num_layers - 1, gpu_elapsed_ms)

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """
        self._maybe_update_ratio()
        if self.infoflow_mode:
            self._infoflow_blend(tokens, mask, **kwargs)
            return

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)

        # Collect structured timing data from components
        if self.enable_layer_timing:
            self._collect_timing_data()

    # ------------------------------------------------------------------
    # InfoFlow KV (Teng et al., ICML'26; arXiv 2603.05353, Sec 4 + App F;
    # official code github.com/tx467/InfoFlow-KV,
    # llm/scripts/inference_with_recompute_kv.py)
    # ------------------------------------------------------------------
    # The blend range is the cached context only; vLLM computes the query
    # afterwards from the paged cache.  The adapter passes `query_tokens`.
    #
    # Without reorder (official make_1_layer_recompute_fn):
    #   stage 1 scores at GLOBAL positions; top-k excludes the first chunk.
    # With reorder, LMCACHE_INFOFLOW_REORDER=1 (official make_double_guided_fn):
    #   stage 1 scores at chunk-LOCAL positions (extract_without_RoPE_correction);
    #   chunks are reordered by their share of selected tokens (ascending, so the
    #   most informative sit next to the query) and their KV is rebased to the
    #   new positions (reorder_and_rebase_kv); stage 2 re-scores at GLOBAL
    #   positions; neither stage excludes the first chunk.  Unit 0 (whatever the
    #   first stored chunk is) stays in front: the official code pins only its
    #   1-token prefix, so the driver can make the template its own unit 0.
    # Both: k = int(context_len * ratio); the selected tokens plus any tokens
    # not covered by a stored chunk (the "# #" separators) are recomputed at ALL
    # layers with a position-aware causal mask.  Everything stays in HBM after
    # the single disk read; all stages are inside the request (TTFT), as in the
    # official code (start_time is taken before extraction).
    def _infoflow_blend(self, tokens, mask=None, query_tokens=None, **kwargs):
        if query_tokens is None or query_tokens.numel() == 0:
            raise ValueError("InfoFlow needs the query tokens (see vllm_v1_adapter)")
        device = torch.device("cuda")
        query_tokens = query_tokens.to(tokens.device)
        ctx_len = tokens.shape[0]
        self._if_ctx_len = ctx_len
        self._if_kvcaches = kwargs["kvcaches"]
        self._if_slot_mapping = kwargs["slot_mapping"].to(device)
        budget = int(ctx_len * self.infoflow_ratio)
        events = {}

        def mark(name):
            events[name] = torch.cuda.Event(enable_timing=True)
            events[name].record()

        mark("start")
        # Stage 1: layer-wise retrieve (context) interleaved with a query-only
        # forward, mirroring blend_layer().
        self._if_phase = "score_retrieve"
        self._if_local_pos = None
        self._if_scores = torch.zeros(ctx_len, dtype=torch.float32, device=device)
        executor = self.layerwise_model.compute_layer(torch.cat([tokens, query_tokens]))
        retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)
        next(retriever)
        if self.infoflow_reorder:
            self._if_local_pos = self._chunk_local_positions(ctx_len, device)
        for _ in range(self.num_layers):
            next(retriever)
            next(executor)
        next(retriever)
        self.metadata.clean()
        mark("stage1")

        starts = list(self.gpu_connector.chunk_starts)
        ends = list(self.gpu_connector.chunk_ends)
        covered = torch.zeros(ctx_len, dtype=torch.bool, device=device)
        for s_, e_ in zip(starts, ends):
            covered[s_:e_] = True
        gaps = torch.nonzero(~covered).flatten()

        chunk_order = None
        if self.infoflow_reorder:
            sel = torch.topk(self._if_scores, k=budget).indices
            perm, chunk_order = self._infoflow_reorder_perm(sel, starts, ends, ctx_len, device)
            self._infoflow_permute_paged(perm)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(ctx_len, device=device)
            gaps = inv[gaps]
            tokens = tokens[perm.to(tokens.device)]
            mark("reorder")
            # Stage 2: GLOBAL positions on the reordered layout, KV from HBM.
            self._if_phase = "score_paged"
            self._if_scores = torch.zeros(ctx_len, dtype=torch.float32, device=device)
            for _ in self.layerwise_model.compute_layer(torch.cat([tokens, query_tokens])):
                pass
            self.metadata.clean()
            sel = torch.topk(self._if_scores, k=budget).indices
            mark("stage2")
        else:
            candidates = torch.arange(ends[0], ctx_len, device=device)
            k = min(budget, candidates.numel())
            sel = candidates[torch.topk(self._if_scores[candidates], k=k).indices]

        self._if_imp = torch.unique(torch.cat([sel, gaps]))  # sorted
        self._if_phase = "recompute"
        for _ in self.layerwise_model.compute_layer(tokens):
            pass
        mark("end")
        events["end"].synchronize()

        names = list(events)
        self.infoflow_stats = {
            "context_tokens": ctx_len,
            "query_tokens": int(query_tokens.numel()),
            "num_chunks": len(starts),
            "first_chunk_tokens": ends[0],
            "selected_tokens": int(sel.numel()),
            "gap_tokens": int(gaps.numel()),
            "recomputed_context_tokens": int(self._if_imp.numel()),
            "reorder": bool(self.infoflow_reorder),
            "chunk_order": chunk_order,
            **{f"{b}_ms": events[a].elapsed_time(events[b]) for a, b in zip(names, names[1:])},
        }
        logger.info("InfoFlow blend: %s", self.infoflow_stats)
        # Per-layer disk / H2D / RoPE timing of the stage-1 retrieve (same
        # collector and /tmp/blend_timing.json file as CacheBlend's breakdown).
        if self.enable_layer_timing:
            self._collect_timing_data()
        stats_path = os.getenv("LMCACHE_INFOFLOW_STATS_PATH")
        if stats_path:  # read back by the experiment driver (engine runs in a subprocess)
            import json

            with open(stats_path, "a") as f:
                f.write(json.dumps(self.infoflow_stats) + "\n")
        self.metadata.clean()
        self._if_phase = None
        self._if_scores = None
        self._if_local_pos = None

    def _chunk_local_positions(self, ctx_len, device):
        """Position of every context token inside its stored chunk (HL geometry)."""
        local = torch.arange(ctx_len, device=device)
        for s_, e_ in zip(self.gpu_connector.chunk_starts, self.gpu_connector.chunk_ends):
            local[s_:e_] -= s_
        return local

    def _infoflow_reorder_perm(self, sel, starts, ends, ctx_len, device):
        """Permutation of context tokens after reordering stored chunks.

        Each unit is a stored chunk plus any trailing uncovered tokens
        (separator); a leading uncovered region joins unit 0.  Unit 0 stays
        first; the rest are sorted by (ratio, index) ascending, as in the
        official reorder_and_rebase_kv(put_higher_ratio_to_tail=True).
        """
        n = len(starts)
        bounds = [0] + list(starts[1:]) + [ctx_len]
        counts = torch.bincount(
            torch.bucketize(sel, torch.tensor(ends, device=device), right=True),
            minlength=n + 1,
        )[:n].tolist()
        ratios = [counts[i] / max(1, ends[i] - starts[i]) for i in range(n)]
        order = [0] + sorted(range(1, n), key=lambda i: (ratios[i], i))
        perm = torch.cat([torch.arange(bounds[i], bounds[i + 1], device=device) for i in order])
        return perm, order

    def _infoflow_permute_paged(self, perm):
        """Reorder the context KV in the paged cache and rebase key RoPE."""
        ctx_len = perm.numel()
        new_pos = torch.arange(ctx_len, device=perm.device)
        rope = self.layerwise_model.fused_rotary_emb
        for layer_id in range(self.num_layers):
            k, v = self._paged_get_kv(layer_id)
            k = rope(perm, new_pos, k[perm].contiguous())
            self._paged_put_kv(layer_id, new_pos, k, v[perm].contiguous())

    def _infoflow_process_qkv(
        self, q, k, v, residual, layer_id, attn_output, attn_metadata
    ):
        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        attn_layer = self.layerwise_model.vllm_model.model.layers[layer_id].self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)
        ctx_len = self._if_ctx_len
        scoring = self._if_phase in ("score_retrieve", "score_paged")

        if layer_id == 0:
            if scoring:
                idx = torch.arange(ctx_len, q.shape[0], device=q.device)
            else:
                idx = self._if_imp
            num = idx.shape[0]
            q, k, v = q[idx], k[idx], v[idx]
            residual = residual[idx]
            self.metadata.imp_indices = idx
            self.metadata.positions = self.metadata.positions[idx]
            attn_output = attn_output[:num]
            attn_metadata.max_query_len = num
            attn_metadata.query_start_loc = torch.tensor(
                [0, num], dtype=torch.int32, device=q.device
            )

        if scoring:
            # Context KV (retrieval buffer or HBM paged cache), query KV appended
            # as a suffix, so FA's bottom-right causal mask is exact.
            if self._if_phase == "score_retrieve":
                old_k, old_v = self.gpu_connector.get_kv(layer_id)
                old_k, old_v = old_k[:ctx_len], old_v[:ctx_len]
                if self._if_local_pos is not None:  # stage 1 of +Reorder: HL geometry
                    old_k = self.layerwise_model.fused_rotary_emb(
                        torch.arange(ctx_len, device=q.device), self._if_local_pos,
                        old_k.contiguous(),
                    )
            else:
                old_k, old_v = self._paged_get_kv(layer_id)
            k_all = torch.cat([old_k, k])
            v_all = torch.cat([old_v, v])
            if layer_id in self.infoflow_layers:
                self._infoflow_accumulate(q, k_all, layer_id)
            return q, k_all, v_all, residual, attn_output, attn_metadata

        old_k, old_v = self._paged_get_kv(layer_id)
        idx = self.metadata.imp_indices
        old_k[idx] = k
        old_v[idx] = v
        self._paged_put_kv(layer_id, idx, k, v)
        attn_metadata.query_positions = self.metadata.positions
        return q, old_k, old_v, residual, attn_output, attn_metadata

    def _infoflow_accumulate(self, q, k_all, layer_id):
        """Add this layer's InfoFlow "norm" score to every token's total.

        Follows the official code (tx467/InfoFlow-KV,
        llm/models/llama/kv_cache/importance_scorer.py::_compute_norm): mean of
        the attention over heads, then the L2 norm over query tokens, summed over
        the scoring layers.  (The paper's Eq. 7 states a plain column sum; the
        released code, which produced their results, uses this L2 form.)
        """
        attn = self.layerwise_model.vllm_attn_layers[layer_id]
        num_heads, num_kv, head = attn.num_heads, attn.num_kv_heads, attn.head_size
        qh = q.view(-1, num_heads, head).float()
        kh = k_all.view(-1, num_kv, head).float()
        kh = kh.repeat_interleave(num_heads // num_kv, dim=1)
        logits = torch.einsum("qhd,nhd->hqn", qh, kh) * (head ** -0.5)
        q_pos = self.metadata.positions
        key_pos = torch.arange(kh.shape[0], device=q.device)
        logits.masked_fill_(key_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
        attn_mean = torch.softmax(logits, dim=-1).mean(dim=0)  # [Q, N]
        self._if_scores += attn_mean[:, : self._if_ctx_len].norm(p=2, dim=0)

    def _paged_slots(self, layer_id, slots):
        kv = self._if_kvcaches[layer_id]
        block_size = kv.shape[2]
        return kv, slots // block_size, slots % block_size

    def _paged_get_kv(self, layer_id):
        """Gather this layer's [num_tokens, kv_dim] K/V from vLLM's paged cache."""
        kv, blk, off = self._paged_slots(layer_id, self._if_slot_mapping)
        if kv.shape[0] == 2:      # [2, num_blocks, block_size, heads, dim]
            k, v = kv[0][blk, off], kv[1][blk, off]
        else:                     # [num_blocks, 2, block_size, heads, dim]
            k, v = kv[blk, 0, off], kv[blk, 1, off]
        return k.flatten(1).clone(), v.flatten(1).clone()

    def _paged_put_kv(self, layer_id, idx, k, v):
        """Scatter recomputed K/V rows (token indices `idx`) into the paged cache."""
        kv, blk, off = self._paged_slots(layer_id, self._if_slot_mapping[idx])
        if kv.shape[0] == 2:
            shape = kv[0][blk, off].shape
            kv[0][blk, off] = k.view(shape)
            kv[1][blk, off] = v.view(shape)
        else:
            shape = kv[blk, 0, off].shape
            kv[blk, 0, off] = k.view(shape)
            kv[blk, 1, off] = v.view(shape)

    def _collect_timing_data(self):
        """Gather per-layer timing from disk backend, gpu connector, and recompute events."""
        disk_data = {}
        h2d_data = {}
        rope_data = {}
        recompute_data = {}

        # Disk read timing from backend
        try:
            storage_mgr = self.cache_engine.storage_manager
            for backend in storage_mgr.storage_backends.values():
                if hasattr(backend, "get_layer_timing_data"):
                    disk_data = backend.get_layer_timing_data()
                    break
        except Exception:
            pass

        # H2D timing from gpu connector
        try:
            if hasattr(self.gpu_connector, "get_h2d_layer_timing_data"):
                h2d_data = self.gpu_connector.get_h2d_layer_timing_data()
        except Exception:
            pass

        # Per-layer RoPE timing from gpu connector (fused_rotary_emb over
        # all context tokens that runs on the compute stream before the
        # layer's attention+MLP recompute).
        try:
            if hasattr(self.gpu_connector, "get_rope_layer_timing_data"):
                rope_data = self.gpu_connector.get_rope_layer_timing_data()
        except Exception:
            pass

        # Raw recompute timing from blend_layer (CUDA events around generator next())
        raw_recompute = getattr(self, "_per_layer_recompute_ms", {})

        # Fold RoPE into per_layer_recompute so the "per-layer compute" bar in
        # timing plots reflects all compute-stream work (RoPE + attention + MLP),
        # not just the attention/MLP part. RoPE is measured independently for
        # debug visibility via per_layer_rope.
        recompute_data = {}
        all_layers = set(raw_recompute.keys()) | set(rope_data.keys())
        for li in all_layers:
            recompute_data[li] = raw_recompute.get(li, 0.0) + rope_data.get(li, 0.0)

        # Forward timing from inside model (CUDA events around actual kernels)
        forward_data = {}
        try:
            if hasattr(self.layerwise_model, "_per_layer_forward_ms"):
                forward_data = self.layerwise_model._per_layer_forward_ms
        except Exception:
            pass

        self._timing_data = {
            "num_layers": self.num_layers,
            "per_layer_disk_read": disk_data,
            "per_layer_h2d": h2d_data,
            "per_layer_recompute": recompute_data,       # includes RoPE
            "per_layer_recompute_raw": raw_recompute,    # attention/MLP only
            "per_layer_rope": rope_data,
            "per_layer_forward": forward_data,
        }

        # For TP>1: save timing data to a temp file so the main process
        # (which can't access worker-side blender objects) can read it.
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        if rank == 0:
            import json as _json
            timing_path = os.getenv("LMCACHE_BLEND_TIMING_PATH", "/tmp/blend_timing.json")
            try:
                # Convert any non-serializable keys (int layer ids) to strings
                serializable = {}
                for k, v in self._timing_data.items():
                    if isinstance(v, dict):
                        serializable[k] = {str(kk): vv for kk, vv in v.items()}
                    else:
                        serializable[k] = v
                with open(timing_path, "w") as f:
                    _json.dump(serializable, f)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to save blend timing to {timing_path}: {e}"
                )
