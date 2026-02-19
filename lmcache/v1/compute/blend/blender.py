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

        if self.epic_mode:
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

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

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
                logger.info("GPU recompute layer %d: %.3f ms", i - 1, gpu_elapsed_ms)

            if self.enable_layer_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            next(layerwise_model_executor) # compute layer i (blending + recompute), async
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
        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)
