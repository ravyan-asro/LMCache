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
from typing import TYPE_CHECKING, Dict

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.gpu_connector import GPUConnectorInterface
    from lmcache.v1.compute.indexcache.config import IndexCacheConfig

logger = init_logger(__name__)


class LMCBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        cache_engine: "LMCacheEngine",
        gpu_connector: "GPUConnectorInterface",
    ):
        """
        Get or create a blender for the given instance_id.
        """

        if instance_id not in cls._blenders:
            logger.info(f"Creating blender for {instance_id}")
            vllm_model = VLLMModelTracker.get_model(instance_id)
            blender = LMCBlender(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                vllm_model=vllm_model,
            )
            cls._blenders[instance_id] = blender
        else:
            logger.info(
                f"Blender for {instance_id} already exists, returning the original one."
            )
        return cls._blenders[instance_id]

    @classmethod
    def get(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the blender by instance_id.
        """
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]


class IndexCacheBlenderBuilder:
    """Builder for IndexCacheBlender — parallel to LMCBlenderBuilder."""

    _blenders: Dict[str, "IndexCacheBlender"] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: "IndexCacheConfig",
    ):
        if instance_id not in cls._blenders:
            from lmcache.v1.compute.indexcache.blender import IndexCacheBlender

            logger.info(f"Creating IndexCacheBlender for {instance_id}")
            vllm_model = VLLMModelTracker.get_model(instance_id)
            cls._blenders[instance_id] = IndexCacheBlender(vllm_model, config)
        else:
            logger.info(
                f"IndexCacheBlender for {instance_id} already exists, "
                "returning the original one."
            )
        return cls._blenders[instance_id]

    @classmethod
    def get(cls, instance_id: str):
        if instance_id not in cls._blenders:
            raise ValueError(
                f"IndexCacheBlender for {instance_id} not found."
            )
        return cls._blenders[instance_id]


class IndexCacheSchedulerTracker:
    """Lightweight scheduler-side token tracker for IndexCache with TP>1.

    When TP>1, the IndexCacheBlender lives in worker processes (not the
    scheduler process).  This tracker sits in the scheduler process and
    records which tokens have been cache-gen'd so that
    get_num_new_matched_tokens() can claim the full prompt.

    Populated by the test script (which shares a process with the scheduler
    when using vLLM's LLM API).
    """

    _instances: Dict[str, "IndexCacheSchedulerTracker"] = {}

    def __init__(self):
        self.cached_token_ids: list = []

    def add_chunk(self, chunk_token_ids: list):
        """Record a chunk's token IDs after cache_gen completes."""
        self.cached_token_ids.extend(chunk_token_ids)

    def lookup(self, token_ids) -> int:
        """Same semantics as IndexCacheBlender.lookup().

        If the cached prefix matches, claim the FULL prompt so that
        blend() handles context + question in one sparse-attention pass.
        """
        if not self.cached_token_ids:
            return 0
        n = min(len(token_ids), len(self.cached_token_ids))
        if n == 0:
            return 0
        if hasattr(token_ids, "tolist"):
            query_prefix = token_ids[:n].tolist()
        else:
            query_prefix = list(token_ids[:n])
        if query_prefix == self.cached_token_ids[:n]:
            return len(token_ids)
        return 0

    def reset(self):
        self.cached_token_ids.clear()

    @classmethod
    def get_or_create(cls, instance_id: str) -> "IndexCacheSchedulerTracker":
        if instance_id not in cls._instances:
            cls._instances[instance_id] = cls()
        return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str):
        return cls._instances.get(instance_id)
