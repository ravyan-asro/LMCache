"""IndexCacheBlender — orchestrates cache gen and prefill for IndexCache.

Cache gen:  runs dense attention on each chunk using vLLM weights,
            stores metadata (kvcolidx + LA_hot_tile_code) per layer.
            No KV is stored anywhere.

Prefill:    runs lean_attn on context tokens with merged metadata,
            writes resulting KV to vLLM paged buffer for decode.
"""

import itertools

import torch

from lmcache.logging import init_logger
from lmcache.v1.compute.indexcache.config import IndexCacheConfig
from lmcache.v1.compute.models.indexcache_llama import LMCIndexCacheLlamaModel

logger = init_logger(__name__)


class IndexCacheBlender:
    """Manages IndexCache metadata and drives cache gen / prefill."""

    def __init__(self, vllm_model, config: IndexCacheConfig):
        self.model = LMCIndexCacheLlamaModel(vllm_model, config)
        self.config = config

        # Per-chunk metadata: [(chunk_len, [kvcolidx_L0..Ln], [hot_tile_L0..Ln])]
        self.chunk_metadata: list = []
        # Flat list of cached token IDs (for prefix matching in lookup)
        self.cached_token_ids: list = []

    # ------------------------------------------------------------------
    # Cache Gen
    # ------------------------------------------------------------------
    def cache_gen(self, chunk_token_ids: torch.Tensor):
        """Run dense attention on a single chunk, store metadata.

        Args:
            chunk_token_ids: 1-D token ID tensor for one chunk.
        """
        chunk_len = chunk_token_ids.shape[0]
        logger.info(
            f"IndexCache cache_gen: {chunk_len} tokens "
            f"({chunk_len // self.config.page_size} pages)"
        )

        layer_kvcolidx = []
        layer_hot_tile = []

        with torch.no_grad():
            for kvcolidx, hot_tile in self.model.cache_gen_forward(chunk_token_ids):
                # Move metadata to CPU to save GPU memory
                layer_kvcolidx.append(kvcolidx.cpu())
                layer_hot_tile.append(hot_tile.cpu())

        self.chunk_metadata.append((chunk_len, layer_kvcolidx, layer_hot_tile))
        self.cached_token_ids.extend(chunk_token_ids.tolist())
        logger.info(
            f"IndexCache cache_gen done. Total cached tokens: "
            f"{len(self.cached_token_ids)}"
        )

    # ------------------------------------------------------------------
    # Prefix lookup (for get_num_new_matched_tokens)
    # ------------------------------------------------------------------
    def lookup(self, token_ids: torch.Tensor) -> int:
        """Return how many tokens can be handled via IndexCache prefill.

        When the cached context prefix matches, returns the FULL prompt
        length (context + question) so that vLLM routes all tokens through
        lean_attn in blend().  This matches the non-vLLM behaviour where
        lean_attn processes context + question together in one pass.
        """
        if not self.cached_token_ids:
            return 0

        n = min(len(token_ids), len(self.cached_token_ids))
        if n == 0:
            return 0

        query_prefix = token_ids[:n].tolist()
        if query_prefix == self.cached_token_ids[:n]:
            # Claim the entire prompt so lean_attn handles context+question
            return len(token_ids)
        return 0

    # ------------------------------------------------------------------
    # Prefill (called from start_load_kv)
    # ------------------------------------------------------------------
    def blend(self, tokens: torch.Tensor, mask=None, **kwargs):
        """Run lean_attn prefill on context tokens, write KV to paged buffer.

        Args:
            tokens: 1-D token IDs for the prefix (context tokens only)
            mask: unused (kept for API compatibility with CacheBlend)
            **kwargs: must contain 'kvcaches' and 'slot_mapping'
        """
        kvcaches = kwargs["kvcaches"]
        slot_mapping = kwargs["slot_mapping"]

        if not self.chunk_metadata:
            logger.warning("IndexCacheBlender.blend() called with no metadata")
            return

        # Compute offset_list from chunk lengths
        offset_list = list(
            itertools.accumulate(cl for cl, _, _ in self.chunk_metadata)
        )

        # Compute cumulative page offsets per chunk (for global page indexing)
        page_size = self.config.page_size
        chunk_num_pages = [cl // page_size for cl, _, _ in self.chunk_metadata]
        # pages_before[i] = total pages in chunks 0..i-1
        pages_before = [0]
        for np in chunk_num_pages[:-1]:
            pages_before.append(pages_before[-1] + np)

        # Merge per-chunk metadata across chunks for each layer.
        # IMPORTANT: Exclude the last chunk from kvcolidx (cross-attention
        # metadata). The last chunk's attention is purely local, handled
        # by hot_tile only. Including it causes lean_attn CUDA asserts.
        # This matches the non-vLLM code: schema.py sets num_topk_kvcols=0
        # for the last document.
        num_chunks = len(self.chunk_metadata)
        kvcolidx_caches = []
        hot_tile_caches = []
        device = tokens.device if tokens.is_cuda else "cuda"

        for layer_idx in range(self.model.num_layers):
            # Concatenate kvcolidx across chunks 0..N-2 (exclude last)
            # Offset local page indices → global page indices
            kvcolidx_list = []
            for chunk_idx in range(num_chunks - 1):
                meta = self.chunk_metadata[chunk_idx]
                kv = meta[1][layer_idx].to(device)
                offset = pages_before[chunk_idx]
                if offset > 0:
                    # Preserve -1 sentinel (unused slots), offset valid indices
                    kv = torch.where(kv >= 0, kv + offset, kv)
                kvcolidx_list.append(kv)
            if kvcolidx_list:
                kvcolidx_caches.append(torch.cat(kvcolidx_list, dim=-1))
            else:
                # Edge case: only 1 chunk → empty kvcolidx
                B = self.chunk_metadata[0][1][layer_idx].shape[0]
                H = self.chunk_metadata[0][1][layer_idx].shape[1]
                kvcolidx_caches.append(
                    torch.empty(B, H, 0, dtype=torch.long, device=device)
                )

            # Concatenate hot_tile across chunks: (1, H, total_bytes)
            hot_tile_list = [
                meta[2][layer_idx].to(device)
                for meta in self.chunk_metadata
            ]
            hot_tile_caches.append(torch.cat(hot_tile_list, dim=-1))

        logger.info(
            f"IndexCache blend: {len(tokens)} context tokens, "
            f"{len(self.chunk_metadata)} chunks, offsets={offset_list}"
        )

        with torch.no_grad():
            gen = self.model.compute_layer(
                tokens,
                kvcolidx_caches,
                hot_tile_caches,
                offset_list,
                self.config.page_size,
                kvcaches,
                slot_mapping,
            )
            for _ in range(self.model.num_layers):
                next(gen)

        logger.info("IndexCache blend done — KV written to paged buffer")

    # ------------------------------------------------------------------
    # Reset (between entries)
    # ------------------------------------------------------------------
    def reset(self):
        """Clear stored metadata for the next query."""
        self.chunk_metadata.clear()
        self.cached_token_ids.clear()
        logger.info("IndexCache metadata reset")
