"""IndexCacheBlender — orchestrates cache gen and prefill for IndexCache.

Cache gen:  runs dense attention on each chunk using vLLM weights,
            stores metadata (kvcolidx + LA_hot_tile_code) per layer.
            No KV is stored anywhere.

Prefill:    runs lean_attn/FA3 on context tokens with merged metadata,
            writes resulting KV to vLLM paged buffer for decode.

For FA3 backend, sparse metadata (sparse_n_indices, sparse_n_offsets,
sparse_n_mask_counts) is pre-computed BEFORE the prefill forward pass
so the conversion cost is not included in TTFT.
"""

import itertools
import time

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
    # FA3 metadata pre-computation (called BEFORE llm.generate())
    # ------------------------------------------------------------------
    def precompute_sparse_metadata(
        self, total_seqlen: int, question_len: int, device: str = "cuda"
    ):
        """Pre-compute FA3 sparse metadata for all layers.

        Call this after all cache_gen() calls and before llm.generate()
        so the conversion cost is excluded from TTFT.

        Args:
            total_seqlen: total number of tokens (context + question)
            question_len: number of question tokens
            device: target device for sparse tensors
        """
        if self.config.attn_backend != "fa3":
            return

        if not self.chunk_metadata:
            logger.warning("precompute_sparse_metadata: no chunk metadata")
            return

        from lmcache.v1.compute.models.indexcache_llama import _get_fa3
        _, build_meta = _get_fa3()

        # Merge metadata (same logic as blend())
        page_size = self.config.page_size
        offset_list = list(
            itertools.accumulate(cl for cl, _, _ in self.chunk_metadata)
        )
        chunk_num_pages = [cl // page_size for cl, _, _ in self.chunk_metadata]
        pages_before = [0]
        for np in chunk_num_pages[:-1]:
            pages_before.append(pages_before[-1] + np)

        num_chunks = len(self.chunk_metadata)

        t0 = time.time()
        self._precomputed_sparse_metadata = []

        for layer_idx in range(self.model.num_layers):
            # Merge kvcolidx (exclude last chunk)
            kvcolidx_list = []
            for chunk_idx in range(num_chunks - 1):
                meta = self.chunk_metadata[chunk_idx]
                kv = meta[1][layer_idx].to(device)
                offset = pages_before[chunk_idx]
                if offset > 0:
                    kv = torch.where(kv >= 0, kv + offset, kv)
                kvcolidx_list.append(kv)
            if kvcolidx_list:
                merged_kv = torch.cat(kvcolidx_list, dim=-1)
            else:
                B = self.chunk_metadata[0][1][layer_idx].shape[0]
                H = self.chunk_metadata[0][1][layer_idx].shape[1]
                merged_kv = torch.empty(
                    B, H, 0, dtype=torch.long, device=device
                )

            # Merge hot_tile (all chunks)
            hot_tile_list = [
                meta[2][layer_idx].to(device)
                for meta in self.chunk_metadata
            ]
            merged_ht = torch.cat(hot_tile_list, dim=-1)

            # Convert to FA3 sparse format
            indices, offsets, mask_counts = build_meta(
                seqlen_total=total_seqlen,
                offset_list=offset_list,
                question_len=question_len,
                kvcolidx_cache=merged_kv.contiguous(),
                la_hot_tile_code_cache=merged_ht.contiguous(),
                num_heads_kv=self.model.num_kv_heads,
                kBlockM=128,
                kBlockN=128,
                page_size=page_size,
                device=device,
            )
            self._precomputed_sparse_metadata.append(
                (indices, offsets, mask_counts)
            )

        t1 = time.time()
        logger.info(
            f"FA3 sparse metadata pre-computed for {self.model.num_layers} "
            f"layers in {t1 - t0:.3f}s (excluded from TTFT)"
        )

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

        # Compute question_len: total tokens minus context tokens
        # tokens = full prompt (context + question), offset_list[-1] = context length
        question_len = len(tokens) - offset_list[-1]

        logger.info(
            f"IndexCache blend: {len(tokens)} total tokens "
            f"({len(tokens) - question_len} context + {question_len} question), "
            f"{len(self.chunk_metadata)} chunks, offsets={offset_list}, "
            f"backend={self.config.attn_backend}"
        )

        # --- Use pre-computed FA3 sparse metadata if available ---
        sparse_metadata_caches = getattr(
            self, '_precomputed_sparse_metadata', None
        )
        if self.config.attn_backend == "fa3" and sparse_metadata_caches is None:
            # Fallback: compute in-band (will be part of TTFT)
            logger.warning(
                "FA3 sparse metadata not pre-computed — "
                "call precompute_sparse_metadata() before generate() "
                "to exclude conversion from TTFT"
            )
            from lmcache.v1.compute.models.indexcache_llama import _get_fa3
            _, build_meta = _get_fa3()

            t0 = time.time()
            sparse_metadata_caches = []
            for layer_idx in range(self.model.num_layers):
                kvcolidx = kvcolidx_caches[layer_idx].contiguous()
                la_hot_tile = hot_tile_caches[layer_idx].contiguous()

                indices, offsets, mask_counts = build_meta(
                    seqlen_total=len(tokens),
                    offset_list=offset_list,
                    question_len=question_len,
                    kvcolidx_cache=kvcolidx,
                    la_hot_tile_code_cache=la_hot_tile,
                    num_heads_kv=self.model.num_kv_heads,
                    kBlockM=128,
                    kBlockN=128,
                    page_size=self.config.page_size,
                    device=str(device),
                )
                sparse_metadata_caches.append(
                    (indices, offsets, mask_counts)
                )
            t1 = time.time()
            logger.info(
                f"IndexCache FA3 metadata computed in-band: "
                f"{t1 - t0:.3f}s (included in TTFT)"
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
                attn_backend=self.config.attn_backend,
                question_len=question_len,
                sparse_metadata_caches=sparse_metadata_caches,
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
        self._precomputed_sparse_metadata = None
        logger.info("IndexCache metadata reset")
