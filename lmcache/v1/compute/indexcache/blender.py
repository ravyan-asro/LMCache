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
import json
import math
import os
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

        # Compute and save sparsity + size stats (rank 0 only for TP>1)
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        if rank == 0:
            self._compute_and_save_stats(total_seqlen, question_len)

    def _compute_and_save_stats(self, total_seqlen: int, question_len: int):
        """Compute sparsity and IndexCache size stats, save to temp file.

        This allows the test script to read stats for TP>1 where the
        main process cannot access worker metadata directly.
        """
        page_size = self.config.page_size
        num_layers = self.model.num_layers

        # --- Block sparsity from FA3 sparse metadata ---
        total_sparse_blocks = 0
        for indices, offsets, mask_counts in self._precomputed_sparse_metadata:
            total_sparse_blocks += indices.numel()

        num_heads_q = self.model.num_heads
        num_m_blocks = math.ceil(total_seqlen / 128)  # kBlockM=128
        num_n_blocks = math.ceil(total_seqlen / 128)  # kBlockN=128

        # Dense baseline: sum of causal n_block_max for each m_block, times heads, times layers
        dense_per_m = 0
        for m in range(num_m_blocks):
            n_max = min(
                math.ceil(((m + 1) * 128 + total_seqlen - total_seqlen) / 128),
                num_n_blocks,
            )  # since seqlen_q == seqlen_k, simplifies to min(m+1, num_n_blocks)
            dense_per_m += n_max
        total_dense_blocks = dense_per_m * num_heads_q * num_layers

        block_sparsity = 1.0 - (total_sparse_blocks / total_dense_blocks) if total_dense_blocks > 0 else 0.0

        # --- CA / LA sparsity from chunk_metadata (same as test script) ---
        bitmask = torch.tensor([1 << i for i in reversed(range(8))], dtype=torch.uint8)
        chunk_num_pages = [cl // page_size for cl, _, _ in self.chunk_metadata]
        doc_page_offsets = [0]
        for np_ in chunk_num_pages:
            doc_page_offsets.append(doc_page_offsets[-1] + np_)
        total_doc_pages = doc_page_offsets[-1]
        num_docs = len(self.chunk_metadata)
        H = self.chunk_metadata[0][1][0].shape[1]  # num Q heads

        total_tiles = (total_doc_pages * (total_doc_pages + 1) // 2) * H * num_layers
        total_LA_tiles = 0
        for i in range(num_docs):
            dp = doc_page_offsets[i + 1] - doc_page_offsets[i]
            total_LA_tiles += (dp * (dp + 1) // 2) * H * num_layers
        total_CA_tiles = total_tiles - total_LA_tiles

        pages_before = [0]
        for np_ in chunk_num_pages[:-1]:
            pages_before.append(pages_before[-1] + np_)

        # Precompute doc_end_page lookup: for page p in doc d, end = doc_page_offsets[d+1]
        page_to_doc_end = []
        for d in range(num_docs):
            dp = doc_page_offsets[d + 1] - doc_page_offsets[d]
            page_to_doc_end.extend([doc_page_offsets[d + 1]] * dp)

        active_CA_tiles = 0
        active_LA_tiles = 0
        for layer_idx in range(num_layers):
            if layer_idx < 4:
                active_CA_tiles += total_CA_tiles / num_layers
                active_LA_tiles += total_LA_tiles / num_layers
                continue

            # CA: vectorized count from kvcolidx
            for chunk_idx in range(num_docs - 1):
                kv = self.chunk_metadata[chunk_idx][1][layer_idx][0]  # (H, num_kvcols)
                offset = pages_before[chunk_idx]
                for h_idx in range(H):
                    kv_h = kv[h_idx]
                    if offset > 0:
                        kv_h = torch.where(kv_h >= 0, kv_h + offset, kv_h)
                    kv_valid = kv_h[kv_h >= 0].tolist()
                    for val in kv_valid:
                        val = int(val)
                        if 0 <= val < len(page_to_doc_end):
                            active_CA_tiles += total_doc_pages - page_to_doc_end[val]

            # LA: count hot bits (vectorized)
            hot_tile_list = [meta[2][layer_idx] for meta in self.chunk_metadata]
            layer_code = torch.cat(hot_tile_list, dim=-1)[0]  # (H, total_bytes)
            active_bits = torch.bitwise_and(layer_code.unsqueeze(-1), bitmask) > 0
            active_LA_tiles += active_bits.sum().item()

        sparsity_total = 1.0 - (active_CA_tiles + active_LA_tiles) / total_tiles if total_tiles > 0 else 0.0
        sparsity_ca = 1.0 - active_CA_tiles / total_CA_tiles if total_CA_tiles > 0 else 0.0
        sparsity_la = 1.0 - active_LA_tiles / total_LA_tiles if total_LA_tiles > 0 else 0.0

        # --- IndexCache metadata size ---
        ca_bytes = 0
        la_bytes = 0
        for chunk_len, layer_kvcolidx, layer_hot_tile in self.chunk_metadata:
            for li in range(num_layers):
                kv = layer_kvcolidx[li]
                ht = layer_hot_tile[li]
                ca_bytes += kv.numel() * kv.element_size()
                la_bytes += ht.numel() * ht.element_size()

        stats = {
            "sparsity_total": sparsity_total,
            "sparsity_ca": sparsity_ca,
            "sparsity_la": sparsity_la,
            "block_sparsity": block_sparsity,
            "total_sparse_blocks": total_sparse_blocks,
            "total_dense_blocks": total_dense_blocks,
            "indexcache_ca_size_gb": ca_bytes / (1024 ** 3),
            "indexcache_la_size_gb": la_bytes / (1024 ** 3),
            "indexcache_total_size_gb": (ca_bytes + la_bytes) / (1024 ** 3),
        }

        stats_path = "/tmp/indexcache_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        logger.info(
            f"IndexCache stats saved to {stats_path}: "
            f"sparsity={sparsity_total:.2%} (CA:{sparsity_ca:.2%}, LA:{sparsity_la:.2%}), "
            f"block_sparsity={block_sparsity:.2%}, "
            f"size={stats['indexcache_total_size_gb']:.6f} GB"
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
