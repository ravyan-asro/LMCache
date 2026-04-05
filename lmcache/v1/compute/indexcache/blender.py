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

        # Disk backend (optional)
        self.disk_backend = None
        if config.disk_cache_dir:
            from lmcache.v1.compute.indexcache.disk_backend import (
                IndexCacheDiskBackend,
            )
            self.disk_backend = IndexCacheDiskBackend(config.disk_cache_dir)

        # Merged per-layer CPU tensors (populated by _merge_chunk_metadata_cpu)
        self._merged_kvcolidx_cpu = None  # list of (H, num_indices) per layer
        self._merged_hot_tile_cpu = None  # list of (H, num_bytes) per layer
        self._merged_offset_list = None
        self._merged_question_len = None

        # Structured timing data (populated by _blend_pipelined / blend)
        self._timing_data = None
        self.enable_layer_timing = (
            os.getenv("LMCACHE_ENABLE_LAYER_TIMING", "0").lower() in {"1", "true"}
        )

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
    # Disk save / load
    # ------------------------------------------------------------------
    def save_to_disk(self, key: str):
        """Save current chunk_metadata to disk."""
        if self.disk_backend is None:
            logger.warning("save_to_disk called but no disk backend configured")
            return
        self.disk_backend.save(key, self.chunk_metadata, self.cached_token_ids)

    def load_from_disk(self, key: str) -> bool:
        """Load pre-merged metadata from disk via O_DIRECT.

        Populates _merged_kvcolidx_cpu and _merged_hot_tile_cpu directly
        (no CPU merge needed). Returns True on success.
        """
        if self.disk_backend is None:
            return False
        result = self.disk_backend.load(key)
        if result is None:
            return False

        self._merged_kvcolidx_cpu = result["merged_kvcolidx"]
        self._merged_hot_tile_cpu = result["merged_hot_tile"]
        self._merged_offset_list = result["offset_list"]
        self.cached_token_ids = result["cached_token_ids"]

        # Reconstruct minimal chunk_metadata for sparsity stats
        # (only chunk lengths needed, not full tensors)
        offsets = [0] + result["offset_list"]
        self.chunk_metadata = [
            (offsets[i+1] - offsets[i], [], [])
            for i in range(result["num_chunks"])
        ]

        self._disk_read_ms = result.get("disk_read_ms", 0)
        self._deserialize_ms = result.get("deserialize_ms", 0)
        self._disk_file_bytes = result.get("disk_file_bytes", 0)

        logger.info(
            f"IndexCache loaded from disk: {result['num_chunks']} chunks, "
            f"{len(self.cached_token_ids)} tokens, "
            f"disk_read={self._disk_read_ms:.2f}ms, "
            f"deserialize={self._deserialize_ms:.2f}ms"
        )
        return True

    # ------------------------------------------------------------------
    # Merge chunk metadata into per-layer CPU tensors
    # ------------------------------------------------------------------
    def _merge_chunk_metadata_cpu(self):
        """Merge per-chunk metadata into per-layer CPU tensors.

        Populates self._merged_kvcolidx_cpu and self._merged_hot_tile_cpu.
        These are lists of num_layers elements, each a CPU tensor ready
        for per-layer GPU transfer.
        """
        if not self.chunk_metadata:
            return

        page_size = self.config.page_size
        num_layers = self.model.num_layers
        num_chunks = len(self.chunk_metadata)

        offset_list = list(
            itertools.accumulate(cl for cl, _, _ in self.chunk_metadata)
        )

        kvcolidx_list = []
        hot_tile_list = []

        for layer_idx in range(num_layers):
            # Merge kvcolidx across chunks 0..N-2 (exclude last)
            # kvcolidx is now uint8 bit-packed — just concatenate bytes
            # (each chunk is byte-aligned, no offset adjustment needed)
            kv_parts = [
                self.chunk_metadata[ci][1][layer_idx]
                for ci in range(num_chunks - 1)
            ]
            if kv_parts:
                merged_kv = torch.cat(kv_parts, dim=-1).squeeze(0)  # (H, total_bytes)
            else:
                H = self.chunk_metadata[0][1][layer_idx].shape[1]
                merged_kv = torch.empty(H, 0, dtype=torch.uint8)
            kvcolidx_list.append(merged_kv.contiguous())

            # Merge hot_tile across all chunks (same as before — already byte-aligned)
            ht_parts = [
                self.chunk_metadata[ci][2][layer_idx]
                for ci in range(num_chunks)
            ]
            merged_ht = torch.cat(ht_parts, dim=-1).squeeze(0)  # (H, total_bytes)
            hot_tile_list.append(merged_ht.contiguous())

        self._merged_kvcolidx_cpu = kvcolidx_list
        self._merged_hot_tile_cpu = hot_tile_list
        self._merged_offset_list = offset_list

        logger.info(
            f"Merged chunk metadata on CPU: {num_layers} layers, "
            f"{num_chunks} chunks, offsets={offset_list}"
        )

    # ------------------------------------------------------------------
    # FA3 metadata pre-computation (called BEFORE llm.generate())
    # ------------------------------------------------------------------
    def precompute_sparse_metadata(
        self, total_seqlen: int, question_len: int, device: str = "cuda"
    ):
        """Pre-compute FA3 sparse metadata for all layers.

        Call this after all cache_gen() calls and before llm.generate()
        so the conversion cost is excluded from TTFT.

        When pipeline_blend is enabled, this only merges chunk metadata
        on CPU (the GPU transfer + inflate happens per-layer in blend()).

        Args:
            total_seqlen: total number of tokens (context + question)
            question_len: number of question tokens
            device: target device for sparse tensors
        """
        if self.config.attn_backend != "fa3":
            return

        has_pending_disk = (hasattr(self, '_pending_disk_key')
                            and self._pending_disk_key is not None
                            and self.disk_backend is not None)

        if not self.chunk_metadata and self._merged_kvcolidx_cpu is None:
            if has_pending_disk:
                # Disk load deferred to blend() for timing
                logger.info(
                    "precompute: metadata on disk, load deferred to blend()")
                return
            else:
                logger.warning("precompute_sparse_metadata: no chunk metadata")
                return

        # When pipelining, only merge on CPU — GPU work deferred to blend()
        if self.config.pipeline_blend:
            if self._merged_kvcolidx_cpu is None:
                self._merge_chunk_metadata_cpu()
            logger.info(
                "Pipeline blend enabled — CPU merge done, "
                "GPU transfer deferred to blend()"
            )
            return

        # Merge metadata (same logic as blend())
        page_size = self.config.page_size
        offset_list = list(
            itertools.accumulate(cl for cl, _, _ in self.chunk_metadata)
        )

        num_chunks = len(self.chunk_metadata)
        num_layers = self.model.num_layers

        import sys
        fa3_path = "/var/tmp/rsanovar3/flash-attention/hopper"
        if fa3_path not in sys.path:
            sys.path.insert(0, fa3_path)
        from indexcache_inflate import build_indexcache_metadata_gpu_all_layers

        torch.cuda.synchronize()
        t0 = time.time()

        # --- Merge on CPU, then bulk transfer to GPU ---
        # kvcolidx: uint8 bit vectors, just concatenate bytes across chunks 0..N-2
        merged_kv_list = []
        merged_ht_list = []
        for layer_idx in range(num_layers):
            kv_parts = [
                self.chunk_metadata[ci][1][layer_idx]
                for ci in range(num_chunks - 1)
            ]
            if kv_parts:
                merged_kv_list.append(
                    torch.cat(kv_parts, dim=-1).squeeze(0)
                )
            else:
                H = self.chunk_metadata[0][1][layer_idx].shape[1]
                merged_kv_list.append(
                    torch.empty(H, 0, dtype=torch.uint8)
                )
            ht_parts = [
                self.chunk_metadata[ci][2][layer_idx]
                for ci in range(num_chunks)
            ]
            merged_ht_list.append(
                torch.cat(ht_parts, dim=-1).squeeze(0)
            )

        kvcolidx_stacked = torch.stack(
            merged_kv_list, dim=0
        ).contiguous().to(device)
        la_stacked = torch.stack(
            merged_ht_list, dim=0
        ).contiguous().to(device)

        torch.cuda.synchronize()
        t_transfer = time.time()
        transfer_ms = (t_transfer - t0) * 1000

        # --- CUDA inflate kernel (all layers, single launch) ---
        self._precomputed_sparse_metadata = \
            build_indexcache_metadata_gpu_all_layers(
                seqlen_total=total_seqlen,
                offset_list=offset_list,
                question_len=question_len,
                kvcolidx_per_layer=kvcolidx_stacked,
                la_hot_tile_code_per_layer=la_stacked,
                device=device,
            )
        torch.cuda.synchronize()
        t1 = time.time()
        kernel_ms = (t1 - t_transfer) * 1000
        total_ms = (t1 - t0) * 1000

        logger.info(
            f"FA3 sparse metadata pre-computed for {num_layers} "
            f"layers in {total_ms:.1f}ms "
            f"(cpu_to_gpu: {transfer_ms:.1f}ms, cuda_kernel: {kernel_ms:.1f}ms) "
            f"— excluded from TTFT"
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

            # CA: count selected pages from bit-packed kvcolidx
            # kvcolidx is now (B, H, num_bytes) uint8 bit vector per chunk
            global_page = 0
            for chunk_idx in range(num_docs - 1):
                kv = self.chunk_metadata[chunk_idx][1][layer_idx][0]  # (H, num_bytes) uint8
                n_pages = chunk_num_pages[chunk_idx]
                for h_idx in range(H):
                    kv_h = kv[h_idx]  # (num_bytes,) uint8
                    # Unpack bits for this chunk
                    for page_idx in range(n_pages):
                        byte_idx = page_idx // 8
                        bit_offset = 7 - (page_idx % 8)
                        if (kv_h[byte_idx].item() >> bit_offset) & 1:
                            gp = pages_before[chunk_idx] + page_idx
                            if 0 <= gp < len(page_to_doc_end):
                                active_CA_tiles += total_doc_pages - page_to_doc_end[gp]
                global_page += n_pages

            # LA: count hot bits (vectorized — same as before)
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
        """Run sparse prefill on context tokens, write KV to paged buffer.

        Supports two modes:
        - Pipelined (config.pipeline_blend=True, default): per-layer CPU→GPU
          transfer + inflate overlapped with previous layer's compute via
          CUDA streams. Transfer is hidden behind compute.
        - Bulk (config.pipeline_blend=False or pre-computed): uses
          precompute_sparse_metadata() result computed before llm.generate().

        Args:
            tokens: 1-D token IDs for the prefix (context tokens only)
            mask: unused (kept for API compatibility with CacheBlend)
            **kwargs: must contain 'kvcaches' and 'slot_mapping'
        """
        kvcaches = kwargs["kvcaches"]
        slot_mapping = kwargs["slot_mapping"]

        # Load from disk if we have a pending disk key and no in-memory data
        if (self._merged_kvcolidx_cpu is None
                and hasattr(self, '_pending_disk_key')
                and self._pending_disk_key is not None
                and self.disk_backend is not None):
            self.load_from_disk(self._pending_disk_key)
            self._pending_disk_key = None

        if not self.chunk_metadata and self._merged_kvcolidx_cpu is None:
            logger.warning("IndexCacheBlender.blend() called with no metadata")
            return

        device = tokens.device if tokens.is_cuda else "cuda"
        num_layers = self.model.num_layers

        # Ensure merged CPU metadata exists
        if self._merged_kvcolidx_cpu is None:
            self._merge_chunk_metadata_cpu()

        offset_list = self._merged_offset_list
        question_len = len(tokens) - offset_list[-1]

        logger.info(
            f"IndexCache blend: {len(tokens)} total tokens "
            f"({len(tokens) - question_len} context + {question_len} question), "
            f"{len(self.chunk_metadata)} chunks, offsets={offset_list}, "
            f"backend={self.config.attn_backend}, "
            f"pipeline={self.config.pipeline_blend}"
        )

        # --- Check for pre-computed metadata (bulk path) ---
        sparse_metadata_caches = getattr(
            self, '_precomputed_sparse_metadata', None
        )

        # --- Pipelined path: per-layer CPU→GPU + inflate overlapped w/ compute ---
        if (self.config.pipeline_blend
                and self.config.attn_backend == "fa3"
                and sparse_metadata_caches is None):
            self._blend_pipelined(
                tokens, offset_list, question_len,
                kvcaches, slot_mapping, device,
            )
            return

        # --- Bulk / fallback path ---
        # Transfer all metadata to GPU (old path for lean_attn or if
        # precomputed sparse metadata exists)
        kvcolidx_caches = [
            kv.unsqueeze(0).to(device) for kv in self._merged_kvcolidx_cpu
        ]
        hot_tile_caches = [
            ht.unsqueeze(0).to(device) for ht in self._merged_hot_tile_cpu
        ]

        if self.config.attn_backend == "fa3" and sparse_metadata_caches is None:
            # Fallback: compute all layers in-band
            logger.warning(
                "FA3 sparse metadata not pre-computed and pipeline disabled — "
                "computing in-band (included in TTFT)"
            )
            from lmcache.v1.compute.models.indexcache_llama import _get_fa3
            _, build_meta = _get_fa3()
            t0 = time.time()
            sparse_metadata_caches = []
            for layer_idx in range(num_layers):
                indices, offsets, mask_counts = build_meta(
                    seqlen_total=len(tokens),
                    offset_list=offset_list,
                    question_len=question_len,
                    kvcolidx_cache=kvcolidx_caches[layer_idx].contiguous(),
                    la_hot_tile_code_cache=hot_tile_caches[layer_idx].contiguous(),
                    device=str(device),
                )
                sparse_metadata_caches.append((indices, offsets, mask_counts))
            logger.info(f"FA3 metadata in-band: {(time.time()-t0)*1000:.1f}ms")

        with torch.no_grad():
            gen = self.model.compute_layer(
                tokens, kvcolidx_caches, hot_tile_caches,
                offset_list, self.config.page_size, kvcaches, slot_mapping,
                attn_backend=self.config.attn_backend,
                question_len=question_len,
                sparse_metadata_caches=sparse_metadata_caches,
            )
            for _ in range(num_layers):
                next(gen)

        logger.info("IndexCache blend done — KV written to paged buffer")

    def _blend_pipelined(self, tokens, offset_list, question_len,
                         kvcaches, slot_mapping, device):
        """Pipelined blend: bulk CPU→GPU transfer + inflate for all layers,
        then per-layer sparse prefill compute.

        The inflate kernel contains an internal cudaStreamSynchronize (for
        prefix-sum output allocation) which blocks the CPU thread, preventing
        true per-layer overlap of inflate with compute. Instead, we:
          1. Bulk transfer all merged metadata CPU→GPU (async, ~few ms)
          2. Run the batched all-layers inflate kernel (fast, ~4.5ms)
          3. Run per-layer sparse prefill compute

        Steps 1+2 happen at the start of blend() and are included in TTFT,
        but they're fast enough (~10ms total) that the impact is minimal
        compared to the ~800ms+ of sparse prefill compute.
        """
        import sys
        fa3_path = "/var/tmp/rsanovar3/flash-attention/hopper"
        if fa3_path not in sys.path:
            sys.path.insert(0, fa3_path)
        from indexcache_inflate import build_indexcache_metadata_gpu_all_layers

        num_layers = self.model.num_layers
        seqlen = len(tokens)

        timing = self.enable_layer_timing

        # Step 1: Bulk CPU→GPU transfer (stack per-layer → one contiguous tensor)
        if timing:
            torch.cuda.synchronize()  # drain GPU queue before timing
        t0 = time.time()

        kvcolidx_cpu_stacked = torch.stack(
            self._merged_kvcolidx_cpu, dim=0
        ).contiguous()
        la_cpu_stacked = torch.stack(
            self._merged_hot_tile_cpu, dim=0
        ).contiguous()
        metadata_bytes_cpu = (kvcolidx_cpu_stacked.numel() * kvcolidx_cpu_stacked.element_size()
                              + la_cpu_stacked.numel() * la_cpu_stacked.element_size())

        kvcolidx_stacked = kvcolidx_cpu_stacked.to(device)
        la_stacked = la_cpu_stacked.to(device)

        if timing:
            torch.cuda.synchronize()  # wait for async H2D to complete
        t_transfer = time.time()
        transfer_ms = (t_transfer - t0) * 1000

        # Step 2: Batched inflate kernel (all layers, single launch)
        if timing:
            gmem_before = torch.cuda.memory_allocated(device)

        sparse_metadata_caches = build_indexcache_metadata_gpu_all_layers(
            seqlen_total=seqlen,
            offset_list=offset_list,
            question_len=question_len,
            kvcolidx_per_layer=kvcolidx_stacked,
            la_hot_tile_code_per_layer=la_stacked,
            device=str(device),
        )

        torch.cuda.synchronize()  # inflate kernel has internal sync anyway
        if timing:
            gmem_after = torch.cuda.memory_allocated(device)
            inflate_gmem_bytes = gmem_after - gmem_before
        else:
            inflate_gmem_bytes = 0
        t_inflate = time.time()
        inflate_ms = (t_inflate - t_transfer) * 1000

        disk_ms = getattr(self, '_disk_read_ms', 0)
        deser_ms = getattr(self, '_deserialize_ms', 0)
        logger.info(
            f"Pipelined blend: disk_read={disk_ms:.1f}ms, "
            f"deserialize={deser_ms:.1f}ms, "
            f"cpu_to_gpu={transfer_ms:.1f}ms, "
            f"inflate={inflate_ms:.1f}ms"
            + (f", inflate_gmem={inflate_gmem_bytes / (1024**2):.1f}MB" if timing else "")
            + " (all included in TTFT)"
        )

        # Step 3: Per-layer sparse prefill compute
        # Pass placeholder GPU tensors for kvcolidx/hot_tile (FA3 only reads
        # sparse_metadata_caches, not the raw bit vectors)
        kvcolidx_caches = [
            self._merged_kvcolidx_cpu[0].unsqueeze(0).to(device)
        ] * num_layers
        hot_tile_caches = [
            self._merged_hot_tile_cpu[0].unsqueeze(0).to(device)
        ] * num_layers

        per_layer_compute_ms = []
        with torch.no_grad():
            gen = self.model.compute_layer(
                tokens, kvcolidx_caches, hot_tile_caches,
                offset_list, self.config.page_size, kvcaches, slot_mapping,
                attn_backend="fa3",
                question_len=question_len,
                sparse_metadata_caches=sparse_metadata_caches,
            )
            if timing:
                for layer_idx in range(num_layers):
                    torch.cuda.synchronize()
                    t_layer_start = time.time()
                    next(gen)
                    torch.cuda.synchronize()
                    t_layer_end = time.time()
                    per_layer_compute_ms.append((t_layer_end - t_layer_start) * 1000)
            else:
                for _ in range(num_layers):
                    next(gen)

        t1 = time.time()
        compute_ms = (t1 - t_inflate) * 1000

        if timing:
            disk_file_bytes = getattr(self, '_disk_file_bytes', 0)
            disk_bw_gbs = (
                disk_file_bytes / (disk_ms / 1000) / (1024 ** 3)
                if disk_ms > 0 else float("inf")
            )
            transfer_bw_gbs = (
                metadata_bytes_cpu / (transfer_ms / 1000) / (1024 ** 3)
                if transfer_ms > 0 else float("inf")
            )
            self._timing_data = {
                "disk_read_ms": disk_ms,
                "disk_file_bytes": disk_file_bytes,
                "disk_bw_gbs": disk_bw_gbs,
                "deserialize_ms": deser_ms,
                "cpu_to_gpu_ms": transfer_ms,
                "cpu_to_gpu_bytes": metadata_bytes_cpu,
                "cpu_to_gpu_bw_gbs": transfer_bw_gbs,
                "inflate_ms": inflate_ms,
                "inflate_gmem_bytes": inflate_gmem_bytes,
                "inflate_gmem_mb": inflate_gmem_bytes / (1024 ** 2),
                "compute_total_ms": compute_ms,
                "per_layer_compute_ms": per_layer_compute_ms,
                "num_layers": num_layers,
                "total_ms": (t1 - t0) * 1000,
            }
            logger.info(
                f"IndexCache pipelined blend done — {num_layers} layers "
                f"in {(t1-t0)*1000:.1f}ms "
                f"(disk={disk_ms:.1f}ms, deser={deser_ms:.1f}ms, "
                f"cpu2gpu={transfer_ms:.1f}ms [{transfer_bw_gbs:.2f} GB/s], "
                f"inflate={inflate_ms:.1f}ms [{inflate_gmem_bytes/(1024**2):.1f}MB GMEM], "
                f"compute={compute_ms:.1f}ms)"
            )
        else:
            logger.info(
                f"IndexCache pipelined blend done — {num_layers} layers "
                f"in {(t1-t0)*1000:.1f}ms"
            )

        # Compute and save sparsity + size stats (rank 0 only for TP>1)
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        if rank == 0:
            try:
                self._compute_and_save_stats(len(tokens), question_len)
            except Exception as e:
                logger.warning(f"Failed to save IndexCache stats: {e}")

    # ------------------------------------------------------------------
    # Reset (between entries)
    # ------------------------------------------------------------------
    def reset(self):
        """Clear stored metadata for the next query."""
        self.chunk_metadata.clear()
        self.cached_token_ids.clear()
        self._precomputed_sparse_metadata = None
        self._merged_kvcolidx_cpu = None
        self._merged_hot_tile_cpu = None
        self._merged_offset_list = None
        self._merged_question_len = None
        self._timing_data = None
        logger.info("IndexCache metadata reset")
