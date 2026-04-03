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
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, List, Optional
import asyncio
import ctypes
import errno
import os
import math
import threading
import time

# Third Party
import aiofiles
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.dict: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        self.lookup_server = lookup_server

        # Initialize the evictor
        self.evictor = LRUEvictor(max_cache_size=config.max_local_disk_size)

        self.loop = loop
        self.put_tasks: List[CacheEngineKey] = []
        self._clear_generation = 0  # bumped on each clear()

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0
        self.use_direct_io = (
            os.getenv("LMCACHE_LOCAL_DISK_DIRECT_IO", "0").lower() in {"1", "true"}
        )
        self.direct_io_block_size = int(
            os.getenv("LMCACHE_LOCAL_DISK_DIRECT_IO_BLOCK_SIZE", "4096")
        )
        self._libc = None
        self.bw_multiplier = int(
            os.getenv("LMCACHE_DISK_BW_MULTIPLIER", "1")
        )
        self.enable_layer_timing = (
            os.getenv("LMCACHE_ENABLE_LAYER_TIMING", "0").lower() in {"1", "true"}
        )
        self.unified_read = (
            os.getenv("LMCACHE_UNIFIED_READ", "0").lower() in {"1", "true"}
        )
        logger.info(
            "LocalDiskBackend: use_direct_io=%s, block_size=%d, path=%s, bw_multiplier=%d, unified_read=%s",
            self.use_direct_io, self.direct_io_block_size, self.path,
            self.bw_multiplier, self.unified_read,
        )
        # Per-layer read stats accumulator (thread-safe)
        self._chunk_stats: list = []  # list of (t_start, t_end, total_bytes)
        self._chunk_stats_lock = threading.Lock()
        # Structured per-layer timing data for benchmark collection
        self._layer_timing_data: dict = {}  # layer_id -> {bytes, ms, bw_gbs, num_chunks}

    def __str__(self):
        return self.__class__.__name__

    def _round_up(self, size: int) -> int:
        block = self.direct_io_block_size
        return int(math.ceil(size / block) * block)

    def _get_libc(self):
        if self._libc is None:
            self._libc = ctypes.CDLL("libc.so.6")
        return self._libc

    def pop_and_log_layer_stats(self, layer_id: int) -> None:
        """Aggregate and log read stats collected across all chunks for one layer, then clear."""
        if not self.enable_layer_timing:
            return
        with self._chunk_stats_lock:
            stats = self._chunk_stats[:]
            self._chunk_stats.clear()
        if not stats:
            return
        t_start = min(s[0] for s in stats)
        t_end = max(s[1] for s in stats)
        total_bytes = sum(s[2] for s in stats)
        elapsed = t_end - t_start
        bw_gbs = total_bytes / elapsed / (1024 ** 3) if elapsed > 0 else float("inf")
        # Store structured data for benchmark collection
        self._layer_timing_data[layer_id] = {
            "bytes": total_bytes,
            "ms": elapsed * 1000,
            "bw_gbs": bw_gbs,
            "num_chunks": len(stats),
        }
        logger.info(
            "DiskRead layer %d: %.3f MB (%d chunks x%d) in %.3f ms => %.2f GB/s",
            layer_id,
            total_bytes / (1024 ** 2),
            len(stats),
            self.bw_multiplier,
            elapsed * 1000,
            bw_gbs,
        )

    def get_layer_timing_data(self):
        """Return collected per-layer timing data and clear it."""
        data = dict(self._layer_timing_data)
        self._layer_timing_data.clear()
        return data

    def _alloc_aligned_buffer(self, size: int) -> tuple[memoryview, ctypes.c_void_p]:
        libc = self._get_libc()
        buf_ptr = ctypes.c_void_p()
        ret = libc.posix_memalign(
            ctypes.byref(buf_ptr), self.direct_io_block_size, size
        )
        if ret != 0:
            raise OSError(ret, "posix_memalign failed")
        buf_type = (ctypes.c_char * size).from_address(buf_ptr.value)
        return memoryview(buf_type), buf_ptr

    def _free_aligned_buffer(self, buf_ptr: ctypes.c_void_p) -> None:
        libc = self._get_libc()
        libc.free(buf_ptr)

    def _read_direct_into(self, path: str, memory_obj: MemoryObj, size: int) -> None:
        padded_size = self._round_up(size)
        buf_view, buf_ptr = self._alloc_aligned_buffer(padded_size)
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            remaining = padded_size
            offset = 0
            while remaining > 0:
                read = os.readv(fd, [buf_view[offset : offset + remaining]])
                if read == 0:
                    break
                offset += read
                remaining -= read
            memory_obj.byte_array.cast('B')[:size] = buf_view.cast('B')[:size]
        finally:
            if fd is not None:
                os.close(fd)
            self._free_aligned_buffer(buf_ptr)

    def _read_direct_dummy(self, path: str, size: int) -> None:
        """Read a file into a scratch aligned buffer and discard (for BW simulation)."""
        padded_size = self._round_up(size)
        buf_view, buf_ptr = self._alloc_aligned_buffer(padded_size)
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            remaining = padded_size
            offset = 0
            while remaining > 0:
                read = os.readv(fd, [buf_view[offset : offset + remaining]])
                if read == 0:
                    break
                offset += read
                remaining -= read
        finally:
            if fd is not None:
                os.close(fd)
            self._free_aligned_buffer(buf_ptr)

    def _write_direct_from(self, path: str, memory_obj: MemoryObj, size: int) -> None:
        padded_size = self._round_up(size)
        buf_view, buf_ptr = self._alloc_aligned_buffer(padded_size)
        buf_view.cast('B')[:size] = memory_obj.byte_array.cast('B')[:size]
        if padded_size > size:
            buf_view.cast('B')[size:padded_size] = b"\x00" * (padded_size - size)
        fd = None
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
            remaining = padded_size
            offset = 0
            while remaining > 0:
                written = os.writev(fd, [buf_view[offset : offset + remaining]])
                if written == 0:
                    raise OSError("writev returned 0 bytes")
                offset += written
                remaining -= written
            os.fsync(fd)
        finally:
            if fd is not None:
                os.close(fd)
            self._free_aligned_buffer(buf_ptr)

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                if hasattr(key, 'layer_id') and key.layer_id == 0:
                    logger.info(
                        "disk contains MISS: hash=%s layer=%d, dict_size=%d",
                        key.chunk_hash[:16], key.layer_id, len(self.dict),
                    )
                return False
            if pin:
                self.dict[key].pin()
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.put_tasks

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        with self.disk_lock:
            if key not in self.dict:
                return
            path = self.dict.pop(key).path
        try:
            size = os.path.getsize(path)
            self.usage -= size
            self.evictor.current_cache_size -= size
            self.stats_monitor.update_local_storage_usage(self.usage)
            os.remove(path)
        except FileNotFoundError:
            # File already deleted (e.g. by shutil.rmtree between entries)
            pass

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "disk")
            )

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        # Store old_positions if it exists (set during batched_from_gpu)
        old_positions = None
        if hasattr(memory_obj.metadata, 'old_positions') and memory_obj.metadata.old_positions is not None:
            old_positions = memory_obj.metadata.old_positions

        has_stored = False
        with self.disk_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)
                has_stored = True

            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype, fmt, False, old_positions)

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "disk")
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:
        assert memory_obj.tensor is not None

        # Update cache recency
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, memory_obj.get_physical_size()
        )
        if put_status == PutStatus.ILLEGAL:
            return None
        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)

        memory_obj.ref_count_up()

        self.disk_lock.acquire()
        self.put_tasks.append(key)
        self.disk_lock.release()

        if hasattr(key, 'layer_id') and key.layer_id == 0:
            logger.info(
                "submit_put_task: scheduling async write hash=%s layer=%d, "
                "loop_running=%s, loop_closed=%s",
                key.chunk_hash[:16], key.layer_id,
                self.loop.is_running(), self.loop.is_closed(),
            )

        gen = self._clear_generation
        future = asyncio.run_coroutine_threadsafe(
            self.async_save_bytes_to_disk(key, memory_obj, gen), self.loop
        )
        return future

    def batched_submit_put_task(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> Optional[List[Future]]:
        return [
            self.submit_put_task(key, memory_obj)
            for key, memory_obj in zip(keys, memory_objs, strict=False)
        ]

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        self.evictor.update_on_hit(key, self.dict)

        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        fmt = self.dict[key].fmt
        self.disk_lock.release()
        #logger.info(f"Prefetching {key} from disk.")

        assert dtype is not None
        assert shape is not None
        future = asyncio.run_coroutine_threadsafe(
            self.async_load_bytes_from_disk(path, dtype, shape, fmt, key), self.loop
        )
        return future

    def _read_all_chunks_direct(
        self,
        chunk_infos: List[tuple],
    ) -> List[MemoryObj]:
        """Read all chunks sequentially in a single thread using O_DIRECT."""
        results: List[MemoryObj] = []
        for memory_obj, path, size in chunk_infos:
            if self.enable_layer_timing:
                t_start = time.perf_counter()
            self._read_direct_into(path, memory_obj, size)
            if self.enable_layer_timing:
                t_end = time.perf_counter()
                with self._chunk_stats_lock:
                    self._chunk_stats.append((t_start, t_end, size))
            results.append(memory_obj)
        return results

    async def async_unified_load_from_disk(
        self,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """Async wrapper: look up all keys, read all chunks in ONE thread."""
        chunk_infos = []
        with self.disk_lock:
            for key in keys:
                meta = self.dict[key]
                self.evictor.update_on_hit(key, self.dict)
                memory_obj = self.local_cpu_backend.allocate(
                    meta.shape, meta.dtype, meta.fmt
                )
                assert memory_obj is not None
                size = memory_obj.get_size()
                chunk_infos.append((memory_obj, meta.path, size))

        results = await asyncio.to_thread(
            self._read_all_chunks_direct, chunk_infos
        )

        # Restore old_positions from disk metadata
        for key, memory_obj in zip(keys, results, strict=False):
            with self.disk_lock:
                if key in self.dict and self.dict[key].old_positions is not None:
                    memory_obj.metadata.old_positions = self.dict[key].old_positions

        return results

    def submit_unified_prefetch_task(
        self,
        keys: List[CacheEngineKey],
    ) -> Future:
        """Submit all chunks for a layer as a single sequential read task."""
        #for key in keys:
        #    logger.info(f"Prefetching {key} from disk (unified).")
        future = asyncio.run_coroutine_threadsafe(
            self.async_unified_load_from_disk(keys), self.loop
        )
        return future

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        self.evictor.update_on_hit(key, self.dict)

        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        fmt = self.dict[key].fmt
        assert dtype is not None
        assert shape is not None
        memory_obj = self.load_bytes_from_disk(path, dtype=dtype, shape=shape, fmt=fmt, key=key)
        self.disk_lock.release()
        return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Non-blocking get function.
        Using a dummy wrapper around prefetch for now.
        """
        # TODO(Jiayi): Need to align prefetch and get_non_blocking
        return self.submit_prefetch_task(key)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        generation: int = -1,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        try:
            is_layer0 = hasattr(key, 'layer_id') and key.layer_id == 0
            if is_layer0:
                logger.info(
                    "async_save START: hash=%s layer=%d",
                    key.chunk_hash[:16], key.layer_id,
                )

            kv_chunk = memory_obj.tensor
            assert kv_chunk is not None
            byte_array = memory_obj.byte_array
            path = self._key_to_path(key)

            size = len(byte_array)
            self.usage += size
            self.stats_monitor.update_local_storage_usage(self.usage)

            if self.use_direct_io:
                try:
                    await asyncio.to_thread(
                        self._write_direct_from, path, memory_obj, len(byte_array)
                    )
                    # Write dummy copies for BW simulation
                    for i in range(1, self.bw_multiplier):
                        dup_path = path + f".dup{i}"
                        await asyncio.to_thread(
                            self._write_direct_from, dup_path, memory_obj, len(byte_array)
                        )
                except OSError as e:
                    if e.errno in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP):
                        logger.warning(
                            "Direct I/O unsupported for %s; falling back to buffered I/O.",
                            path,
                        )
                    else:
                        raise
                    logger.warning(
                        "Direct I/O unsupported for %s; falling back to buffered I/O.",
                        path,
                    )
                    async with aiofiles.open(path, "wb") as f:
                        await f.write(byte_array)
            else:
                async with aiofiles.open(path, "wb") as f:
                    await f.write(byte_array)

            if is_layer0:
                logger.info(
                    "async_save WRITTEN: hash=%s layer=%d, path=%s",
                    key.chunk_hash[:16], key.layer_id, path,
                )

            # If a clear() happened after this write was submitted,
            # do NOT insert into the dict — the file will be deleted
            # by shutil.rmtree and inserting would create a stale entry.
            if generation >= 0 and generation != self._clear_generation:
                if is_layer0:
                    logger.info(
                        "async_save STALE (gen %d != %d): hash=%s layer=%d, "
                        "skipping insert",
                        generation, self._clear_generation,
                        key.chunk_hash[:16], key.layer_id,
                    )
                memory_obj.ref_count_down()
                self.disk_lock.acquire()
                if key in self.put_tasks:
                    self.put_tasks.remove(key)
                self.disk_lock.release()
                return

            self.insert_key(key, memory_obj)

            if is_layer0:
                logger.info(
                    "async_save DONE: hash=%s layer=%d, dict_size=%d",
                    key.chunk_hash[:16], key.layer_id, len(self.dict),
                )

            memory_obj.ref_count_down()

            self.disk_lock.acquire()
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            self.disk_lock.release()
        except Exception as e:
            logger.error(
                "async_save EXCEPTION: hash=%s, error=%s",
                key.chunk_hash[:16] if hasattr(key, 'chunk_hash') else "?",
                str(e),
            )
            import traceback
            logger.error("async_save traceback: %s", traceback.format_exc())

    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    async def async_load_bytes_from_disk(
        self, path: str, dtype: torch.dtype, shape: torch.Size, fmt: MemoryFormat, key: Optional[CacheEngineKey] = None
    ) -> Optional[MemoryObj]:
        """
        Async load bytearray from disk.
        """
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async disk load.")
            return None
        size = memory_obj.get_size()
        if self.use_direct_io:
            try:
                real_task = asyncio.to_thread(
                    self._read_direct_into, path, memory_obj, size
                )
                dummy_tasks = [
                    asyncio.to_thread(
                        self._read_direct_dummy, path + f".dup{i}", size
                    )
                    for i in range(1, self.bw_multiplier)
                ]
                if self.enable_layer_timing:
                    t_start = time.perf_counter()
                await asyncio.gather(real_task, *dummy_tasks)
                if self.enable_layer_timing:
                    t_end = time.perf_counter()
                    with self._chunk_stats_lock:
                        self._chunk_stats.append((t_start, t_end, size * self.bw_multiplier))
            except OSError as e:
                if e.errno in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP):
                    logger.warning(
                        "Direct I/O unsupported for %s; falling back to buffered I/O.",
                        path,
                    )
                else:
                    raise
                logger.warning(
                    "Direct I/O unsupported for %s; falling back to buffered I/O.",
                    path,
                )
                buffer = memory_obj.byte_array
                async with aiofiles.open(path, "rb") as f:
                    await f.readinto(buffer)
        else:
            buffer = memory_obj.byte_array
            async with aiofiles.open(path, "rb") as f:
                await f.readinto(buffer)

        # Restore old_positions from metadata if available (same as CPU backend)
        if key is not None:
            with self.disk_lock:
                if key in self.dict and self.dict[key].old_positions is not None:
                    memory_obj.metadata.old_positions = self.dict[key].old_positions

        return memory_obj

    # TODO(Jiayi): use memory allocator to redeuce cpu buffer allocation
    # TODO(Jiayi): the pinned cpu memory_obj should directly be passed into
    # gpu connector; this gpu buffer could be avoided
    def load_bytes_from_disk(
        self, path: str, dtype: torch.dtype, shape: torch.Size, fmt: MemoryFormat, key: Optional[CacheEngineKey] = None
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async disk load.")
            return None
        size = memory_obj.get_size()
        if self.use_direct_io:
            try:
                if self.enable_layer_timing:
                    t_start = time.perf_counter()
                self._read_direct_into(path, memory_obj, size)
                # Dummy reads for BW simulation (sequential in blocking path)
                for i in range(1, self.bw_multiplier):
                    self._read_direct_dummy(path + f".dup{i}", size)
                if self.enable_layer_timing:
                    t_end = time.perf_counter()
                    with self._chunk_stats_lock:
                        self._chunk_stats.append((t_start, t_end, size * self.bw_multiplier))
            except OSError as e:
                if e.errno in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP):
                    logger.warning(
                        "Direct I/O unsupported for %s; falling back to buffered I/O.",
                        path,
                    )
                else:
                    raise
                logger.warning(
                    "Direct I/O unsupported for %s; falling back to buffered I/O.",
                    path,
                )
                buffer = memory_obj.byte_array
                with open(path, "rb") as f:
                    f.readinto(buffer)
        else:
            buffer = memory_obj.byte_array
            with open(path, "rb") as f:
                f.readinto(buffer)

        # Restore old_positions from metadata if available (same as CPU backend)
        if key is not None:
            with self.disk_lock:
                if key in self.dict and self.dict[key].old_positions is not None:
                    memory_obj.metadata.old_positions = self.dict[key].old_positions

        return memory_obj

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def load_disk(
        self,
        path: str,
        backend: str = "bytes",
        dtype: Optional[torch.dtype] = None,
        shape: Optional[torch.Size] = None,
        fmt: Optional[MemoryFormat] = None,
    ) -> Optional[MemoryObj]:
        """
        Load KV from disk.
        """
        if backend == "bytes":
            assert dtype is not None
            assert shape is not None
            memory_obj = self.load_bytes_from_disk(path, dtype, shape, fmt)
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return memory_obj

    def close(self) -> None:
        if self.lookup_server is not None:
            self.disk_lock.acquire()
            self.lookup_server.batched_remove(list(self.dict.keys()))
            self.disk_lock.release()

    def clear(self) -> int:
        """
        Clear all cached KV chunks from disk.
        Returns the number of cleared keys.
        """
        # Bump generation so in-flight async writes skip insert_key
        self._clear_generation += 1
        logger.info(
            "disk clear: generation=%d, dict_size=%d, pending_puts=%d",
            self._clear_generation, len(self.dict), len(self.put_tasks),
        )
        with self.disk_lock:
            clear_keys = list(self.dict.keys())
            # Also discard pending put_tasks — they belong to the old gen
            self.put_tasks.clear()
        for key in clear_keys:
            self.remove(key)
        # Reset evictor's size counter so it matches the now-empty disk
        self.evictor.current_cache_size = 0
        return len(clear_keys)
