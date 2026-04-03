"""Disk backend for IndexCache metadata persistence.

Stores merged per-layer IndexCache metadata on disk using O_DIRECT
to bypass the OS page cache (same as CacheBlend's KV disk path).
On load, reads raw bytes via O_DIRECT into an aligned buffer, then
deserializes. This gives honest disk-to-CPU transfer latency.

Storage format: a single file per entry containing:
  - 4-byte magic
  - 4-byte version
  - 4-byte num_layers
  - 4-byte num_chunks
  - offset_list as int32 array (num_chunks entries)
  - cached_token_ids length (4 bytes) + token IDs as int32 array
  - Per layer: kvcolidx raw bytes + hot_tile raw bytes
  - Layer index table for seeking
"""

import ctypes
import hashlib
import io
import os
import struct
import tempfile
import time

import torch
import numpy as np

from lmcache.logging import init_logger

logger = init_logger(__name__)

MAGIC = b'IC01'
VERSION = 2
BLOCK_SIZE = 4096  # O_DIRECT alignment


def _align_up(n, alignment=BLOCK_SIZE):
    return ((n + alignment - 1) // alignment) * alignment


class IndexCacheDiskBackend:
    """Save/load IndexCache metadata to/from disk with O_DIRECT."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key_to_path(self, key: str) -> str:
        safe = hashlib.sha256(key.encode()).hexdigest()[:32]
        return os.path.join(self.cache_dir, f"ic_{safe}.bin")

    def exists(self, key: str) -> bool:
        return os.path.exists(self._key_to_path(key))

    def save(self, key: str, chunk_metadata: list, cached_token_ids: list):
        """Save merged IndexCache metadata to disk.

        Serializes all chunk metadata into a flat binary format that
        can be read back with O_DIRECT.
        """
        path = self._key_to_path(key)
        num_chunks = len(chunk_metadata)
        num_layers = len(chunk_metadata[0][1])

        # Build offset_list from chunk lengths
        offset_list = []
        cumulative = 0
        for chunk_len, _, _ in chunk_metadata:
            cumulative += chunk_len
            offset_list.append(cumulative)

        # Serialize into a byte buffer
        buf = io.BytesIO()

        # Header
        buf.write(MAGIC)
        buf.write(struct.pack('<III', VERSION, num_layers, num_chunks))

        # Offset list
        offset_arr = np.array(offset_list, dtype=np.int32)
        buf.write(offset_arr.tobytes())

        # Cached token IDs
        token_arr = np.array(cached_token_ids, dtype=np.int32)
        buf.write(struct.pack('<I', len(token_arr)))
        buf.write(token_arr.tobytes())

        # Per-layer metadata: kvcolidx (uint8 bit vector) + hot_tile for each layer
        layer_data = []
        for layer_idx in range(num_layers):
            # Merge kvcolidx across chunks 0..N-2 (exclude last)
            # kvcolidx is uint8 bit-packed — just concatenate bytes
            kv_parts = [
                chunk_metadata[ci][1][layer_idx]
                for ci in range(num_chunks - 1)
            ]
            if kv_parts:
                merged_kv = torch.cat(kv_parts, dim=-1).squeeze(0).contiguous()
            else:
                H = chunk_metadata[0][1][layer_idx].shape[1]
                merged_kv = torch.empty(H, 0, dtype=torch.uint8)

            # Merge hot_tile across all chunks (already byte-aligned)
            ht_parts = [chunk_metadata[ci][2][layer_idx] for ci in range(num_chunks)]
            merged_ht = torch.cat(ht_parts, dim=-1).squeeze(0).contiguous()

            kv_bytes = merged_kv.numpy().tobytes()
            ht_bytes = merged_ht.numpy().tobytes()

            # Store shape info + data
            kv_shape = merged_kv.shape  # (H, num_bytes) uint8
            ht_shape = merged_ht.shape  # (H, num_bytes) uint8
            layer_data.append((kv_shape, kv_bytes, ht_shape, ht_bytes))

        # Write layer index (shapes + sizes for seeking)
        for kv_shape, kv_bytes, ht_shape, ht_bytes in layer_data:
            buf.write(struct.pack('<IIII',
                                  kv_shape[0], kv_shape[1] if len(kv_shape) > 1 else 0,
                                  ht_shape[0], ht_shape[1] if len(ht_shape) > 1 else 0))
            buf.write(struct.pack('<II', len(kv_bytes), len(ht_bytes)))
            buf.write(kv_bytes)
            buf.write(ht_bytes)

        raw = buf.getvalue()

        # Atomic write with O_DIRECT
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".bin.tmp")
        os.close(fd)
        try:
            self._write_direct(tmp_path, raw)
            os.rename(tmp_path, path)
            file_size = _align_up(len(raw))
            logger.info(f"IndexCache saved to disk: {path} "
                        f"({file_size / 1024:.1f} KB, {num_layers} layers)")
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load(self, key: str):
        """Load IndexCache metadata from disk using O_DIRECT.

        Returns:
            dict with 'chunk_metadata', 'cached_token_ids', 'merged_kvcolidx',
            'merged_hot_tile', 'offset_list' — or None if not found.
            merged_kvcolidx and merged_hot_tile are lists of CPU tensors per layer.
        """
        path = self._key_to_path(key)
        if not os.path.exists(path):
            return None

        file_size = os.path.getsize(path)

        t0 = time.time()

        # Read with O_DIRECT (bypasses page cache)
        raw = self._read_direct(path, file_size)

        t_disk = time.time()
        disk_ms = (t_disk - t0) * 1000

        # Deserialize
        buf = io.BytesIO(raw)

        magic = buf.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        version, num_layers, num_chunks = struct.unpack('<III', buf.read(12))

        # Offset list
        offset_arr = np.frombuffer(buf.read(num_chunks * 4), dtype=np.int32)
        offset_list = offset_arr.tolist()

        # Cached token IDs
        n_tokens = struct.unpack('<I', buf.read(4))[0]
        token_arr = np.frombuffer(buf.read(n_tokens * 4), dtype=np.int32)
        cached_token_ids = token_arr.tolist()

        # Per-layer merged metadata
        merged_kvcolidx = []
        merged_hot_tile = []
        for layer_idx in range(num_layers):
            kv_h, kv_cols, ht_h, ht_bytes_dim = struct.unpack('<IIII', buf.read(16))
            kv_nbytes, ht_nbytes = struct.unpack('<II', buf.read(8))

            kv_raw = buf.read(kv_nbytes)
            ht_raw = buf.read(ht_nbytes)

            if kv_cols > 0:
                kv_np = np.frombuffer(kv_raw, dtype=np.uint8).reshape(kv_h, kv_cols)
                kv_tensor = torch.from_numpy(kv_np.copy()).contiguous()
            else:
                kv_tensor = torch.empty(kv_h, 0, dtype=torch.uint8)

            if ht_bytes_dim > 0:
                ht_np = np.frombuffer(ht_raw, dtype=np.uint8).reshape(ht_h, ht_bytes_dim)
                ht_tensor = torch.from_numpy(ht_np.copy()).contiguous()
            else:
                ht_tensor = torch.empty(ht_h, 0, dtype=torch.uint8)

            merged_kvcolidx.append(kv_tensor)
            merged_hot_tile.append(ht_tensor)

        t_deser = time.time()
        deser_ms = (t_deser - t_disk) * 1000

        logger.info(
            f"IndexCache loaded from disk (O_DIRECT): {path} "
            f"({file_size / 1024:.1f} KB, {num_layers} layers, "
            f"disk_read={disk_ms:.2f}ms, deserialize={deser_ms:.2f}ms)"
        )

        return {
            "merged_kvcolidx": merged_kvcolidx,
            "merged_hot_tile": merged_hot_tile,
            "offset_list": offset_list,
            "cached_token_ids": cached_token_ids,
            "num_layers": num_layers,
            "num_chunks": num_chunks,
            "disk_file_bytes": file_size,
            "disk_read_ms": disk_ms,
            "deserialize_ms": deser_ms,
        }

    def delete(self, key: str):
        path = self._key_to_path(key)
        if os.path.exists(path):
            os.unlink(path)

    def clear(self):
        for f in os.listdir(self.cache_dir):
            if f.startswith("ic_") and f.endswith(".bin"):
                os.unlink(os.path.join(self.cache_dir, f))

    def _get_libc(self):
        if not hasattr(self, '_libc') or self._libc is None:
            self._libc = ctypes.CDLL("libc.so.6")
        return self._libc

    def _alloc_aligned_buffer(self, size):
        """Allocate a page-aligned buffer using posix_memalign."""
        libc = self._get_libc()
        buf_ptr = ctypes.c_void_p()
        ret = libc.posix_memalign(ctypes.byref(buf_ptr), BLOCK_SIZE, size)
        if ret != 0:
            raise OSError(ret, "posix_memalign failed")
        buf_type = (ctypes.c_char * size).from_address(buf_ptr.value)
        return memoryview(buf_type), buf_ptr

    def _free_aligned_buffer(self, buf_ptr):
        libc = self._get_libc()
        libc.free(buf_ptr)

    def _read_direct(self, path, size):
        """Read file using O_DIRECT into aligned buffer, return bytes."""
        padded_size = _align_up(size)
        buf_view, buf_ptr = self._alloc_aligned_buffer(padded_size)
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            remaining = padded_size
            offset = 0
            while remaining > 0:
                read = os.readv(fd, [buf_view[offset:offset + remaining]])
                if read == 0:
                    break
                offset += read
                remaining -= read
            result = bytes(buf_view.cast('B')[:size])
            return result
        finally:
            if fd is not None:
                os.close(fd)
            self._free_aligned_buffer(buf_ptr)

    def _write_direct(self, path, data):
        """Write data to file using O_DIRECT with aligned buffer."""
        padded_size = _align_up(len(data))
        buf_view, buf_ptr = self._alloc_aligned_buffer(padded_size)
        fd = None
        try:
            buf_bytes = buf_view.cast('B')
            buf_bytes[:len(data)] = data
            if padded_size > len(data):
                buf_bytes[len(data):padded_size] = b'\x00' * (padded_size - len(data))
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
            remaining = padded_size
            offset = 0
            while remaining > 0:
                written = os.writev(fd, [buf_view[offset:offset + remaining]])
                if written == 0:
                    break
                offset += written
                remaining -= written
            os.fsync(fd)
        finally:
            if fd is not None:
                os.close(fd)
            self._free_aligned_buffer(buf_ptr)
