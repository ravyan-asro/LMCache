"""IndexCache configuration — reads from LMCACHE_INDEXCACHE_* env vars."""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class IndexCacheConfig:
    page_size: int = 64
    threshold: float = 0.001       # LA hot-tile threshold
    ca_threshold: float = 0.01     # CA important-page threshold
    ca_consistency: float = 0.1    # CA strong-consistency threshold
    if_thresh: bool = True         # use threshold mode (vs top-pct)
    local_pct: float = 0.5        # fraction when if_thresh=False
    misbehaving_heads: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "IndexCacheConfig":
        return cls(
            page_size=int(os.getenv("LMCACHE_INDEXCACHE_PAGE_SIZE", "64")),
            threshold=float(os.getenv("LMCACHE_INDEXCACHE_THRESHOLD", "0.001")),
            ca_threshold=float(os.getenv("LMCACHE_INDEXCACHE_CA_THRESHOLD", "0.01")),
            ca_consistency=float(os.getenv("LMCACHE_INDEXCACHE_CA_CONSISTENCY", "0.1")),
            if_thresh=os.getenv("LMCACHE_INDEXCACHE_IF_THRESH", "1") == "1",
            local_pct=float(os.getenv("LMCACHE_INDEXCACHE_LOCAL_PCT", "0.5")),
        )

    @property
    def misbehaving_heads_set(self) -> set:
        return set(self.misbehaving_heads)
