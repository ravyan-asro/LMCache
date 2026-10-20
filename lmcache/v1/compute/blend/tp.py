# Tensor-parallel consistency for CacheBlend / InfoFlow token selection
# (SIFT MLSys, 10-19).
#
# Under tensor parallelism each rank holds only its share of the attention heads,
# so a per-token score computed from local K / attention (CacheBlend's K
# deviation, InfoFlow's head-mean attention) differs between ranks.  If ranks
# pick different tokens, the rows they keep stop lining up: the row-parallel
# o_proj all-reduce then sums partial outputs of different tokens.  The scores
# are therefore summed over the TP group before ranking -- the value a single
# rank holding every head computes -- and rank 0's selection is broadcast, so
# every rank keeps exactly the same rows.  Both are no-ops at TP 1.

# Standard
import os

# Third Party
import torch

# LMCACHE_BLEND_TP_LOCAL=1 restores the old per-rank selection (each rank ranks
# tokens by its own heads), for showing the difference only.
_LOCAL = os.getenv("LMCACHE_BLEND_TP_LOCAL", "0") == "1"


def tp_rank() -> int:
    try:
        # Third Party
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except (ImportError, AssertionError):
        return 0


def tp_size() -> int:
    try:
        # Third Party
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except (ImportError, AssertionError):
        return 1


def tp_sum(x: torch.Tensor) -> torch.Tensor:
    """x summed over the TP group (x itself at TP 1)."""
    if _LOCAL or tp_size() == 1:
        return x
    # Third Party
    from vllm.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x.contiguous())


def tp_same(x: torch.Tensor) -> torch.Tensor:
    """TP rank 0's x on every rank (x must have the same shape on all ranks)."""
    if _LOCAL or tp_size() == 1:
        return x
    # Third Party
    from vllm.distributed import get_tp_group

    return get_tp_group().broadcast(x.contiguous(), src=0)


def tp_check_same(values, what: str) -> None:
    """Fail loudly (on every rank) unless all TP ranks hold the same integers.
    Used where a rank-local lookup decides the shape of a later broadcast
    (InfoFlow's candidate range comes from each rank's chunk lookup): a
    mismatch would otherwise hang or corrupt the broadcast.  Sums of x and x^2
    over the ranks agree with n*x and n*x^2 only if every rank has the same x,
    and every rank sees the same sums, so all ranks raise together."""
    if _LOCAL or tp_size() == 1:
        return
    x = torch.tensor([float(v) for v in values], dtype=torch.float64, device="cuda")
    tot = tp_sum(torch.cat([x, x * x]))
    n, k = tp_size(), len(values)
    if not torch.equal(tot[:k] * tot[:k], n * tot[k:]):
        raise RuntimeError(f"TP ranks disagree on {what}: this rank has {list(values)}")

