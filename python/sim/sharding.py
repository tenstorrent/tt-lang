# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shard placement and locality from tensor memory configuration.

:class:`~sim.ttnnsim.Tensor` carries :class:`~sim.ttnnsim.MemoryConfig` with
``ShardSpec`` or ``NdShardSpec`` whose ``shard_shape`` fields use **element**
units, consistent with tt-metal / the tensor sharding tech report.

This module classifies access as local L1, remote L1, or DRAM in **element**
counts for sharded tensors (and total **elements** for interleaved tensors).
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Optional, Tuple

from greenlet import getcurrent

from .ttnnsim import (
    NdShardSpec,
    ShardDistributionStrategy,
    ShardSpec,
    ShardingStrategy,
    Tensor,
)
from .typedefs import Count, Index, Selector, Shape, TensorKey


def normalize_tensor_key(key: TensorKey) -> Tuple[Selector, ...]:
    """Normalize a :class:`~sim.typedefs.TensorKey` to a tuple of selectors."""
    match key:
        case tuple():
            return key
        case _:
            return (key,)


def shard_origin_from_key(t: Tensor, key: Tuple[Selector, ...]) -> Shape:
    """Element-coordinate origin of the subtensor ``t[key]`` within ``t``.

    Delegates to :meth:`Tensor.element_slice_starts` (same rules as
    :meth:`Tensor.__getitem__`).
    """
    return t.element_slice_starts(key)


def _count_height_sharded_elements(
    core: Index,
    spec: ShardSpec,
    origin: Shape,
    shape: Shape,
) -> Tuple[Count, Count, Count]:
    shard_h = spec.shard_shape[-2]
    core_row_start = core * shard_h
    core_row_end = (core + 1) * shard_h

    r0 = origin[-2]
    h = shape[-2]
    w = shape[-1] if len(shape) >= 2 else 1

    local_rows = sum(1 for r in range(r0, r0 + h) if core_row_start <= r < core_row_end)
    remote_rows = h - local_rows
    return (local_rows * w, remote_rows * w, 0)


def _count_width_sharded_elements(
    core: Index,
    spec: ShardSpec,
    origin: Shape,
    shape: Shape,
) -> Tuple[Count, Count, Count]:
    shard_w = spec.shard_shape[-1]
    core_col_start = core * shard_w
    core_col_end = (core + 1) * shard_w

    c0 = origin[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    w: int = shape[-1]  # type: ignore[misc]

    local_cols = sum(1 for c in range(c0, c0 + w) if core_col_start <= c < core_col_end)
    remote_cols = w - local_cols
    return (local_cols * h, remote_cols * h, 0)


def _count_block_sharded_elements(
    core: Index,
    spec: ShardSpec,
    origin: Shape,
    shape: Shape,
) -> Tuple[Count, Count, Count]:
    num_core_cols = spec.shard_grid[-1]
    core_row = core // num_core_cols
    core_col = core % num_core_cols

    shard_h = spec.shard_shape[-2]
    shard_w = spec.shard_shape[-1]
    core_row_start = core_row * shard_h
    core_row_end = (core_row + 1) * shard_h
    core_col_start = core_col * shard_w
    core_col_end = (core_col + 1) * shard_w

    r0 = origin[-2]
    c0 = origin[-1]
    h = shape[-2]
    w = shape[-1]

    local_elements = sum(
        1
        for r in range(r0, r0 + h)
        for c in range(c0, c0 + w)
        if core_row_start <= r < core_row_end and core_col_start <= c < core_col_end
    )
    return (local_elements, h * w - local_elements, 0)


_SHARD_ELEMENT_COUNTERS: dict[
    ShardingStrategy,
    Callable[[Index, ShardSpec, Shape, Shape], Tuple[Count, Count, Count]],
] = {
    ShardingStrategy.HEIGHT_SHARDED: _count_height_sharded_elements,
    ShardingStrategy.WIDTH_SHARDED: _count_width_sharded_elements,
    ShardingStrategy.BLOCK_SHARDED: _count_block_sharded_elements,
}


def _linear_to_nd_pos(linear: Index, grid: Shape) -> Shape:
    pos: list[int] = []
    remaining = linear
    for size in reversed(grid):
        pos.append(remaining % size)
        remaining //= size
    return tuple(reversed(pos))


def _count_nd_grid2d_elements(
    core: Index,
    spec: NdShardSpec,
    origin: Shape,
    shape: Shape,
) -> Tuple[Count, Count, Count]:
    ndim = len(spec.shard_grid)
    core_pos = _linear_to_nd_pos(core, spec.shard_grid)

    owned_start = tuple(core_pos[d] * spec.shard_shape[d] for d in range(ndim))
    owned_end = tuple((core_pos[d] + 1) * spec.shard_shape[d] for d in range(ndim))

    ranges = [range(origin[d], origin[d] + shape[d]) for d in range(ndim)]
    local_elements = sum(
        1
        for pos in itertools.product(*ranges)
        if all(owned_start[d] <= pos[d] < owned_end[d] for d in range(ndim))
    )
    total = math.prod(shape)
    return (local_elements, total - local_elements, 0)


def _count_nd_round_robin_elements(
    core: Index,
    spec: NdShardSpec,
    origin: Shape,
    shape: Shape,
) -> Tuple[Count, Count, Count]:
    ndim = len(spec.shard_grid)
    num_shard_slots = math.prod(spec.shard_grid)
    num_cores = spec.num_cores if spec.num_cores is not None else num_shard_slots
    if num_cores < 1:
        raise ValueError("NdShardSpec.num_cores must be at least 1")

    shard_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        shard_strides[d] = shard_strides[d + 1] * spec.shard_grid[d + 1]

    ranges = [range(origin[d], origin[d] + shape[d]) for d in range(ndim)]
    local_elements = sum(
        1
        for el_pos in itertools.product(*ranges)
        if sum(
            (el_pos[d] // spec.shard_shape[d]) * shard_strides[d] for d in range(ndim)
        )
        % num_cores
        == core
    )
    total = math.prod(shape)
    return (local_elements, total - local_elements, 0)


def _count_nd_sharded_elements(
    core: Index,
    spec: NdShardSpec,
    origin: Shape,
    shape: Shape,
) -> Tuple[Count, Count, Count]:
    if spec.shard_grid is None:
        raise ValueError(
            "NdShardSpec.shard_grid is unset; use Tensor/from_torch so shard_grid "
            "is derived from tensor shape, or pass shard_grid explicitly."
        )
    match spec.distribution:
        case ShardDistributionStrategy.GRID_2D:
            return _count_nd_grid2d_elements(core, spec, origin, shape)
        case ShardDistributionStrategy.ROUND_ROBIN_1D:
            return _count_nd_round_robin_elements(core, spec, origin, shape)
        case _:
            raise ValueError(f"Unsupported distribution strategy: {spec.distribution}")


def count_local_remote_l1_dram(
    t: Tensor,
    current_core_linear: Index,
    *,
    origin_in_parent_elements: Optional[Shape] = None,
) -> Tuple[Count, Count, Count]:
    """Classify access to ``t`` as local L1, remote L1, or DRAM for ``current_core_linear``.

    For sharded tensors, returned counts are **element** totals.  For interleaved
    tensors, the third component is the total number of elements in ``t``
    (same as ``math.prod(t.shape)`` for physical storage).

    Args:
        t: Tensor view (often a slice of a larger sharded tensor).
        current_core_linear: 0-based linear core index.
        origin_in_parent_elements: Element-coordinate origin of ``t`` within its
            logical parent.  When ``None``, the origin is all zeros (full tensor).
    """
    mc = t.memory_config
    if mc.strategy == ShardingStrategy.INTERLEAVED:
        return (0, 0, math.prod(t.shape))

    eshape = t.shape
    origin = (
        origin_in_parent_elements
        if origin_in_parent_elements is not None
        else (0,) * len(eshape)
    )

    if mc.strategy == ShardingStrategy.ND_SHARDED:
        if mc.nd_shard_spec is None:
            raise ValueError("ND_SHARDED requires nd_shard_spec")
        return _count_nd_sharded_elements(
            current_core_linear, mc.nd_shard_spec, origin, eshape
        )

    if mc.shard_spec is None:
        return (0, 0, math.prod(t.shape))

    counter = _SHARD_ELEMENT_COUNTERS.get(mc.strategy)
    if counter is None:
        raise ValueError(f"Unsupported sharding strategy: {mc.strategy}")
    return counter(current_core_linear, mc.shard_spec, origin, eshape)


def try_count_locality(t: Tensor) -> Optional[Tuple[int, int, int]]:
    """Return (local_l1, remote_l1, dram) element counts for the current kernel core.

    Returns ``None`` when called outside a kernel context (no ``_sim_core`` tag on
    the current greenlet) or when the tensor has no memory config information.
    The returned counts are in **elements**, not tiles.
    """
    core: Optional[int] = getattr(getcurrent(), "_sim_core", None)
    if core is None:
        return None
    origin: Optional[Tuple[int, ...]] = getattr(t, "_element_origin", None)
    return count_local_remote_l1_dram(t, core, origin_in_parent_elements=origin)


def count_local_remote_l1_dram_for_getitem(
    parent: Tensor,
    key: TensorKey,
    core: Index,
) -> Tuple[Count, Count, Count]:
    """Like :func:`count_local_remote_l1_dram` on ``parent[key]`` with the correct origin.

    ``key`` must be valid for :meth:`Tensor.__getitem__`.  :func:`shard_origin_from_key`
    supplies the element-space origin in ``parent``.
    """
    normalized = normalize_tensor_key(key)
    child = parent[key]
    origin = shard_origin_from_key(parent, normalized)
    return count_local_remote_l1_dram(child, core, origin_in_parent_elements=origin)
