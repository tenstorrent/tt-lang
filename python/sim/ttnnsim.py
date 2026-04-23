# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal TTNN simulator built on top of PyTorch.

This module provides a thin compatibility layer that mirrors a subset of
TTNN's public API, sufficient to exercise simulator examples and tests.

Scope:
- Device open/close (no-op, returns simple handle)
- Tensor wrapper over torch.Tensor with shape/dtype access
- Random/empty tensor creation
- Helpers to convert to native torch tensors
- Constants for tile layout and tile size
- Core coordinate / range / grid types, ``TensorSpec``, ``BufferType``,
  ``TensorMemoryLayout`` (aligned with tt-metal / tensor sharding examples)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import torch

# Try to import actual ttnn, track if availability
TTNN_AVAILABLE: bool
try:
    import ttnn  # type: ignore[reportMissingImports]

    TTNN_AVAILABLE = True  # type: ignore[reportConstantRedefinition]
except ImportError:
    TTNN_AVAILABLE = False  # type: ignore[reportConstantRedefinition]

from .constants import TILE_SHAPE
from .typedefs import Count, IndexType, Selector, Shape, TensorKey

# Public constants (mirror TTL constants)
TILE_SIZE: int = TILE_SHAPE[0]
TILE_LAYOUT = IndexType.TILE
ROW_MAJOR_LAYOUT = IndexType.ROW_MAJOR


class ShardingStrategy(Enum):
    """Tensor memory layout sharding strategy."""

    INTERLEAVED = auto()
    HEIGHT_SHARDED = auto()
    WIDTH_SHARDED = auto()
    BLOCK_SHARDED = auto()
    ND_SHARDED = auto()


class ShardStrategy(Enum):
    """Sharding strategy passed to create_sharded_memory_config.

    Mirrors ttnn.ShardStrategy.  Maps to ShardingStrategy internally.
    """

    HEIGHT = auto()
    WIDTH = auto()
    BLOCK = auto()


class ShardOrientation(Enum):
    """Order in which cores are traversed when reading/writing shards.

    Mirrors ttnn.ShardOrientation.
    """

    ROW_MAJOR = auto()
    COL_MAJOR = auto()


class ShardDistributionStrategy(Enum):
    """How shards are mapped to cores for ND_SHARDED tensors.

    ROUND_ROBIN_1D: shards are numbered row-major and assigned to cores
        round-robin (shard i goes to core i % num_cores).  shard_grid is
        N-D and encodes the number of shards in each tensor dimension;
        math.prod(shard_grid) is the total number of cores.
    GRID_2D: core at N-D grid position (p0, p1, ...) owns the shard at
        the same position.  Generalises BLOCK_SHARDED to N dimensions.
    """

    ROUND_ROBIN_1D = auto()
    GRID_2D = auto()


class BufferType(Enum):
    """Buffer placement for tensor storage (mirrors ``ttnn.BufferType``)."""

    DRAM = auto()
    L1 = auto()


class TensorMemoryLayout(Enum):
    """How tensor data is laid out in memory (mirrors ``ttnn.TensorMemoryLayout``).

    See the `tensor sharding tech report
    <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_sharding/tensor_sharding.md>`__.
    """

    INTERLEAVED = auto()
    HEIGHT_SHARDED = auto()
    WIDTH_SHARDED = auto()
    BLOCK_SHARDED = auto()
    ND_SHARDED = auto()


class ShardSpec:
    """Shard grid and per-shard shape (ttnn / tt-metal API).

    Supported forms:

    - Legacy simulator: ``ShardSpec(shard_grid=(n,), shard_shape=(h, w), ...)``
    - tt-metal positional: ``ShardSpec(core_range_set, (h, w), ShardOrientation.ROW_MAJOR)``
    - tt-metal keywords: ``ShardSpec(grid=..., shard_shape=[h, w], shard_orientation=...)``
      (``shard_grid`` is derived from ``grid`` and :class:`TensorMemoryLayout` when using
      :class:`MemoryConfig`).

    ``shard_shape`` uses **element** units; see the `tensor sharding tech report
    <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_sharding/tensor_sharding.md>`__.
    """

    __slots__ = ("_shard_grid", "shard_shape", "orientation", "grid")

    def __init__(
        self,
        *args: Any,
        shard_grid: Optional[Shape] = None,
        shard_shape: Optional[Sequence[int]] = None,
        orientation: ShardOrientation = ShardOrientation.ROW_MAJOR,
        grid: Optional["CoreRangeSet"] = None,
        shard_orientation: Optional[ShardOrientation] = None,
    ) -> None:
        ori = shard_orientation if shard_orientation is not None else orientation
        # CoreRangeSet is defined later in this module; avoid isinstance forward-ref.
        if args and type(args[0]).__name__ == "CoreRangeSet":
            self.grid = args[0]
            self.shard_shape = tuple(args[1])
            self.orientation = args[2] if len(args) > 2 else ori
            self._shard_grid = None
            return
        sg = shard_grid
        ss = shard_shape
        gr = grid
        if args:
            if sg is None:
                sg = args[0]
            if ss is None and len(args) > 1:
                ss = args[1]
            if len(args) > 2 and isinstance(args[2], ShardOrientation):
                ori = args[2]
        if ss is None:
            raise TypeError("shard_shape is required")
        self.shard_shape = tuple(int(x) for x in ss)
        self.orientation = ori
        self.grid = gr
        self._shard_grid = tuple(int(x) for x in sg) if sg is not None else None
        if self._shard_grid is None and self.grid is None:
            raise TypeError(
                "ShardSpec requires shard_grid=, or grid=, or CoreRangeSet as first arg"
            )

    @property
    def shard_grid(self) -> Shape:
        if self._shard_grid is None:
            raise ValueError(
                "ShardSpec uses a CoreRangeSet grid; build MemoryConfig(TensorMemoryLayout, BufferType, spec) to resolve shard_grid"
            )
        return self._shard_grid

    def with_resolved_shard_grid(self, layout: "TensorMemoryLayout") -> ShardSpec:
        """Return a spec with ``shard_grid`` set from ``grid`` and layout (tt-metal path)."""
        if self._shard_grid is not None:
            return self
        if self.grid is None:
            raise ValueError("ShardSpec has no CoreRangeSet grid to resolve")
        cg = core_range_set_to_core_grid(self.grid)
        if layout in (
            TensorMemoryLayout.HEIGHT_SHARDED,
            TensorMemoryLayout.WIDTH_SHARDED,
        ):
            sg: Shape = (cg.num_cores,)
        elif layout == TensorMemoryLayout.BLOCK_SHARDED:
            sg = (cg.y, cg.x)
        else:
            raise ValueError(
                f"Cannot resolve ShardSpec shard_grid for TensorMemoryLayout {layout}"
            )
        return ShardSpec(
            shard_grid=sg,
            shard_shape=self.shard_shape,
            orientation=self.orientation,
            grid=self.grid,
        )

    def __eq__(self, other: object) -> bool:
        match other:
            case ShardSpec():
                return (
                    self._shard_grid == other._shard_grid
                    and self.shard_shape == other.shard_shape
                    and self.orientation == other.orientation
                    and self.grid == other.grid
                )
            case _:
                return False

    def __repr__(self) -> str:
        return (
            f"ShardSpec(shard_grid={self._shard_grid!r}, shard_shape={self.shard_shape!r}, "
            f"orientation={self.orientation!r}, grid={self.grid!r})"
        )


@dataclass
class NdShardSpec:
    """Shard specification for ND_SHARDED tensors (simulator + tech report style).

    Matches the tensor sharding tech report surface API:

    - ``shard_shape``: extent of one shard along each tensor dimension in
      **element** units.
    - ``core_ranges``: which device cores participate (optional in the simulator
      when only locality math is needed).

    If ``shard_grid`` is omitted, it is derived when a :class:`Tensor` is
    constructed as ``tensor_shape[i] // shard_shape[i]`` (each full tensor
    dimension must divide evenly by ``shard_shape[i]``).

    ``distribution`` defaults to :data:`ShardDistributionStrategy.ROUND_ROBIN_1D`,
    matching tt-metal's Python binding for ``NdShardSpec`` (see ``tensor.cpp``).
    When ``shard_grid`` is omitted and derived from tensor shape in
    :meth:`with_resolved_shard_grid`, the result uses :data:`ShardDistributionStrategy.GRID_2D`
    (dense N-D shard boxes), which matches the tensor sharding tech report examples
    that only specify ``shard_shape``.


    ``num_cores`` applies only to ROUND_ROBIN (modulus for shard assignment).
    """

    shard_shape: Shape
    core_ranges: Optional["CoreRangeSet"] = None
    shard_grid: Optional[Shape] = None
    distribution: ShardDistributionStrategy = ShardDistributionStrategy.ROUND_ROBIN_1D
    num_cores: Optional[int] = None

    def __post_init__(self) -> None:
        # Accept list inputs like the tech report (``shard_shape=[...]``).
        object.__setattr__(self, "shard_shape", tuple(self.shard_shape))
        if self.shard_grid is not None:
            object.__setattr__(self, "shard_grid", tuple(self.shard_grid))

    def with_resolved_shard_grid(self, tensor_shape: Shape) -> NdShardSpec:
        """Return a copy with ``shard_grid`` set from ``tensor_shape`` and ``shard_shape``."""
        if self.shard_grid is not None:
            return self
        if len(tensor_shape) != len(self.shard_shape):
            raise ValueError(
                f"tensor rank {len(tensor_shape)} does not match shard_shape rank {len(self.shard_shape)}"
            )
        grid: list[int] = []
        for i, (ts, ss) in enumerate(zip(tensor_shape, self.shard_shape)):
            if ss < 1:
                raise ValueError(f"shard_shape[{i}] must be positive, got {ss}")
            if ts % ss != 0:
                raise ValueError(
                    f"tensor dimension {i} size {ts} is not divisible by shard_shape[{i}]={ss}"
                )
            grid.append(ts // ss)
        # Implicit shard_grid from tensor shape implies dense grid semantics (tech report).
        return replace(
            self,
            shard_grid=tuple(grid),
            distribution=ShardDistributionStrategy.GRID_2D,
        )


class MemoryConfig:
    """Memory configuration for a tensor (simulator + tt-metal style).

    Simulator style::

        MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=...)

    tt-metal style (three positional args; see tensor sharding tech report)::

        MemoryConfig(
            TensorMemoryLayout.HEIGHT_SHARDED,
            BufferType.L1,
            ShardSpec(...),
        )
    """

    __slots__ = (
        "strategy",
        "shard_spec",
        "nd_shard_spec",
        "buffer_type",
        "tensor_memory_layout",
    )

    def __init__(
        self,
        *args: Any,
        strategy: Optional[ShardingStrategy] = None,
        shard_spec: Optional[ShardSpec] = None,
        nd_shard_spec: Optional[NdShardSpec] = None,
        buffer_type: BufferType = BufferType.DRAM,
        tensor_memory_layout: Optional[TensorMemoryLayout] = None,
    ) -> None:
        if (
            len(args) == 3
            and isinstance(args[0], TensorMemoryLayout)
            and isinstance(args[1], BufferType)
        ):
            layout_tt, buf, spec = args[0], args[1], args[2]
            self.buffer_type = buf
            self.tensor_memory_layout = layout_tt
            if isinstance(spec, ShardSpec):
                resolved = spec.with_resolved_shard_grid(layout_tt)
                self.strategy = _tensor_memory_layout_to_sharding_strategy(layout_tt)
                self.shard_spec = resolved
                self.nd_shard_spec = None
            elif isinstance(spec, NdShardSpec):
                self.strategy = ShardingStrategy.ND_SHARDED
                self.shard_spec = None
                self.nd_shard_spec = spec
            else:
                raise TypeError(
                    f"Third argument must be ShardSpec or NdShardSpec, got {type(spec)}"
                )
            return

        st = strategy if strategy is not None else (args[0] if len(args) == 1 else None)
        if st is None:
            raise TypeError(
                "MemoryConfig requires strategy=... or (TensorMemoryLayout, BufferType, ShardSpec|NdShardSpec)"
            )
        self.strategy = st
        self.shard_spec = shard_spec
        self.nd_shard_spec = nd_shard_spec
        self.buffer_type = buffer_type
        self.tensor_memory_layout = tensor_memory_layout

    def __eq__(self, other: object) -> bool:
        match other:
            case MemoryConfig():
                return (
                    self.strategy == other.strategy
                    and self.shard_spec == other.shard_spec
                    and self.nd_shard_spec == other.nd_shard_spec
                    and self.buffer_type == other.buffer_type
                    and self.tensor_memory_layout == other.tensor_memory_layout
                )
            case _:
                return False

    def __repr__(self) -> str:
        return (
            f"MemoryConfig(strategy={self.strategy!r}, shard_spec={self.shard_spec!r}, "
            f"nd_shard_spec={self.nd_shard_spec!r}, buffer_type={self.buffer_type!r}, "
            f"tensor_memory_layout={self.tensor_memory_layout!r})"
        )


def _tensor_memory_layout_to_sharding_strategy(
    layout: TensorMemoryLayout,
) -> ShardingStrategy:
    return {
        TensorMemoryLayout.INTERLEAVED: ShardingStrategy.INTERLEAVED,
        TensorMemoryLayout.HEIGHT_SHARDED: ShardingStrategy.HEIGHT_SHARDED,
        TensorMemoryLayout.WIDTH_SHARDED: ShardingStrategy.WIDTH_SHARDED,
        TensorMemoryLayout.BLOCK_SHARDED: ShardingStrategy.BLOCK_SHARDED,
        TensorMemoryLayout.ND_SHARDED: ShardingStrategy.ND_SHARDED,
    }[layout]


@dataclass
class CoreGrid:
    """2-D core grid.  Mirrors ttnn.CoreGrid.

    Attributes:
        y: Number of core rows.
        x: Number of core columns.
    """

    y: int
    x: int

    @property
    def num_cores(self) -> int:
        return self.y * self.x


def broadcast_tensors(
    left_tensors: List["Tensor"],
    right_tensors: List["Tensor"],
    left_shape: Shape,
    right_shape: Shape,
    op: Any,
) -> List["Tensor"]:
    """Apply binary operation to tensor lists with broadcasting.

    Stacks tensors into batched tensors, reshapes according to tile grid shapes,
    applies PyTorch broadcasting, and flattens back to list of tensors.

    Args:
        left_tensors: List of left operand tensors
        right_tensors: List of right operand tensors
        left_shape: Tile grid shape for left operand (e.g., (4, 4) for 16 tiles)
        right_shape: Tile grid shape for right operand
        op: Binary operation to apply (e.g., operator.add)

    Returns:
        List of result tensors after broadcasting
    """
    # Extract underlying torch tensors
    left_torch: List[torch.Tensor] = [
        cast(torch.Tensor, getattr(t, "_tensor", t)) for t in left_tensors
    ]
    right_torch: List[torch.Tensor] = [
        cast(torch.Tensor, getattr(t, "_tensor", t)) for t in right_tensors
    ]

    # Stack into batched tensors
    left_batched = torch.stack(left_torch)
    right_batched = torch.stack(right_torch)

    # Reshape to include tile grid dimensions
    left_reshaped = left_batched.reshape(*left_shape, *left_batched.shape[1:])
    right_reshaped = right_batched.reshape(*right_shape, *right_batched.shape[1:])

    # Apply operation with PyTorch broadcasting
    result_batched = op(left_reshaped, right_reshaped)

    # Flatten all grid dimensions back to a flat tile list
    grid_ndim = len(left_shape)
    num_result_tiles = 1
    for d in result_batched.shape[:grid_ndim]:
        num_result_tiles *= d
    result_flat = result_batched.reshape(
        num_result_tiles, *result_batched.shape[grid_ndim:]
    )

    # Wrap each result tile in Tensor
    return [Tensor(result_flat[i]) for i in range(num_result_tiles)]


DRAM_MEMORY_CONFIG: MemoryConfig = MemoryConfig(strategy=ShardingStrategy.INTERLEAVED)
L1_MEMORY_CONFIG: MemoryConfig = MemoryConfig(strategy=ShardingStrategy.INTERLEAVED)

# Type aliases for binary operations
Scalar = Union[float, int]
TensorOrScalar = Union["Tensor", float, int]


class CoreCoord:
    """Logical core coordinate (ttnn API).

    Mirrors tt-metal ``CoreCoord``: first component is the X (column) index,
    second is the Y (row) index, consistent with :class:`CoreGrid` ``(y, x)``
    sizing elsewhere in this module.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"CoreCoord({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        match other:
            case CoreCoord():
                return self.x == other.x and self.y == other.y
            case _:
                return False

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class CoreRange:
    """Inclusive rectangular range of cores (ttnn API)."""

    __slots__ = ("start", "end")

    def __init__(self, start: CoreCoord, end: CoreCoord) -> None:
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"CoreRange({self.start!r}, {self.end!r})"

    def __eq__(self, other: object) -> bool:
        match other:
            case CoreRange():
                return self.start == other.start and self.end == other.end
            case _:
                return False

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def num_cores(self) -> Count:
        """Number of cores in this range."""
        x_range = self.end.x - self.start.x + 1
        y_range = self.end.y - self.start.y + 1
        return x_range * y_range


class CoreRangeSet:
    """Collection of :class:`CoreRange` regions (ttnn API).

    Construct with a list or a ``set`` of ranges, e.g.
    ``CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(0, 3))})``.
    """

    __slots__ = ("_ranges",)

    def __init__(
        self,
        ranges: Union[
            List[CoreRange],
            Set[CoreRange],
            FrozenSet[CoreRange],
            Iterable[CoreRange],
        ],
    ) -> None:
        if isinstance(ranges, list):
            self._ranges = ranges
        else:
            self._ranges = sorted(
                ranges,
                key=lambda r: (r.start.y, r.start.x, r.end.y, r.end.x),
            )

    def ranges(self) -> List[CoreRange]:
        """Core ranges (deterministic order)."""
        return self._ranges

    def num_cores(self) -> Count:
        """Total cores across all ranges."""
        return sum(r.num_cores() for r in self._ranges)

    def __repr__(self) -> str:
        return f"CoreRangeSet({self._ranges!r})"

    def __eq__(self, other: object) -> bool:
        match other:
            case CoreRangeSet():
                return self._ranges == other._ranges
            case _:
                return False


def num_cores_to_corerangeset(
    target_num_cores: int,
    grid_size: Sequence[int],
    row_wise: bool = True,
) -> CoreRangeSet:
    """Pick ``target_num_cores`` cores in a logical grid (ttnn API subset).

    ``grid_size`` is ``[num_rows, num_cols]`` (Y then X in :class:`CoreCoord`).
    Prefer a single row of cores along X when ``target_num_cores <= num_cols``;
    otherwise a single column along Y when ``target_num_cores <= num_rows``;
    otherwise take a bounding box over cores visited in row-major order (sim
    approximation).
    """
    if len(grid_size) != 2:
        raise ValueError("grid_size must be a sequence of two ints")
    rows, cols = int(grid_size[0]), int(grid_size[1])
    if target_num_cores < 1:
        raise ValueError("target_num_cores must be at least 1")
    capacity = rows * cols
    if target_num_cores > capacity:
        raise ValueError(
            f"target_num_cores {target_num_cores} exceeds grid capacity {capacity}"
        )
    if row_wise and target_num_cores <= cols:
        return CoreRangeSet(
            [
                CoreRange(
                    CoreCoord(0, 0),
                    CoreCoord(target_num_cores - 1, 0),
                )
            ]
        )
    if row_wise and target_num_cores <= rows:
        return CoreRangeSet(
            [
                CoreRange(
                    CoreCoord(0, 0),
                    CoreCoord(0, target_num_cores - 1),
                )
            ]
        )
    coords: List[CoreCoord] = []
    for y in range(rows):
        for x in range(cols):
            if len(coords) >= target_num_cores:
                break
            coords.append(CoreCoord(x, y))
        if len(coords) >= target_num_cores:
            break
    min_x = min(c.x for c in coords)
    max_x = max(c.x for c in coords)
    min_y = min(c.y for c in coords)
    max_y = max(c.y for c in coords)
    return CoreRangeSet([CoreRange(CoreCoord(min_x, min_y), CoreCoord(max_x, max_y))])


def core_range_set_to_core_grid(core_ranges: CoreRangeSet) -> CoreGrid:
    """Bounding :class:`CoreGrid` for a :class:`CoreRangeSet` (single-box case).

    Uses the axis-aligned bounding box of all ranges.  For sharding helpers
    this matches typical tt-metal examples with one rectangular ``CoreRange``.
    """
    ranges = core_ranges.ranges()
    if not ranges:
        raise ValueError("CoreRangeSet is empty")
    min_x = min(r.start.x for r in ranges)
    max_x = max(r.end.x for r in ranges)
    min_y = min(r.start.y for r in ranges)
    max_y = max(r.end.y for r in ranges)
    return CoreGrid(y=max_y - min_y + 1, x=max_x - min_x + 1)


def _distribute_cores_across_dims(num_cores: int, k: int) -> Tuple[int, ...]:
    """Split ``num_cores`` into ``k`` positive integers whose product is ``num_cores``."""
    if k <= 0:
        return ()
    if k == 1:
        return (num_cores,)
    factors = [1] * k
    n = num_cores
    p = 2
    i = 0
    while n > 1:
        if p * p > n:
            factors[i % k] *= n
            break
        if n % p == 0:
            factors[i % k] *= p
            n //= p
            i += 1
        else:
            p += 1
    return tuple(factors)


def _nd_shard_spec_for_dims(
    shape: Shape,
    shard_dims: Sequence[int],
    core_ranges: CoreRangeSet,
) -> NdShardSpec:
    """Build :class:`NdShardSpec` for experimental ND sharding (GRID_2D)."""
    ndim = len(shape)
    dims_sorted = sorted(shard_dims)
    for d in dims_sorted:
        if d < 0 or d >= ndim:
            raise ValueError(f"shard dim {d} out of range for rank {ndim}")
    num_cores = core_ranges.num_cores()
    if num_cores < 1:
        raise ValueError("core range must include at least one core")
    k = len(dims_sorted)
    factors = sorted(_distribute_cores_across_dims(num_cores, k), reverse=True)
    shard_grid_list = [1] * ndim
    for dim, factor in zip(dims_sorted, factors):
        shard_grid_list[dim] = factor
    shard_grid_t = tuple(shard_grid_list)
    shard_shape = tuple(
        (
            (shape[i] + shard_grid_t[i] - 1) // shard_grid_t[i]
            if shard_grid_t[i] > 1
            else shape[i]
        )
        for i in range(ndim)
    )
    return NdShardSpec(
        shard_shape=shard_shape,
        shard_grid=shard_grid_t,
        distribution=ShardDistributionStrategy.GRID_2D,
        core_ranges=core_ranges,
    )


@dataclass(frozen=True)
class TensorSpec:
    """Tensor shape/dtype/layout/buffer metadata with optional sharding (ttnn API).

    Use ``height_sharded`` / ``width_sharded`` / ``block_sharded`` /
    ``sharded_across_dims`` / ``nd_sharded`` to attach a :class:`MemoryConfig`,
    then pass the spec to :func:`from_torch` (see tt-metal tensor sharding examples).
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32
    layout: IndexType = TILE_LAYOUT
    buffer_type: BufferType = BufferType.DRAM
    memory_layout: TensorMemoryLayout = TensorMemoryLayout.INTERLEAVED
    memory_config: Optional[MemoryConfig] = None
    core_ranges: Optional[CoreRangeSet] = None

    def height_sharded(self, core_ranges: CoreRangeSet) -> TensorSpec:
        """2-D height sharding: collapse leading dims to height, shard along height."""
        cg = core_range_set_to_core_grid(core_ranges)
        mc = create_sharded_memory_config(
            self.shape,
            cg,
            ShardStrategy.HEIGHT,
            orientation=ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        return replace(
            self,
            memory_layout=TensorMemoryLayout.HEIGHT_SHARDED,
            memory_config=mc,
            core_ranges=core_ranges,
        )

    def width_sharded(self, core_ranges: CoreRangeSet) -> TensorSpec:
        """2-D width sharding: collapse leading dims to height, shard along width."""
        cg = core_range_set_to_core_grid(core_ranges)
        mc = create_sharded_memory_config(
            self.shape,
            cg,
            ShardStrategy.WIDTH,
            orientation=ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        return replace(
            self,
            memory_layout=TensorMemoryLayout.WIDTH_SHARDED,
            memory_config=mc,
            core_ranges=core_ranges,
        )

    def block_sharded(self, core_ranges: CoreRangeSet) -> TensorSpec:
        """2-D block sharding on a core grid."""
        cg = core_range_set_to_core_grid(core_ranges)
        mc = create_sharded_memory_config(
            self.shape,
            cg,
            ShardStrategy.BLOCK,
            orientation=ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        return replace(
            self,
            memory_layout=TensorMemoryLayout.BLOCK_SHARDED,
            memory_config=mc,
            core_ranges=core_ranges,
        )

    def sharded_across_dims(
        self,
        dims: Sequence[int],
        core_ranges: CoreRangeSet,
    ) -> TensorSpec:
        """Experimental ND sharding across the given tensor dimensions."""
        nd = _nd_shard_spec_for_dims(self.shape, dims, core_ranges)
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=nd)
        return replace(
            self,
            memory_layout=TensorMemoryLayout.ND_SHARDED,
            memory_config=mc,
            core_ranges=core_ranges,
        )

    def nd_sharded(
        self,
        shard_shape: Shape,
        core_ranges: CoreRangeSet,
    ) -> TensorSpec:
        """ND sharding with explicit per-dimension shard sizes (element units).

        Matches the tensor sharding tech report style: ``shard_shape`` gives the
        extent of one shard along each dimension of :attr:`shape`; device
        placement is ``core_ranges``. The logical shard count per dimension is
        ``shape[i] // shard_shape[i]`` (each tensor dimension must divide evenly).

        For ND sharding derived from ``shard_dims`` and core count instead, use
        :meth:`sharded_across_dims`.
        """
        nd = NdShardSpec(shard_shape=tuple(shard_shape), core_ranges=core_ranges)
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=nd)
        return replace(
            self,
            memory_layout=TensorMemoryLayout.ND_SHARDED,
            memory_config=mc,
            core_ranges=core_ranges,
        )


# Dtype aliases
bfloat16: torch.dtype = torch.bfloat16
float32: torch.dtype = torch.float32

# Original value saved so set_matmul_promote_bf16 can restore it.
_original_bfloat16: torch.dtype = torch.bfloat16


def set_matmul_promote_bf16(value: bool) -> None:
    """Redirect bfloat16 to float32 for the entire process when the flag is active.

    When enabled, both ``torch.bfloat16`` and the module-level ``bfloat16``
    alias are rebound to ``torch.float32``.  Any subsequent use of
    ``dtype=torch.bfloat16`` or ``dtype=ttnn.bfloat16`` in the user script
    therefore creates float32 tensors natively, with no dispatch overhead or
    casting.  Note: this doubles tensor memory usage; avoid for very large
    examples on memory-constrained machines.  When disabled the originals are
    restored.
    """
    global bfloat16
    if value:
        torch.bfloat16 = torch.float32
        bfloat16 = torch.float32
    else:
        torch.bfloat16 = _original_bfloat16
        bfloat16 = _original_bfloat16


class Device:
    """Simple device handle.

    In the simulator, this is a no-op placeholder with an id.
    """

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id

    def __repr__(self) -> str:
        return f"Device(id={self.device_id})"

    def compute_with_storage_grid_size(self) -> CoreCoord:
        """Return the compute grid size for the device.

        In the simulator, returns a fixed 8x8 grid to match the default
        'auto' grid size used by kernels.

        Returns:
            CoreCoord: Grid size (x=8, y=8)
        """
        return CoreCoord(8, 8)


def open_device(device_id: int = 0) -> Device:
    """Open a simulated device (no-op)."""
    return Device(device_id)


def close_device(device: Device) -> None:
    """Close a simulated device (no-op)."""
    # Nothing to do in simulator
    return None


# -------------------------------------------------------------------------
# Multi-device (mesh) support
#
# The simulator treats multi-device operations as single-device: all mesh
# and sharding APIs are stubs that accept the same arguments as the real
# ttnn but otherwise do nothing.  Kernels execute on the full tensor as if
# there were a single device, which is sufficient for functional correctness
# testing.
# -------------------------------------------------------------------------


def GetNumAvailableDevices() -> int:
    """Return the configured number of simulated devices."""
    from .context import get_context

    return get_context().config.num_devices


def set_num_devices(n: int) -> None:
    """Set the number of devices returned by GetNumAvailableDevices."""
    from .context import get_context

    if n < 1:
        raise ValueError(f"num_devices must be >= 1, got {n}")
    get_context().config.num_devices = n


class FabricConfig:
    """Fabric interconnect configuration constants (mirrors ttnn.FabricConfig).

    In the simulator the fabric is not modeled, so these constants are accepted
    by :func:`set_fabric_config` for API compatibility only.
    """

    FABRIC_1D = "FABRIC_1D"


def set_fabric_config(config: Any) -> None:
    """Configure the inter-device fabric (no-op in the simulator).

    The fabric controls physical routing of data across the NoC between
    devices.  The functional simulator cares only about correct output values,
    not about which links data travels over, so this call has no effect.
    """


class MeshShape:
    """Logical shape of a device mesh (rows x cols)."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols


class MeshDevice:
    """Handle for a simulated mesh of ``rows * cols`` virtual devices."""

    def __init__(self, shape: MeshShape) -> None:
        self.shape = shape
        self.num_devices = shape.rows * shape.cols


def open_mesh_device(shape: MeshShape) -> MeshDevice:
    """Open a simulated mesh device (stub)."""
    return MeshDevice(shape)


def close_mesh_device(mesh: MeshDevice) -> None:
    """Close a simulated mesh device (no-op)."""


@dataclass
class MeshShardInfo:
    """Mesh-level partition metadata attached to a Tensor by ShardTensorToMesh.

    Records which axis of the full tensor is partitioned across devices and
    how many device partitions exist.  Kept separate from MemoryConfig to
    avoid conflating inter-device distribution with intra-device sharding
    strategies (HEIGHT_SHARDED, WIDTH_SHARDED, etc.).
    """

    dim: int
    num_devices: int


class TensorToMesh:
    """Base class for mesh mappers passed to :func:`from_torch` (mirrors ``ttnn.TensorToMesh``)."""


class ShardTensorToMesh(TensorToMesh):
    """Mapper for from_torch: shards a tensor across mesh devices along ``dim``.

    When passed to :func:`from_torch`, the resulting :class:`Tensor` carries a
    :class:`MeshShardInfo` recording the partition axis and device count.
    :func:`all_reduce` reads this metadata to perform the reduction without
    consulting global device-count state or intra-device sharding strategies.
    """

    def __init__(self, mesh: MeshDevice, dim: int) -> None:
        self.mesh = mesh
        self.dim = dim


class ReplicateTensorToMesh(TensorToMesh):
    """Mapper for from_torch: replicates a tensor identically across all devices.

    In the simulator there is no physical device split, so the full tensor
    already represents the replicated copy.  Passing this to :func:`from_torch`
    is a no-op beyond accepting the argument for API compatibility.
    """

    def __init__(self, mesh: MeshDevice) -> None:
        pass


class ConcatMeshToTensor:
    """Composer for to_torch: reconstructs a full tensor from per-device shards.

    In the simulator the tensor is never physically split across devices, so
    :func:`to_torch` already returns the full underlying tensor regardless of
    this composer.  The argument is accepted for API compatibility.
    """

    def __init__(self, mesh: MeshDevice, dim: int) -> None:
        pass


def tile_shape_from_tensor(t: "Tensor") -> Shape:
    """Return the tile-grid shape of a tensor.

    For tiled tensors the last two element dimensions are divided by TILE_SHAPE
    (treating H==1 or W==1 as degenerate single-tile dimensions); leading
    dimensions are returned as-is.  For 1-D tensors the single element dimension
    is divided by TILE_SHAPE[0].
    """
    s = t.shape
    if len(s) == 1:
        w = s[0]
        tk = 1 if w == 1 else w // TILE_SHAPE[0]
        return (tk,)
    h, w = s[-2], s[-1]
    tm = 1 if h == 1 else h // TILE_SHAPE[0]
    tk = 1 if w == 1 else w // TILE_SHAPE[1]
    if len(s) > 2:
        return (*s[:-2], tm, tk)
    return (tm, tk)


def tile_count_from_tensor(t: "Tensor") -> int:
    """Return the number of logical units a Tensor represents.

    For row-major tensors each scalar is a unit, so the count equals the total
    number of elements: math.prod(shape).

    For tiled tensors, delegates to :func:`tile_shape_from_tensor` and
    multiplies the resulting tile-grid dimensions.
    """
    if t.layout == ROW_MAJOR_LAYOUT:
        return math.prod(t.shape)
    return math.prod(tile_shape_from_tensor(t))


def check_count_match(
    src_count: int,
    dst_count: int,
    layout: IndexType,
    src_desc: str,
    dst_desc: str,
) -> None:
    """Raise ValueError if src_count != dst_count, with a layout-aware message.

    Args:
        src_count: Logical unit count of the source (tiles or elements).
        dst_count: Logical unit count of the destination.
        layout: Layout that determines the unit name ("tile" or "element").
        src_desc: Human-readable description of the source (e.g. "Tensor shape (32, 32)").
        dst_desc: Human-readable description of the destination.

    Raises:
        ValueError: If src_count != dst_count.
    """
    if src_count == dst_count:
        return
    unit = "element" if layout == ROW_MAJOR_LAYOUT else "tile"
    raise ValueError(
        f"{src_desc} does not match {dst_desc} "
        f"({unit} counts: {src_count} vs {dst_count})"
    )


def normalize_selector_to_slice(selector: Selector) -> slice:
    """Convert an integer index to a unit slice, or return slice as-is.

    Shared by :meth:`Tensor._normalize_index` and :mod:`sim.sharding` when
    interpreting :class:`~sim.typedefs.Selector` values.
    """
    match selector:
        case int():
            return slice(selector, selector + 1)
        case _:
            return selector


def _maybe_resolve_nd_shard_spec_for_tensor(
    tensor_shape: Shape, memory_config: MemoryConfig
) -> MemoryConfig:
    """Fill ``NdShardSpec.shard_grid`` from tensor shape when it was omitted."""
    if memory_config.strategy != ShardingStrategy.ND_SHARDED:
        return memory_config
    nd = memory_config.nd_shard_spec
    if nd is None or nd.shard_grid is not None:
        return memory_config
    resolved_nd = nd.with_resolved_shard_grid(tensor_shape)
    return MemoryConfig(
        strategy=memory_config.strategy,
        shard_spec=memory_config.shard_spec,
        nd_shard_spec=resolved_nd,
        buffer_type=memory_config.buffer_type,
        tensor_memory_layout=memory_config.tensor_memory_layout,
    )


class Tensor:
    """TTNN-like Tensor wrapper built on torch.Tensor.

    Exposes `.shape`, `.dtype`, and `.layout`.  The layout determines how
    indices are interpreted: TILE_LAYOUT uses tile-space indexing (each index
    unit = 32 elements); ROW_MAJOR_LAYOUT uses element-space indexing directly.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        layout: IndexType = TILE_LAYOUT,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
    ) -> None:
        if tensor.ndim < 1:
            raise ValueError(f"Tensor must have at least 1 dimension, got 0-d scalar")
        self._tensor: torch.Tensor = tensor
        self._layout: IndexType = layout
        self.memory_config: MemoryConfig = _maybe_resolve_nd_shard_spec_for_tensor(
            tuple(tensor.shape), memory_config
        )
        self.mesh_shard_info: Optional[MeshShardInfo] = None

    @property
    def shape(self) -> Shape:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def layout(self) -> IndexType:
        return self._layout

    @property
    def element_size(self) -> int:
        """Number of bytes per element for this tensor's dtype."""
        return self._tensor.element_size()

    def _validate_tile_alignment(self) -> None:
        """Validate that this tensor supports tile-style indexing.

        Must only be called for TILE_LAYOUT tensors.

        For 2-D+ tensors the last two dimensions must be tile-aligned (or
        degenerate); leading batch dimensions may have any size.
        For 1-D tensors the single dimension must be a multiple of
        TILE_SHAPE[0] (or exactly 1).

        Raises:
            ValueError: If the tensor has fewer than 1 dimension,
                or if the tile dimensions are not aligned.
        """
        ndim = len(self._tensor.shape)
        if ndim < 1:
            raise ValueError(
                f"Tile-style indexing requires at least 1 dimension, "
                f"got {ndim}D tensor"
            )
        if ndim == 1:
            dim_size = self._tensor.shape[0]
            if dim_size != 1 and dim_size % TILE_SHAPE[0] != 0:
                raise ValueError(
                    f"Tensor dimension 0 has size {dim_size} which is not "
                    f"a multiple of tile dimension {TILE_SHAPE[0]}"
                )
            return
        for i, (dim_size, tile_dim) in enumerate(
            zip(self._tensor.shape[-2:], TILE_SHAPE)
        ):
            if dim_size == 1:
                continue
            if dim_size % tile_dim != 0:
                raise ValueError(
                    f"Tensor dimension {ndim - 2 + i} has size {dim_size} which is not "
                    f"a multiple of tile dimension {tile_dim}"
                )

    @staticmethod
    def _normalize_index(selector: Selector) -> slice:
        """Convert an integer index to a unit slice, or return slice as-is."""
        return normalize_selector_to_slice(selector)

    @staticmethod
    def _validate_tile_slice(s: slice, dim_name: str) -> None:
        """Validate a tile-coordinate slice has explicit bounds and no step.

        Raises:
            ValueError: If start or stop is None, or step is set.
        """
        if s.start is None:
            raise ValueError(
                f"Tile slice '{dim_name}' must have explicit start value, "
                f"got slice({s.start}, {s.stop}, {s.step})"
            )
        if s.stop is None:
            raise ValueError(
                f"Tile slice '{dim_name}' must have explicit stop value, "
                f"got slice({s.start}, {s.stop}, {s.step})"
            )
        if s.step is not None:
            raise ValueError(
                f"Tile slice '{dim_name}' must not have a step value, "
                f"got slice({s.start}, {s.stop}, {s.step}). Only simple slices are supported."
            )

    def _to_element_key(self, key: Tuple[Selector, ...]) -> Tuple[Selector, ...]:
        """Translate a coordinate key to an element-space index tuple.

        All integer indices are first normalized to unit slices via
        _normalize_index so that no dimension is ever collapsed.

        For ROW_MAJOR_LAYOUT tensors no further scaling is applied.

        For TILE_LAYOUT tensors the last two (row, col) slices are multiplied
        by TILE_SHAPE to convert from tile-space to element-space.  Batch
        slices are left as-is (implicit tile size 1).

        Args:
            key: Tuple whose length must exactly match the tensor's rank.
                For a 1-D tensor: 1 element.  For an N-D tensor (N >= 2): N
                elements.

        Returns:
            Tuple suitable for indexing the underlying torch.Tensor directly.

        Raises:
            ValueError: If key length does not match tensor rank, the tensor
                is not tile-aligned (tiled only), or a tile slice has missing
                or stepped bounds.
        """
        ndim = len(self._tensor.shape)
        if len(key) != ndim:
            raise ValueError(
                f"Key length {len(key)} does not match tensor rank {ndim}: "
                f"expected exactly {ndim} element(s)"
            )

        normalized = tuple(self._normalize_index(k) for k in key)

        if self._layout == ROW_MAJOR_LAYOUT:
            # Element-space indexing: no tile scaling needed.
            return normalized

        self._validate_tile_alignment()
        if ndim == 1:
            self._validate_tile_slice(normalized[0], "col")
            return (
                slice(
                    normalized[0].start * TILE_SHAPE[0],
                    normalized[0].stop * TILE_SHAPE[0],
                ),
            )
        *batch_s, row_s, col_s = normalized
        self._validate_tile_slice(row_s, "row")
        self._validate_tile_slice(col_s, "col")
        return (
            *batch_s,
            slice(row_s.start * TILE_SHAPE[0], row_s.stop * TILE_SHAPE[0]),
            slice(col_s.start * TILE_SHAPE[1], col_s.stop * TILE_SHAPE[1]),
        )

    def element_slice_starts(self, key: TensorKey) -> Shape:
        """Element-space start offset per dimension for ``key`` (``slice.start`` values).

        Uses the same rules as :meth:`__getitem__`: tile indices for
        ``TILE_LAYOUT`` are converted to element bounds; ``ROW_MAJOR_LAYOUT`` keys
        are already element-space.
        """
        match key:
            case tuple():
                normalized: Tuple[Selector, ...] = key
            case _:
                normalized = (key,)
        ek = self._to_element_key(normalized)
        starts: list[int] = []
        for i, s in enumerate(ek):
            if not isinstance(s, slice) or s.start is None:
                raise ValueError(
                    f"element_slice_starts requires explicit slice bounds on dimension {i}, got {s!r}"
                )
            starts.append(s.start)
        return tuple(starts)

    def __getitem__(self, key: TensorKey) -> "Tensor":
        # Python passes a bare int/slice (not a tuple) for single-element indexing.
        match key:
            case tuple():
                normalized: Tuple[Selector, ...] = key
            case _:
                normalized = (key,)
        result = Tensor(
            self._tensor[cast(Any, self._to_element_key(normalized))], self._layout
        )
        if hasattr(self, "_name"):
            result._name = self._name  # type: ignore
        result.memory_config = self.memory_config
        # Accumulate the element-space origin so locality analysis can find the
        # position of this slice within the original (root) sharded tensor.
        # Open-ended slices (e.g. tensor[:]) have no computable start, so fall
        # back to the parent's origin (which is correct when selecting the full extent).
        parent_origin: Tuple[int, ...] = getattr(
            self, "_element_origin", (0,) * len(self.shape)
        )
        try:
            slice_origin = self.element_slice_starts(normalized)
            result._element_origin = tuple(p + s for p, s in zip(parent_origin, slice_origin))  # type: ignore[attr-defined]
        except ValueError:
            result._element_origin = parent_origin  # type: ignore[attr-defined]
        return result

    def __setitem__(self, key: TensorKey, value: "Tensor") -> None:
        normalized: Tuple[Selector, ...] = key if isinstance(key, tuple) else (key,)
        self._tensor[cast(Any, self._to_element_key(normalized))] = value._tensor

    def __repr__(self) -> str:
        # Delegate to torch for value and dtype formatting (handles truncation for large tensors).
        layout_str = (
            f", layout={self._layout.name}" if self._layout != TILE_LAYOUT else ""
        )
        return f"Tensor(shape={tuple(self._tensor.shape)}{layout_str}, data={repr(self._tensor)})"

    def to_torch(self) -> torch.Tensor:
        """Public accessor for the underlying torch tensor."""
        return self._tensor

    # ---- Binary operations (element-wise) ----

    def __add__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise addition."""
        match other:
            case Tensor():
                return Tensor(self._tensor + other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor + other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __sub__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise subtraction."""
        match other:
            case Tensor():
                return Tensor(self._tensor - other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor - other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __mul__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise multiplication."""
        match other:
            case Tensor():
                return Tensor(self._tensor * other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor * other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __truediv__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise true division."""
        match other:
            case Tensor():
                return Tensor(self._tensor / other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor / other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __floordiv__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise floor division."""
        match other:
            case Tensor():
                return Tensor(self._tensor // other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor // other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __mod__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise modulo."""
        match other:
            case Tensor():
                return Tensor(self._tensor % other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor % other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __pow__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise exponentiation."""
        match other:
            case Tensor():
                return Tensor(self._tensor**other._tensor, self._layout)
            case float() | int():
                return Tensor(self._tensor**other, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        match other:
            case Tensor():
                return Tensor(self._tensor @ other._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __neg__(self) -> "Tensor":
        """Unary negation."""
        return Tensor(-self._tensor, self._layout)

    def __abs__(self) -> "Tensor":
        """Absolute value."""
        return Tensor(torch.abs(self._tensor), self._layout)

    # ---- Reverse binary operations ----

    def __radd__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise addition."""
        match other:
            case float() | int():
                return Tensor(other + self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rsub__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise subtraction."""
        match other:
            case float() | int():
                return Tensor(other - self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rmul__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise multiplication."""
        match other:
            case float() | int():
                return Tensor(other * self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rtruediv__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise true division."""
        match other:
            case float() | int():
                return Tensor(other / self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rfloordiv__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise floor division."""
        match other:
            case float() | int():
                return Tensor(other // self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rmod__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise modulo."""
        match other:
            case float() | int():
                return Tensor(other % self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rpow__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise exponentiation."""
        match other:
            case float() | int():
                return Tensor(other**self._tensor, self._layout)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented


def rand(
    shape: Shape,
    dtype: torch.dtype = bfloat16,
    layout: IndexType = TILE_LAYOUT,
    device: object = None,
    memory_config: object = None,
) -> Tensor:
    """Create a random tensor with given shape, dtype, and layout."""
    t = torch.rand(shape, dtype=torch.float32)
    t = t.to(dtype)
    return Tensor(t, layout)


def empty(
    shape: Shape,
    dtype: torch.dtype = bfloat16,
    layout: IndexType = TILE_LAYOUT,
    device: object = None,
    memory_config: object = None,
) -> Tensor:
    """Create an uninitialized tensor with given shape, dtype, and layout."""
    t = torch.empty(shape, dtype=dtype)
    return Tensor(t, layout)


def to_torch(
    t: Union[Tensor, torch.Tensor],
    mesh_composer: Optional[ConcatMeshToTensor] = None,
) -> torch.Tensor:
    """Convert a simulator Tensor or torch.Tensor to torch.Tensor.

    Args:
        t: Tensor to convert.
        mesh_composer: Ignored in the simulator; accepted for API compatibility.

    Returns:
        Plain torch.Tensor.
    """
    match t:
        case Tensor() as tw:
            return tw.to_torch()
        case torch.Tensor() as tt:
            return tt
        case _:
            raise TypeError(f"Unsupported type for to_torch: {type(t)}")


def from_torch(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    layout: IndexType = TILE_LAYOUT,
    device: Optional[Union[Device, MeshDevice]] = None,
    memory_config: Optional[MemoryConfig] = None,
    mesh_mapper: Optional[TensorToMesh] = None,
    spec: Optional[TensorSpec] = None,
) -> Tensor:
    """Convert a torch.Tensor to a TTNN simulator Tensor.

    Args:
        tensor: Input torch tensor to wrap
        dtype: Optional dtype to convert to (defaults to tensor's dtype, or
            ``spec.dtype`` when ``spec`` is given)
        layout: Layout for the resulting Tensor (overridden by ``spec.layout``
            when ``spec`` is given)
        device: Device parameter (no-op in simulator)
        memory_config: MemoryConfig to attach (ignored when ``spec`` is given;
            used as-is when ``mesh_mapper`` is given alongside an explicit config).
        mesh_mapper: When a :class:`ShardTensorToMesh`, records the partition
            axis and device count in the tensor's :attr:`~Tensor.mesh_shard_info`
            attribute so that :func:`all_reduce` can determine the partition
            structure without consulting global state.  :class:`ReplicateTensorToMesh`
            is accepted for API compatibility but has no effect.
        spec: Optional :class:`TensorSpec` from ``TensorSpec(...).width_sharded`` /
            ``nd_sharded`` / etc.; when set, shape must match ``tensor`` and
            sharding metadata is applied.

    Returns:
        Tensor wrapping the input (potentially dtype-converted) torch tensor.
    """
    if spec is not None:
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(
                f"tensor shape {tuple(tensor.shape)} does not match spec.shape {spec.shape}"
            )
        layout = spec.layout
        eff_dtype = spec.dtype if dtype is None else dtype
        eff_mc = (
            spec.memory_config if spec.memory_config is not None else DRAM_MEMORY_CONFIG
        )
    elif isinstance(mesh_mapper, ShardTensorToMesh):
        eff_dtype = dtype
        eff_mc = memory_config if memory_config is not None else DRAM_MEMORY_CONFIG
    else:
        eff_dtype = dtype
        eff_mc = memory_config if memory_config is not None else DRAM_MEMORY_CONFIG

    if eff_dtype is not None and tensor.dtype != eff_dtype:
        tensor = tensor.to(eff_dtype)

    result = Tensor(tensor, layout, memory_config=eff_mc)
    if isinstance(mesh_mapper, ShardTensorToMesh):
        result.mesh_shard_info = MeshShardInfo(
            dim=mesh_mapper.dim % tensor.ndim,
            num_devices=mesh_mapper.mesh.num_devices,
        )
    return result


# Strategy-to-ShardingStrategy mapping for create_sharded_memory_config.
_SHARD_STRATEGY_MAP: dict[ShardStrategy, ShardingStrategy] = {
    ShardStrategy.HEIGHT: ShardingStrategy.HEIGHT_SHARDED,
    ShardStrategy.WIDTH: ShardingStrategy.WIDTH_SHARDED,
    ShardStrategy.BLOCK: ShardingStrategy.BLOCK_SHARDED,
}


def create_sharded_memory_config(
    shape: Union[Tuple[int, ...], List[int]],
    core_grid: CoreGrid,
    strategy: ShardStrategy,
    orientation: Optional[ShardOrientation] = None,
    use_height_and_width_as_shard_shape: bool = False,
) -> MemoryConfig:
    """Create a MemoryConfig for a sharded tensor.

    Mirrors ttnn.create_sharded_memory_config.  The simulator does not execute
    sharding mechanics, but stores the resulting MemoryConfig on tensors so that
    statistics collection can classify local vs. remote L1 accesses.

    Args:
        shape: Tensor element shape.  When use_height_and_width_as_shard_shape
            is False this is the full tensor shape; when True, only the last
            two dimensions are used and they specify the shard dimensions.
        core_grid: 2-D core grid describing the cores to shard across.
        strategy: Sharding strategy (HEIGHT, WIDTH, or BLOCK).
        orientation: Core traversal order (default ROW_MAJOR).
        use_height_and_width_as_shard_shape: When True, shape[-2] and shape[-1]
            are the shard height and width in elements.  When False (default),
            the shard dimensions are derived from shape and core_grid.

    Returns:
        MemoryConfig with ShardSpec computed from the arguments.
    """
    shape_t = tuple(shape)
    shard_orient = (
        orientation if orientation is not None else ShardOrientation.ROW_MAJOR
    )

    if use_height_and_width_as_shard_shape:
        shard_h, shard_w = shape_t[-2], shape_t[-1]
    else:
        total_h = math.prod(shape_t[:-1])
        total_w = shape_t[-1]
        match strategy:
            case ShardStrategy.HEIGHT:
                shard_h = total_h // core_grid.num_cores
                shard_w = total_w
            case ShardStrategy.WIDTH:
                shard_h = total_h
                shard_w = total_w // core_grid.num_cores
            case ShardStrategy.BLOCK:
                shard_h = total_h // core_grid.y
                shard_w = total_w // core_grid.x

    match strategy:
        case ShardStrategy.HEIGHT | ShardStrategy.WIDTH:
            shard_grid: Shape = (core_grid.num_cores,)
        case ShardStrategy.BLOCK:
            shard_grid = (core_grid.y, core_grid.x)

    sharding_strategy = _SHARD_STRATEGY_MAP[strategy]
    spec = ShardSpec(
        shard_grid=shard_grid,
        shard_shape=(shard_h, shard_w),
        orientation=shard_orient,
    )
    return MemoryConfig(strategy=sharding_strategy, shard_spec=spec)


def is_sharded(tensor: Tensor) -> bool:
    """Return True if the tensor's memory config describes a sharded layout.

    Mirrors ttnn.is_sharded.
    """
    return tensor.memory_config.strategy not in (ShardingStrategy.INTERLEAVED,)


def get_memory_config(tensor: Tensor) -> MemoryConfig:
    """Return the MemoryConfig attached to a tensor.

    Mirrors ttnn.get_memory_config.
    """
    return tensor.memory_config


def to_memory_config(tensor: Tensor, memory_config: MemoryConfig) -> Tensor:
    """Return a view of tensor with memory_config replaced.

    Mirrors ttnn.to_memory_config.  The simulator does not move data between
    memory banks; it only updates the MemoryConfig metadata so that subsequent
    statistics collection uses the new layout.
    """
    result = Tensor(tensor.to_torch(), tensor.layout, memory_config)
    if hasattr(tensor, "_name"):
        result._name = tensor._name  # type: ignore[attr-defined]
    return result


def multiply(
    a: Union[Tensor, torch.Tensor],
    b: Union[Tensor, torch.Tensor],
) -> Tensor:
    """Element-wise multiply (simulator shim for ttnn.multiply)."""
    a_t = to_torch(a) if isinstance(a, Tensor) else a
    b_t = to_torch(b) if isinstance(b, Tensor) else b
    return Tensor(a_t * b_t)


def split_work_to_cores(
    core_grid: Union[CoreCoord, CoreRangeSet],
    units_to_divide: int,
    row_wise: bool = False,
) -> Tuple[int, CoreRangeSet, CoreRangeSet, CoreRangeSet, int, int]:
    """Split work units across cores in a grid or CoreRangeSet.

    This function divides a specified number of work units across cores. It returns
    information about how the work is distributed, including core ranges for different
    groups if work cannot be evenly divided.

    Args:
        core_grid: Either a CoreCoord (grid size) or CoreRangeSet to distribute work across
        units_to_divide: The total number of work units to distribute
        row_wise: Whether to distribute work by iterating row-wise. Defaults to False (column-wise)

    Returns:
        tuple: A tuple containing:
            - num_cores (int): Number of cores being used
            - all_cores (CoreRangeSet): All cores involved
            - core_group_1 (CoreRangeSet): Cores doing more work
            - core_group_2 (CoreRangeSet): Cores doing less work (empty if evenly divisible)
            - units_per_core_group_1 (int): Work units per core in group 1
            - units_per_core_group_2 (int): Work units per core in group 2

    Example:
        >>> # Split 100 tiles across an 8x8 core grid
        >>> num_cores, all_cores, core_group_1, core_group_2, units_1, units_2 = \\
        ...     ttnn.split_work_to_cores(ttnn.CoreCoord(8, 8), 100)
        >>> print(f"Using {num_cores} cores, {units_1} units per core in group 1, {units_2} in group 2")
    """
    # Determine the total number of cores and create the all_cores CoreRangeSet
    match core_grid:
        case CoreCoord():
            # Create a CoreRangeSet from the grid dimensions
            num_cores = core_grid.x * core_grid.y
            all_cores = CoreRangeSet(
                [
                    CoreRange(
                        CoreCoord(0, 0), CoreCoord(core_grid.x - 1, core_grid.y - 1)
                    )
                ]
            )
            grid_size = (core_grid.x, core_grid.y)
        case _:
            # CoreRangeSet case
            num_cores = core_grid.num_cores()
            all_cores = core_grid
            # For CoreRangeSet, we'll need to determine the bounding grid size
            # This is a simplification - in practice we'd need to track the actual ranges
            grid_size = None

    # Calculate work distribution
    # Limit number of cores to units_to_divide if there are more cores than work
    num_cores_used = min(num_cores, units_to_divide)

    if num_cores_used == 0 or units_to_divide == 0:
        # No work to distribute
        empty_range_set = CoreRangeSet([])
        return 0, empty_range_set, empty_range_set, empty_range_set, 0, 0

    # Calculate units per core for each group
    units_per_core_base = units_to_divide // num_cores_used  # Floor division
    remainder = units_to_divide % num_cores_used

    # Group 1 gets one extra unit if there's a remainder
    if remainder > 0:
        units_per_core_group_1 = units_per_core_base + 1
        units_per_core_group_2 = units_per_core_base
        num_cores_group_1 = remainder
        num_cores_group_2 = num_cores_used - remainder
    else:
        # Evenly divisible - all cores in group 1
        units_per_core_group_1 = units_per_core_base
        units_per_core_group_2 = 0
        num_cores_group_1 = num_cores_used
        num_cores_group_2 = 0

    # Create core groups based on work distribution
    if num_cores_group_2 == 0:
        # All cores get the same amount of work (evenly divisible)
        match core_grid:
            case CoreCoord() if grid_size:
                # Generate core list for the used cores
                cores_list: List[CoreCoord] = []
                if row_wise:
                    for y in range(grid_size[1]):
                        for x in range(grid_size[0]):
                            if len(cores_list) < num_cores_used:
                                cores_list.append(CoreCoord(x, y))
                else:
                    for x in range(grid_size[0]):
                        for y in range(grid_size[1]):
                            if len(cores_list) < num_cores_used:
                                cores_list.append(CoreCoord(x, y))

                core_group_1 = CoreRangeSet([CoreRange(c, c) for c in cores_list])
            case _:
                # For CoreRangeSet, extract the first num_cores_used cores
                ranges = all_cores.ranges()
                cores_list: List[CoreCoord] = []
                for r in ranges:
                    for y in range(r.start.y, r.end.y + 1):
                        for x in range(r.start.x, r.end.x + 1):
                            if len(cores_list) < num_cores_used:
                                cores_list.append(CoreCoord(x, y))

                core_group_1 = CoreRangeSet([CoreRange(c, c) for c in cores_list])

        core_group_2 = CoreRangeSet([])  # Empty
    else:
        # Split cores into two groups
        match core_grid:
            case CoreCoord() if grid_size:
                # Generate core ranges for the two groups
                cores_list: List[CoreCoord] = []
                if row_wise:
                    # Row-wise iteration: iterate rows first
                    for y in range(grid_size[1]):
                        for x in range(grid_size[0]):
                            cores_list.append(CoreCoord(x, y))
                else:
                    # Column-wise iteration: iterate columns first
                    for x in range(grid_size[0]):
                        for y in range(grid_size[1]):
                            cores_list.append(CoreCoord(x, y))

                # Split into groups
                group_1_cores: List[CoreCoord] = cores_list[:num_cores_group_1]
                group_2_cores: List[CoreCoord] = cores_list[
                    num_cores_group_1:num_cores_used
                ]

                # Convert to CoreRangeSets (simplified: one range per core)
                if group_1_cores:
                    core_group_1 = CoreRangeSet(
                        [CoreRange(c, c) for c in group_1_cores]
                    )
                else:
                    core_group_1 = CoreRangeSet([])

                if group_2_cores:
                    core_group_2 = CoreRangeSet(
                        [CoreRange(c, c) for c in group_2_cores]
                    )
                else:
                    core_group_2 = CoreRangeSet([])
            case _:
                # For CoreRangeSet input, create a simplified distribution
                # This is a basic implementation - a more sophisticated version would
                # iterate through the actual ranges in the CoreRangeSet
                ranges = all_cores.ranges()
                all_cores_list: List[CoreCoord] = []
                for r in ranges:
                    for y in range(r.start.y, r.end.y + 1):
                        for x in range(r.start.x, r.end.x + 1):
                            all_cores_list.append(CoreCoord(x, y))

                group_1_cores: List[CoreCoord] = all_cores_list[:num_cores_group_1]
                group_2_cores: List[CoreCoord] = all_cores_list[
                    num_cores_group_1:num_cores_used
                ]

                if group_1_cores:
                    core_group_1 = CoreRangeSet(
                        [CoreRange(c, c) for c in group_1_cores]
                    )
                else:
                    core_group_1 = CoreRangeSet([])

                if group_2_cores:
                    core_group_2 = CoreRangeSet(
                        [CoreRange(c, c) for c in group_2_cores]
                    )
                else:
                    core_group_2 = CoreRangeSet([])

    return (
        num_cores_used,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
    )


def all_reduce(
    input_tensor: Tensor,
    cluster_axis: Optional[int] = None,
    mesh_device: Optional[Any] = None,
    memory_config: Optional[MemoryConfig] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> Tensor:
    """Sum-reduce across all simulated devices.

    The partition structure is read from the tensor's :attr:`~Tensor.mesh_shard_info`
    attribute, which is set by :func:`from_torch` when a :class:`ShardTensorToMesh`
    mapper is provided.  This attribute records the partition axis (``dim``) and
    device count directly, keeping inter-device distribution separate from
    intra-device sharding strategies stored in :class:`MemoryConfig`.

    The correct output for the all-reduce collective is: sum each group of
    corresponding slices element-wise across all partitions, then give every
    partition that same sum.

    Args:
        input_tensor: Input tensor (must have been created with ShardTensorToMesh).
        cluster_axis: Ignored (accepted for API compatibility).
        mesh_device: Ignored (accepted for API compatibility).
        memory_config: Optional output memory config.
        dtype: Optional output dtype.
        **kwargs: Additional keyword arguments accepted for API compatibility.

    Returns:
        Tensor where every partition contains the element-wise sum of all
        partitions.
    """
    msi = input_tensor.mesh_shard_info
    if msi is None:
        raise ValueError("Mesh device is required for all_reduce operation")

    t = input_tensor.to_torch()
    d = msi.dim % t.ndim
    n = msi.num_devices
    shard = t.shape[d] // n

    if t.shape[d] != n * shard:
        return input_tensor

    # Sum corresponding slices across all n partitions.
    reduced = sum(t.narrow(d, i * shard, shard) for i in range(n))
    # Every partition gets the same reduced result.
    result = torch.cat([reduced] * n, dim=d).contiguous()  # type: ignore[arg-type]

    if dtype is not None and result.dtype != dtype:
        result = result.to(dtype)

    out_memory_config = (
        memory_config if memory_config is not None else input_tensor.memory_config
    )
    result_tensor = Tensor(result, input_tensor.layout, out_memory_config)
    result_tensor.mesh_shard_info = msi
    if hasattr(input_tensor, "_name"):
        result_tensor._name = input_tensor._name  # type: ignore[attr-defined]
    return result_tensor


def all_gather(
    input_tensor: Tensor,
    dim: int,
    cluster_axis: Optional[int] = None,
    mesh_device: Optional[Any] = None,
    memory_config: Optional[MemoryConfig] = None,
    **kwargs: Any,
) -> Tensor:
    """Gather shards from all simulated devices along ``dim``.

    The partition structure is read from the tensor's :attr:`~Tensor.mesh_shard_info`
    attribute, which is set by :func:`from_torch` when a :class:`ShardTensorToMesh`
    mapper is provided.

    Each device contributes its local shard (partitioned along ``msi.dim``).
    After the gather every device holds an identical result: all shards
    concatenated along ``dim``.  The simulator represents n identical copies
    by stacking them along ``msi.dim``, matching the output of
    ``ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor(mesh, msi.dim))``.

    Args:
        input_tensor: Input tensor (must have been created with ShardTensorToMesh).
        dim: Dimension along which to concatenate the gathered shards.
        cluster_axis: Ignored (accepted for API compatibility).
        mesh_device: Ignored (accepted for API compatibility).
        memory_config: Optional output memory config.
        **kwargs: Additional keyword arguments accepted for API compatibility.

    Returns:
        Tensor where every partition contains all shards concatenated along
        ``dim``.  Output shape equals input shape on all dimensions except
        ``dim``, which grows by a factor of ``num_devices``.
    """
    msi = input_tensor.mesh_shard_info
    if msi is None:
        raise ValueError("Mesh device is required for all_gather operation")

    t = input_tensor.to_torch()
    shard_dim = msi.dim % t.ndim
    n = msi.num_devices
    shard_size = t.shape[shard_dim] // n

    if t.shape[shard_dim] != n * shard_size:
        return input_tensor

    gather_dim = dim % t.ndim

    # Each device's shard, sliced along shard_dim.
    shards = [t.narrow(shard_dim, i * shard_size, shard_size) for i in range(n)]
    # All n devices get the same result: every shard concatenated along gather_dim.
    gathered = torch.cat(shards, dim=gather_dim)
    # Stack n identical copies along shard_dim to match the simulator's
    # multi-device representation (same as ConcatMeshToTensor would produce).
    result = torch.cat([gathered] * n, dim=shard_dim).contiguous()

    out_memory_config = (
        memory_config if memory_config is not None else input_tensor.memory_config
    )
    result_tensor = Tensor(result, input_tensor.layout, out_memory_config)
    result_tensor.mesh_shard_info = MeshShardInfo(dim=msi.dim, num_devices=n)
    if hasattr(input_tensor, "_name"):
        result_tensor._name = input_tensor._name  # type: ignore[attr-defined]
    return result_tensor


def synchronize_device(*args: Any, **kwargs: Any) -> None:
    """No-op stub for ttnn.synchronize_device().

    On real hardware this blocks the host until all pending device operations
    have completed.  The simulator executes kernels synchronously, so there is
    nothing to wait for.
    """


def squeeze(input_tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1 from a tensor.

    Args:
        input_tensor: Input tensor
        dim: If specified, only squeeze this dimension if it has size 1.
             If None, squeeze all dimensions of size 1.

    Returns:
        Tensor with singleton dimensions removed
    """
    torch_tensor = input_tensor.to_torch()
    if dim is None:
        result = torch_tensor.squeeze()
    else:
        result = torch_tensor.squeeze(dim)
    return Tensor(result)


# Dynamically generate wrapper functions for all ttnn operations with golden functions
def _create_golden_wrapper(
    operation_name: str, golden_fn: Callable[..., Any]
) -> Callable[..., Any]:
    """Create a wrapper function that calls the golden function and wraps result in Tensor.

    Args:
        operation_name: Name of the operation (for documentation)
        golden_fn: The golden function to wrap

    Returns:
        Wrapper function that converts inputs/outputs appropriately
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert Tensor arguments to torch.Tensor
        def convert_arg(arg: Any) -> Any:
            match arg:
                case Tensor():
                    return arg.to_torch()
                case _:
                    return arg

        torch_args = tuple(convert_arg(arg) for arg in args)
        torch_kwargs = {k: convert_arg(v) for k, v in kwargs.items()}

        # Call golden function
        result = golden_fn(*torch_args, **torch_kwargs)

        # Wrap result in Tensor if it's a torch.Tensor
        match result:
            case torch.Tensor():
                return Tensor(result)
            case _:
                return result

    # Set proper function name and docstring
    wrapper.__name__ = operation_name
    wrapper.__doc__ = (
        f"Wrapper for ttnn.{operation_name} using golden function implementation."
    )

    return wrapper


# Functions that should NOT be auto-wrapped (already implemented or would break things)
_EXCLUDE_FROM_WRAPPING = {
    # Names here are skipped in addition to any symbol already in this module's
    # globals() (those are never overwritten by the golden-function loop).
    # Core infrastructure functions that are already implemented
    "from_torch",
    "to_torch",
    "from_device",
    "to_device",
    "to_dtype",
    "to_layout",
    "to_memory_config",
    # Tensor creation functions that are already implemented
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "arange",
    # Built-in functions that shouldn't be wrapped
    "min",
    "max",
    "sum",
    # Functions that return non-tensor types
    "clone",
    "reshape",
    "permute",
    "concat",
    "pad",
    "squeeze",
    # Sharding/memory functions
    "interleaved_to_sharded",
    "interleaved_to_sharded_partial",
    "sharded_to_interleaved",
    "sharded_to_interleaved_partial",
    "reallocate",
    "reshard",
    "tilize",
    "bitcast",
    "typecast",
}

# Get all operations with golden functions and create wrappers at module load time
if TTNN_AVAILABLE:
    import ttnn  # type: ignore[reportMissingImports]  # Re-import for type checker to know ttnn is bound in this block

    _operations_to_wrap = [name for name in dir(ttnn) if not name.startswith("_")]

    for _op_name in _operations_to_wrap:
        # Skip if already in our namespace or in exclude list
        if _op_name in globals() or _op_name in _EXCLUDE_FROM_WRAPPING:
            continue

        _op = getattr(ttnn, _op_name)

        # Skip non-callable attributes (classes, constants, etc.)
        if not callable(_op):
            continue

        try:
            _golden_fn = ttnn.get_golden_function(_op)  # type: ignore[union-attr]
            # Create wrapper and add to module globals
            globals()[_op_name] = _create_golden_wrapper(
                _op_name, _golden_fn  # type: ignore[arg-type]
            )
        except (RuntimeError, AttributeError):
            # RuntimeError: Operation doesn't have a golden function
            # AttributeError: Object doesn't have golden_function attribute (e.g., enums, classes)
            # Both are expected for many ttnn attributes - skip them
            continue
        # Let other exceptions propagate - they indicate real bugs

    # Clean up temporary variables
    _cleanup_name: Optional[str] = None
    for _cleanup_name in ("_operations_to_wrap", "_op_name", "_op", "_golden_fn"):
        if _cleanup_name in globals():
            del globals()[_cleanup_name]
    if _cleanup_name is not None:
        del _cleanup_name
