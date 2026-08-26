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
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    SupportsIndex,
    Tuple,
    Union,
    cast,
    overload,
)

import torch

# Try to import actual ttnn, track if availability
TTNN_AVAILABLE: bool
try:
    import ttnn  # type: ignore[reportMissingImports]

    TTNN_AVAILABLE = True  # type: ignore[reportConstantRedefinition]
except ImportError:
    TTNN_AVAILABLE = False  # type: ignore[reportConstantRedefinition]

from .constants import FACE_SHAPE, TILE_SHAPE
from .trace import TRACE
from .typedefs import Count, Index, IndexType, Selector, Size, TensorKey

# ``ttl.Shape``, the specification's tuple of dimensions, under an alias: the
# bare name belongs to ttnn's ``Shape`` class in this module.  Annotates what
# this module hands to the DSL rather than to a ttnn caller.
from .typedefs import Shape as TtlShape

# Number of shards along each tensor dimension for a sharded tensor;
# math.prod(ShardGrid) equals the number of participating cores. This is a
# distinct concept from a tensor Shape and from a physical CoreGrid, and ttnn
# has no public named type for it, so the simulator names it here.
ShardGrid = Tuple[Size, ...]

# Public constants (mirror TTL constants)
TILE_SIZE: int = TILE_SHAPE[0]
TILE_LAYOUT = IndexType.TILE
ROW_MAJOR_LAYOUT = IndexType.ROW_MAJOR


def _is_dry_run() -> bool:
    """Return True when the simulator is in dry-run mode.

    Uses a lazy import to avoid a circular dependency
    (context_types imports Tensor from this module).
    """
    from .context import get_context  # noqa: PLC0415 (lazy import intentional)

    return get_context().config.dry_run


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
        shard_grid: Optional[ShardGrid] = None,
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
    def shard_grid(self) -> ShardGrid:
        if self._shard_grid is None:
            raise ValueError(
                "ShardSpec uses a CoreRangeSet grid; build MemoryConfig(TensorMemoryLayout, BufferType, spec) to resolve shard_grid"
            )
        return self._shard_grid

    @property
    def shape(self) -> List[int]:
        """Per-shard extent, under the name ttnn reports it.

        ttnn takes this as ``shard_shape=`` and reports it as ``shape``, as a
        two-element list (its ``std::array<uint32_t, 2>``); both names read the
        same value here.
        """
        return list(self.shard_shape)

    def num_cores(self) -> int:
        """Cores the shards are laid out over, as ttnn reports it.

        ttnn reads it off the core grid; where the simulator was given shard
        counts instead, their product is the same number.
        """
        if self.grid is not None:
            return self.grid.num_cores()
        return math.prod(self.shard_grid)

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
            sg: ShardGrid = (cg.num_cores,)
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
      **element** units.  Taken in any spelling and reported as a
      :class:`Shape`, which is what ttnn holds it as.
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


    ``round_robin_cores`` applies only to ROUND_ROBIN (modulus for shard
    assignment).  It is spelled that way rather than ``num_cores`` because ttnn
    has a ``num_cores()`` *method*, which :meth:`num_cores` is, and the two mean
    different things: how many cores the shards go round, against how many the
    spec covers.
    """

    shard_shape: Sequence[int]
    core_ranges: Optional["CoreRangeSet"] = None
    shard_grid: Optional[ShardGrid] = None
    distribution: ShardDistributionStrategy = ShardDistributionStrategy.ROUND_ROBIN_1D
    round_robin_cores: Optional[int] = None

    def __post_init__(self) -> None:
        # Accept list inputs like the tech report (``shard_shape=[...]``) and
        # report a Shape, which is what ttnn's NdShardSpec holds.
        object.__setattr__(self, "shard_shape", Shape(self.shard_shape))
        if self.shard_grid is not None:
            object.__setattr__(self, "shard_grid", tuple(self.shard_grid))

    @property
    def grid(self) -> Optional["CoreRangeSet"]:
        """The participating cores, under the name ttnn reports them."""
        return self.core_ranges

    @property
    def shard_distribution_strategy(self) -> ShardDistributionStrategy:
        """:attr:`distribution`, under the name ttnn reports it."""
        return self.distribution

    def num_cores(self) -> int:
        """Cores this spec covers, as ttnn's ``num_cores()`` reports it.

        ttnn reads it off the core grid; where the simulator was given shard
        counts instead, their product is the same number.
        """
        if self.core_ranges is not None:
            return self.core_ranges.num_cores()
        if self.round_robin_cores is not None:
            return self.round_robin_cores
        if self.shard_grid is not None:
            return math.prod(self.shard_grid)
        raise ValueError(
            "NdShardSpec has neither core_ranges, round_robin_cores nor "
            "shard_grid, so it covers no known number of cores"
        )

    def with_resolved_shard_grid(self, tensor_shape: Sequence[int]) -> NdShardSpec:
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

    tt-metal style (see tensor sharding tech report)::

        MemoryConfig(
            TensorMemoryLayout.HEIGHT_SHARDED,
            BufferType.L1,
            ShardSpec(...),
        )
        MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)
        MemoryConfig(TensorMemoryLayout.INTERLEAVED)

    A layout given where the strategy goes names the same thing and is
    converted, so :attr:`strategy` is a :class:`ShardingStrategy` however the
    config was spelled.
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
        if len(args) == 3:
            if not isinstance(args[0], TensorMemoryLayout) or not isinstance(
                args[1], BufferType
            ):
                raise TypeError(
                    f"three positional arguments are ttnn's "
                    f"(TensorMemoryLayout, BufferType, ShardSpec|NdShardSpec), got "
                    f"({type(args[0]).__name__}, {type(args[1]).__name__}, "
                    f"{type(args[2]).__name__})"
                )
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

        if len(args) == 2:
            # ttnn's own two-argument form, e.g. its L1_MEMORY_CONFIG:
            # MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1).
            if not isinstance(args[0], TensorMemoryLayout) or not isinstance(
                args[1], BufferType
            ):
                raise TypeError(
                    f"two positional arguments are ttnn's "
                    f"(TensorMemoryLayout, BufferType), got "
                    f"({type(args[0]).__name__}, {type(args[1]).__name__})"
                )
            args, buffer_type = args[:1], args[1]

        if len(args) > 1:
            raise TypeError(
                f"a memory config takes at most three positional arguments "
                f"(TensorMemoryLayout, BufferType, ShardSpec|NdShardSpec), got "
                f"{len(args)}"
            )
        if args and not isinstance(args[0], (ShardingStrategy, TensorMemoryLayout)):
            # Positionally, the first argument is the layout (ttnn's spelling) or
            # the strategy that stands for it (the simulator's). Anything else is
            # an argument in the wrong slot -- a buffer type, a shard spec -- and
            # defaulting past it would build a config the caller did not ask for
            # and hand back no sign of it.
            raise TypeError(
                f"the first positional argument is a TensorMemoryLayout or a "
                f"ShardingStrategy, got {type(args[0]).__name__}; pass a buffer type "
                f"or a shard spec by name (buffer_type=..., shard_spec=...)"
            )

        st = strategy if strategy is not None else (args[0] if len(args) == 1 else None)
        if st is None:
            # ttnn's arguments all have defaults and its default config is an
            # interleaved one, which is also what a tensor gets when no config
            # is named.
            st = ShardingStrategy.INTERLEAVED
        layout_tt = tensor_memory_layout
        if isinstance(st, TensorMemoryLayout):
            # ttnn's first argument is the memory layout and it has no separate
            # notion of a strategy, so a config spelled ttnn's way -- as its
            # documentation spells the interleaved one -- arrives with a layout
            # where the strategy goes.  They name the same thing; record both,
            # or every strategy comparison silently fails to match.
            layout_tt = st
            st = _tensor_memory_layout_to_sharding_strategy(st)
        self.strategy = st
        self.shard_spec = shard_spec
        self.nd_shard_spec = nd_shard_spec
        self.buffer_type = buffer_type
        # Both names for one thing are filled in, whichever the caller spelled, so
        # that two configs describing the same memory are equal: leaving the
        # unspelled one at None would make equality depend on the spelling, and
        # ttnn's own two-argument form and the simulator's strategy form would
        # compare unequal.
        self.tensor_memory_layout = (
            layout_tt
            if layout_tt is not None
            else _sharding_strategy_to_tensor_memory_layout(st)
        )

    @property
    def memory_layout(self) -> TensorMemoryLayout:
        """How the tensor is laid out over memory, under ttnn's name for it.

        Always answers, including for a config built the simulator's way with a
        strategy: the two say the same thing, and both are recorded at
        construction.
        """
        return self.tensor_memory_layout

    def is_sharded(self) -> bool:
        """Whether the tensor is sharded rather than interleaved, as ttnn asks."""
        return self.strategy != ShardingStrategy.INTERLEAVED

    @property
    def interleaved(self) -> bool:
        """The complement of :meth:`is_sharded`, as ttnn reports it."""
        return not self.is_sharded()

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

    def __hash__(self) -> int:
        """Hashed by the memory it names, leaving the shard spec to equality.

        Defining ``__eq__`` alone would make a config unhashable, and a frozen
        :class:`TensorSpec` hashes its fields -- so every spec would be unhashable
        too, now that each one carries a config. ttnn's is hashable, and a spec is
        a natural cache key.

        A shard spec is compared but not hashed: two configs that are equal agree
        on everything hashed here, which is all a hash has to promise.
        """
        return hash((self.strategy, self.buffer_type, self.tensor_memory_layout))

    def __repr__(self) -> str:
        return (
            f"MemoryConfig(strategy={self.strategy!r}, shard_spec={self.shard_spec!r}, "
            f"nd_shard_spec={self.nd_shard_spec!r}, buffer_type={self.buffer_type!r}, "
            f"tensor_memory_layout={self.tensor_memory_layout!r})"
        )


_LAYOUT_TO_STRATEGY: Dict[TensorMemoryLayout, ShardingStrategy] = {
    TensorMemoryLayout.INTERLEAVED: ShardingStrategy.INTERLEAVED,
    TensorMemoryLayout.HEIGHT_SHARDED: ShardingStrategy.HEIGHT_SHARDED,
    TensorMemoryLayout.WIDTH_SHARDED: ShardingStrategy.WIDTH_SHARDED,
    TensorMemoryLayout.BLOCK_SHARDED: ShardingStrategy.BLOCK_SHARDED,
    TensorMemoryLayout.ND_SHARDED: ShardingStrategy.ND_SHARDED,
}


def _tensor_memory_layout_to_sharding_strategy(
    layout: TensorMemoryLayout,
) -> ShardingStrategy:
    return _LAYOUT_TO_STRATEGY[layout]


def _sharding_strategy_to_tensor_memory_layout(
    strategy: ShardingStrategy,
) -> TensorMemoryLayout:
    for layout, mapped in _LAYOUT_TO_STRATEGY.items():
        if mapped == strategy:
            return layout
    raise ValueError(f"no memory layout stands for {strategy!r}")


@dataclass(kw_only=True)
class CoreGrid:
    """2-D core grid.  Mirrors ttnn.CoreGrid.

    Named arguments only, as ttnn's constructor is: it takes ``x`` first where
    this takes ``y`` first, so a positional pair would mean one grid here and
    its transpose on a device.

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
    left_shape: Sequence[int],
    right_shape: Sequence[int],
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


DRAM_MEMORY_CONFIG: MemoryConfig = MemoryConfig(
    TensorMemoryLayout.INTERLEAVED, BufferType.DRAM
)
L1_MEMORY_CONFIG: MemoryConfig = MemoryConfig(
    TensorMemoryLayout.INTERLEAVED, BufferType.L1
)

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

    def grid_size(self) -> CoreCoord:
        """Extent of this range along each axis, as ttnn reports it."""
        return CoreCoord(self.end.x - self.start.x + 1, self.end.y - self.start.y + 1)

    def contains(self, other: Union[CoreCoord, "CoreRange"]) -> bool:
        """Whether a core, or every core of another range, lies in this one."""
        match other:
            case CoreCoord():
                return (
                    self.start.x <= other.x <= self.end.x
                    and self.start.y <= other.y <= self.end.y
                )
            case CoreRange():
                return self.contains(other.start) and self.contains(other.end)


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

    def size(self) -> Count:
        """Number of ranges, as ttnn reports it (not the number of cores)."""
        return len(self._ranges)

    def bounding_box(self) -> CoreRange:
        """Smallest range covering every range in the set.

        tt-lang's own runtime asks a core range set for this when it turns a
        grid into kernel arguments, so a set the simulator produced has to
        answer it.
        """
        if not self._ranges:
            raise ValueError("an empty CoreRangeSet has no bounding box")
        return CoreRange(
            CoreCoord(
                min(r.start.x for r in self._ranges),
                min(r.start.y for r in self._ranges),
            ),
            CoreCoord(
                max(r.end.x for r in self._ranges),
                max(r.end.y for r in self._ranges),
            ),
        )

    def contains(self, core: CoreCoord) -> bool:
        """Whether any range in the set holds ``core``."""
        return any(r.contains(core) for r in self._ranges)

    def empty(self) -> bool:
        """Whether the set holds no ranges."""
        return not self._ranges

    def __repr__(self) -> str:
        return f"CoreRangeSet({self._ranges!r})"

    def __eq__(self, other: object) -> bool:
        match other:
            case CoreRangeSet():
                return self._ranges == other._ranges
            case _:
                return False

    def __hash__(self) -> int:
        # ttnn's is hashable, and a memory config holding one is compared and
        # cached by value.
        return hash(tuple(self._ranges))


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
    if core_ranges.empty():
        raise ValueError("CoreRangeSet is empty")
    extent = core_ranges.bounding_box().grid_size()
    return CoreGrid(y=extent.y, x=extent.x)


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
    shape: Sequence[int],
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


# Stands in for "no config was named", so that a spec can fill in the one its
# own layout and buffer describe.  Never reaches a caller: every spec replaces
# it in __post_init__.
_DERIVE_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.INTERLEAVED)


@dataclass(frozen=True)
class TensorSpec:
    """Tensor shape/dtype/layout/buffer metadata with optional sharding (ttnn API).

    Use ``height_sharded`` / ``width_sharded`` / ``block_sharded`` /
    ``sharded_across_dims`` / ``nd_sharded`` to attach a :class:`MemoryConfig`,
    then pass the spec to :func:`from_torch` (see tt-metal tensor sharding examples).

    ``shape`` is taken in any spelling and reported as a :class:`Shape`, as
    ttnn's ``TensorSpec.shape`` is.
    """

    shape: Sequence[int]
    dtype: torch.dtype = torch.float32
    layout: IndexType = TILE_LAYOUT
    buffer_type: BufferType = BufferType.DRAM
    memory_layout: TensorMemoryLayout = TensorMemoryLayout.INTERLEAVED
    # The factory hands back the one sentinel, which __post_init__ recognises;
    # a dataclass will not take an unhashable default any other way.
    memory_config: MemoryConfig = field(default_factory=lambda: _DERIVE_MEMORY_CONFIG)
    core_ranges: Optional[CoreRangeSet] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", Shape(self.shape))
        # An unsharded spec still describes memory, and ttnn's TensorSpec always
        # answers with a config, so build the one this spec's layout and buffer
        # name rather than leaving a None for the caller to reach through.
        if self.memory_config is _DERIVE_MEMORY_CONFIG:
            object.__setattr__(
                self,
                "memory_config",
                MemoryConfig(self.memory_layout, self.buffer_type),
            )

    @property
    def tile(self) -> Tile:
        """The tile the stored data is cut into, as ttnn's spec reports it."""
        return Tile()

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
        shard_shape: Sequence[int],
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
        nd = NdShardSpec(shard_shape=shard_shape, core_ranges=core_ranges)
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=nd)
        return replace(
            self,
            memory_layout=TensorMemoryLayout.ND_SHARDED,
            memory_config=mc,
            core_ranges=core_ranges,
        )


# Save the native dtypes before any rebinding.  These are the "declared"
# hardware dtypes — they carry the correct element_size (e.g. 2 for bfloat16)
# and are stored in Tensor._dtype for hardware-accurate L1 accounting.
# Maps torch attribute name -> original (pre-rebinding) dtype for every
# narrow floating-point type that has a native torch representation and should
# be promoted to float32 by default.  To add a new promotable type, add it
# here; the rebinding loop, set_disable_float32_promotion, and promote_dtype
# all derive from this single definition.
#
# bfloat8_b is not in this dict because PyTorch has no native bfloat8_b dtype
# and no torch.bfloat8_b attribute to rebind.  promote_dtype handles it
# directly by mapping it to bfloat16 and then applying the normal logic.
_PROMOTABLE_FLOAT_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Rebind every promotable dtype to float32 so that all native PyTorch code
# (e.g. torch.randn(dtype=torch.bfloat16)) transparently uses float32.
# This keeps host operations fast on platforms like Apple Silicon that lack
# hardware bfloat16/float16 support.
for _attr in _PROMOTABLE_FLOAT_DTYPES:
    setattr(torch, _attr, torch.float32)  # type: ignore[assignment]
del _attr  # avoid leaking the loop variable into the module namespace

# ttnn dtype aliases preserve the original (pre-rebinding) torch dtypes so
# that element_size returns the correct hardware byte count and promote_dtype
# can compare against them.
bfloat16: torch.dtype = _PROMOTABLE_FLOAT_DTYPES["bfloat16"]
float16: torch.dtype = _PROMOTABLE_FLOAT_DTYPES["float16"]
float32: torch.dtype = torch.float32

# When True (the default), tensor creation functions promote bfloat16 and
# float16 backing to float32 for accurate computation on all host architectures.
# The declared dtype is always preserved in Tensor._dtype for L1 accounting.
# Toggle with set_disable_float32_promotion().
_float32_promotion_enabled: bool = True


def promote_dtype(dtype: "DType") -> torch.dtype:
    """Return the backing torch dtype to use when creating a tensor.

    All narrow floating-point types are promoted to float32 when promotion is
    active (the default), so the simulator runs at full precision on every host.

    Custom types (e.g. bfloat8_b) that have no native PyTorch representation are
    resolved to their native equivalent via _CUSTOM_DTYPE_BACKING, then the same
    promotion logic applies as for native narrow floats.

    The declared dtype is always preserved separately in Tensor._dtype.
    """
    native: torch.dtype = _CUSTOM_DTYPE_BACKING.get(type(dtype), dtype)  # type: ignore[arg-type]
    if _float32_promotion_enabled and native in _PROMOTABLE_FLOAT_DTYPES.values():
        return torch.float32
    return native


def _quotient_dtype(declared: torch.dtype) -> torch.dtype:
    """The dtype a true division reports, given what its operands declare.

    Dividing integers gives a float, in torch and in ttnn alike, so the quotient's
    declared dtype cannot be the operands'. Anything already floating divides to
    itself.
    """
    if declared.is_floating_point:
        return declared
    return torch.get_default_dtype()


def set_disable_float32_promotion(value: bool) -> None:
    """Disable or re-enable the default float32 promotion of floating-point dtypes.

    When promotion is active (the default), all narrow floating-point types are
    backed by float32 for accurate computation on all host architectures:

    - bfloat16 and float16: the corresponding torch.* attributes are rebound to
      float32 so that native PyTorch code also uses float32.
    - bfloat8_b: promote_dtype maps it to bfloat16 and then promotes that to
      float32; no torch attribute rebinding is needed.

    Passing True restores native dtypes throughout (bfloat16/float16 tensors use
    their declared dtype as backing, bfloat8_b is backed by bfloat16).
    Passing False re-enables the default float32 promotion.
    """
    global _float32_promotion_enabled
    _float32_promotion_enabled = not value
    for attr, original in _PROMOTABLE_FLOAT_DTYPES.items():
        setattr(torch, attr, original if value else torch.float32)  # type: ignore[assignment]


class _BFloat8BDtype:
    """Sentinel class for the bfloat8_b block-floating-point dtype.

    PyTorch has no native bfloat8_b type.  The simulator backs bfloat8_b
    tensors with float32 (when float32 promotion is active, the default) or
    bfloat16 (when promotion is disabled) for computation.

    BFP8B encoding: each element is stored as a 1-byte mantissa; every group
    of 16 elements shares a 1-byte exponent.  Storage cost is therefore
    n + n // 16 bytes for n elements, which is not a fixed per-element
    constant.  Use size_in_bytes(n) for correct capacity accounting.
    """

    # Exponent group size: one shared exponent byte per this many elements.
    _EXPONENT_GROUP_SIZE: int = 16

    @property
    def element_size(self) -> int:
        """Mantissa storage width in bytes (excludes shared exponent overhead)."""
        return 1

    def size_in_bytes(self, n_elements: int) -> int:
        """Total bytes required to store n_elements in BFP8B encoding.

        Accounts for both the per-element mantissa byte and the shared
        exponent byte for every group of _EXPONENT_GROUP_SIZE elements.
        Partial groups still require a full exponent byte (ceiling division).
        """
        return n_elements + math.ceil(n_elements / self._EXPONENT_GROUP_SIZE)

    def __repr__(self) -> str:
        return "bfloat8_b"

    def __eq__(self, other: object) -> bool:
        match other:
            case _BFloat8BDtype():
                return True
            case _:
                return False

    def __hash__(self) -> int:
        return hash("bfloat8_b")


bfloat8_b: _BFloat8BDtype = _BFloat8BDtype()

# Type alias for any value that can be passed as a dtype to ttnn tensor
# creation functions: either a standard torch dtype or the bfloat8_b sentinel.
DType = Union[torch.dtype, _BFloat8BDtype]

# Maps custom dtype classes (those with no native torch representation) to the
# torch.dtype that serves as their native backing before promotion is applied.
# promote_dtype looks up a dtype's class here so that custom types are handled
# with the same logic as native narrow floats, without requiring a match branch
# for each one.  To add a new custom dtype, add it here.
_CUSTOM_DTYPE_BACKING: dict[type, torch.dtype] = {
    _BFloat8BDtype: _PROMOTABLE_FLOAT_DTYPES["bfloat16"],
}


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
        'full' grid size used by kernels.

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
    """Mesh-level partition metadata attached to a Tensor by a mesh mapper.

    Stores the logical shape of the mesh device grid and which tensor dimension
    each mesh axis partitions.  ``None`` in ``dims`` means the corresponding
    mesh axis does not partition the tensor; a non-negative integer means it
    partitions that tensor dimension (already normalized by ``from_torch``).

    For a 1D mesh (MeshShape(1, n)) sharding along tensor dim d:
        mesh_shape=(1, n), dims=(None, d)
    For a 2D mesh (MeshShape(rows, cols)) sharding along dims (d0, d1):
        mesh_shape=(rows, cols), dims=(d0, d1)

    Kept separate from MemoryConfig to avoid conflating inter-device
    distribution with intra-device sharding strategies (HEIGHT_SHARDED, etc.).
    """

    mesh_shape: tuple[int, int]
    dims: tuple[Optional[int], Optional[int]]

    @property
    def num_devices(self) -> int:
        """Total number of devices across all mesh axes."""
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def dim(self) -> int:
        """Single partition dim for 1D meshes.

        Returns the one active (non-``None``) entry in ``dims``.  Raises
        ValueError when no axis or more than one axis actively shards the
        tensor; callers should use ``dims`` directly in those cases.
        """
        active = [d for d in self.dims if d is not None]
        if len(active) == 1:
            return active[0]
        if len(active) == 0:
            raise ValueError(
                "MeshShardInfo.dim is undefined: no mesh axis is actively sharding this tensor"
            )
        raise ValueError(
            "MeshShardInfo.dim is ambiguous for 2D-sharded meshes; use .dims directly"
        )


class TensorToMesh:
    """Base class for mesh mappers passed to :func:`from_torch` (mirrors ``ttnn.TensorToMesh``)."""


class ShardTensorToMesh(TensorToMesh):
    """Mapper for from_torch: shards a tensor across a 1D mesh along ``dim``.

    When passed to :func:`from_torch`, the resulting :class:`Tensor` carries a
    :class:`MeshShardInfo` recording the partition axis and mesh shape.
    Collective operations (:func:`all_reduce`, :func:`all_gather`) read this
    metadata to determine the partition structure without consulting global
    device-count state or intra-device sharding strategies.
    """

    def __init__(self, mesh: MeshDevice, dim: int) -> None:
        self.mesh = mesh
        self.dim = dim


class ShardTensor2dMesh(TensorToMesh):
    """Mapper for from_torch: shards a tensor across a 2D mesh device grid.

    Each mesh axis independently partitions a different tensor dimension.
    Pass ``None`` in ``dims`` when a mesh axis should not shard the tensor;
    negative integers are interpreted as Python-style dim indices and normalized.

    Args:
        mesh: 2D mesh device (e.g. from :func:`open_mesh_device`).
        mesh_shape: ``(rows, cols)`` grid shape for the mesh.
        dims: ``(dim_for_row_axis, dim_for_col_axis)`` — which tensor dim
            each mesh axis partitions.  Use ``None`` to leave that axis
            unsharded.

    Example::

        mesh = ttnnsim.open_mesh_device(ttnnsim.MeshShape(2, 4))
        t = ttnnsim.from_torch(
            data,
            mesh_mapper=ttnnsim.ShardTensor2dMesh(mesh, mesh_shape=(2, 4), dims=(0, 1)),
        )
    """

    def __init__(
        self,
        mesh: MeshDevice,
        mesh_shape: tuple[int, int],
        dims: tuple[Optional[int], Optional[int]],
    ) -> None:
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.dims = dims


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


def tile_shape_from_shape(shape: Sequence[int]) -> TtlShape:
    """Tile-grid shape derived purely from an element-space ``shape``.

    Pure function of the input shape (no Tensor instance required) so callers
    can memoise it.  Always interprets ``shape`` as a tiled layout: for >=2-D
    inputs the last two dimensions are divided by ``TILE_SHAPE`` (with H==1
    or W==1 treated as degenerate single-tile dimensions) and leading
    dimensions pass through; for 1-D inputs the single dimension is divided
    by ``TILE_SHAPE[0]``.

    Returns a ``ttl.Shape`` rather than a :class:`Shape`: a tile grid is a
    block shape, which the DSL slices and concatenates freely (a ``Block``
    holds one, and the matmul shape rules take it apart), and which ttnn has no
    notion of.
    """
    dims = tuple(shape)
    if len(dims) == 1:
        w = dims[0]
        tk = 1 if w == 1 else w // TILE_SHAPE[0]
        return (tk,)
    h, w = dims[-2], dims[-1]
    tm = 1 if h == 1 else h // TILE_SHAPE[0]
    tk = 1 if w == 1 else w // TILE_SHAPE[1]
    if len(dims) > 2:
        return (*dims[:-2], tm, tk)
    return (tm, tk)


def tile_count_from_shape(layout: IndexType, shape: Sequence[int]) -> int:
    """Layout-aware logical unit count derived purely from primitives.

    ROW_MAJOR_LAYOUT counts every element; TILE_LAYOUT counts tile-grid
    cells.  Pure function so callers (e.g. copy-handler validation) can
    safely cache the result.
    """
    if layout == ROW_MAJOR_LAYOUT:
        return math.prod(shape)
    return math.prod(tile_shape_from_shape(shape))


def tile_shape_from_tensor(t: "Tensor") -> TtlShape:
    """Return the tile-grid shape of a tensor (thin wrapper over
    :func:`tile_shape_from_shape`).

    Uses the physical :attr:`~Tensor.padded_shape` because tile geometry is a
    property of the stored (tile-aligned) data, not the logical shape.
    """
    return tile_shape_from_shape(t.padded_shape)


def tile_count_from_tensor(t: "Tensor") -> int:
    """Return the number of logical units a Tensor represents (thin wrapper
    over :func:`tile_count_from_shape`).

    Uses the physical :attr:`~Tensor.padded_shape` so tile/element counts
    reflect the stored data, independent of the logical shape.
    """
    return tile_count_from_shape(t.layout, t.padded_shape)


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

    An integer becomes a unit slice so that no dimension is collapsed, which is
    what keeps a key's rank equal to the tensor's.

    Shared by :meth:`Tensor._normalize_index` and :mod:`sim.sharding` when
    interpreting :class:`~sim.typedefs.Selector` values.

    Raises:
        TypeError: For anything that is not an integer or a slice -- an
            ``Ellipsis``, a ``None``, a list of indices, a tensor. Those are all
            things ttnn's element indexing takes and this does not, so they are
            named here rather than left to fail further in as an attribute error
            about ``step``.
    """
    match selector:
        case bool():
            # Before int(): a bool is one, and indexing by True is a mistake
            # rather than a request for element 1.
            raise TypeError(
                "a tensor is indexed by integers and slices, not by True/False"
            )
        case int():
            return slice(selector, selector + 1)
        case slice():
            return selector
        case _:
            raise TypeError(
                f"a tensor is indexed by integers and slices, got "
                f"{type(selector).__name__}; ttnn's fancier element indexing "
                f"(Ellipsis, None, a list of indices, a tensor) is not modelled."
            )


def _tile_extent(dim_size: int, tile_dim: int) -> int:
    """Tiles along a dimension of ``dim_size`` elements.

    A degenerate dimension -- the size-1 one a broadcast operand carries, which
    :meth:`Tensor._validate_tile_alignment` allows through -- occupies one
    (partly used) tile rather than none.
    """
    return -(-dim_size // tile_dim)


def _validate_selector_bounds(
    start: int, stop: int, extent: int, dim_name: str, unit: str
) -> None:
    """Reject bounds that reach outside a dimension of ``extent`` ``unit`` s.

    A kernel addressing data the tensor does not have is a bug in the kernel,
    so it is reported rather than clamped the way a torch or Python slice would
    be.  That also covers an out-of-range index, which arrives here as a unit
    slice, and a negative one, which the specification's ``ttl.Index`` excludes
    and which would otherwise select nothing at all.

    Raises:
        IndexError: If the bounds fall outside ``[0, extent]`` or run backwards.
    """
    if 0 <= start <= stop <= extent:
        return
    raise IndexError(
        f"{dim_name} slice {start}:{stop} is outside the tensor, which has "
        f"{extent} {unit}(s) along it"
    )


def _maybe_resolve_nd_shard_spec_for_tensor(
    tensor_shape: Sequence[int], memory_config: MemoryConfig
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


class Shape(tuple[int, ...]):
    """Dimensions of a tensor, mirroring ``ttnn.Shape``.

    Built from one sequence -- ``Shape([d0, d1, ...])`` or
    ``Shape((d0, d1, ...))`` -- and offering what ttnn's offers: ``len``,
    integer indexing, iteration, equality, :attr:`rank` and :meth:`to_rank`.

    Everything a shape does not do on a device it does not do here either:
    ttnn's ``Shape`` is not a sequence type, so slicing, concatenating and
    repeating one all raise instead of quietly succeeding, which would let
    code pass under the simulator and fail on hardware.  Convert first, as the
    specification's examples do: ``list(shape)[:-2]``.

    The base class is still ``tuple`` so that a shape can be handed to torch's
    factory functions and compared against the plain tuples used as shapes
    everywhere else.  Two things follow from that, and are the respects in
    which this remains the looser of the two: ``isinstance(shape, tuple)`` is
    true here and false on a device, and ``(1,) + shape`` still concatenates,
    because Python hands that to the tuple on the left.

    One cosmetic difference is deliberate: this prints as ``(2, 3)`` where ttnn
    prints ``Shape([2, 3])``.  The simulator's diagnostics quote shapes in their
    messages, alongside the block shapes that are plain tuples, and the ttnn
    spelling reads badly there.

    This is ttnn's ``Shape``, and is a different type from ``ttl.Shape``
    (``sim.typedefs.Shape``), which the specification defines as a tuple of
    dimensions rather than a class and which is only ever an annotation.  Both
    names are shapes, so this module annotates parameters that accept one as
    ``Sequence[int]``, which every spelling of a shape satisfies -- an instance
    of this class, a plain tuple, a list -- and reserves this class for what it
    returns, matching ttnn, where ``Tensor.shape`` is a ``Shape``.
    """

    def __new__(cls, *dims: Sequence[int]) -> "Shape":
        if len(dims) != 1 or isinstance(dims[0], int):
            spelled = ", ".join(repr(d) for d in dims)
            raise TypeError(
                "Shape takes the dimensions as one sequence: "
                f"Shape([{spelled}]), not Shape({spelled})"
            )
        return super().__new__(cls, dims[0])

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self)

    def to_rank(self, new_rank: int) -> "Shape":
        """The same dimensions expressed at ``new_rank``.

        Growing prepends 1s; shrinking drops leading dimensions, which each
        have to be 1 for the shape to survive the trip.
        """
        if new_rank < 0:
            raise TypeError(f"Shape rank must be non-negative, got {new_rank}")
        dims = tuple(self)
        if new_rank >= len(dims):
            return Shape((1,) * (new_rank - len(dims)) + dims)
        dropped = dims[: len(dims) - new_rank]
        if any(d != 1 for d in dropped):
            raise RuntimeError(
                f"Can't convert shape rank: {dims} to rank {new_rank} would "
                f"drop {dropped}, which is not all ones"
            )
        return Shape(dims[len(dims) - new_rank :])

    @overload
    def __getitem__(self, index: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> NoReturn: ...

    def __getitem__(self, index: Union[SupportsIndex, slice]) -> int:
        if isinstance(index, slice):
            raise TypeError(
                "Shape cannot be sliced; index one dimension, or convert "
                "first: tuple(shape)[1:]"
            )
        return super().__getitem__(index)

    def __eq__(self, other: object) -> bool:
        """Equal to any spelling of the same dimensions, as ttnn's is.

        ttnn converts a list or tuple of sizes to a ``Shape`` before comparing,
        so a shape equals both spellings there.  Inheriting tuple's comparison
        would answer False for a list -- not an error a reader would notice,
        just a different answer than the device gives.
        """
        match other:
            case Shape() | tuple() | list():
                return tuple(self) == tuple(cast("Sequence[int]", other))
            case _:
                return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        """Restore the tuple hash that defining ``__eq__`` sets to None."""
        return super().__hash__()

    def _refuse_ordering(self, other: object) -> NoReturn:
        raise TypeError(
            "Shape cannot be ordered; a shape has no order on a device. "
            "Compare the dimensions that matter, or convert: tuple(shape) < ..."
        )

    __lt__ = _refuse_ordering
    __le__ = _refuse_ordering
    __gt__ = _refuse_ordering
    __ge__ = _refuse_ordering

    def __add__(self, other: object) -> NoReturn:
        raise TypeError(
            "Shape cannot be concatenated; convert first: tuple(shape) + ..."
        )

    def __mul__(self, count: object) -> NoReturn:
        raise TypeError("Shape cannot be repeated; convert first: tuple(shape) * n")

    def __rmul__(self, count: object) -> NoReturn:
        raise TypeError("Shape cannot be repeated; convert first: n * tuple(shape)")


def _dtype_element_size(dtype: torch.dtype) -> int:
    """Return the element size in bytes for a torch dtype."""
    return torch.tensor([], dtype=dtype).element_size()


def _dtype_size_in_bytes(dtype: "DType", n_elements: int) -> int:
    """Bytes ``n_elements`` of ``dtype`` occupy on hardware.

    Elements times their width, except for the block-float dtypes, whose shared
    exponents make the cost more than a per-element constant.
    """
    match dtype:
        case _BFloat8BDtype():
            return dtype.size_in_bytes(n_elements)
        case _:
            return n_elements * _dtype_element_size(dtype)


class Tile:
    """Tile geometry, mirroring ``ttnn.Tile``.

    Exposes the geometry ttnn's does -- :attr:`tile_shape`, :attr:`face_shape`,
    :attr:`num_faces`, :meth:`get_tile_size` -- and compares by geometry, as
    ttnn's ``operator==`` does, so two descriptions of the same tile are equal
    rather than merely identical.  The shapes come back as two-element lists,
    which is what a ``std::array<uint32_t, 2>`` reaches Python as.

    The simulator models the one 32x32 tile the DSL uses, so any other geometry
    is refused rather than silently modelled as 32x32.  That is also why
    :attr:`tile_shape` defaults here while ttnn's constructor requires it: there
    is only one shape to ask for, and the flags ttnn accepts for the others
    (``transpose_tile``) are refused.
    """

    def __init__(
        self,
        tile_shape: Sequence[int] = TILE_SHAPE,
        transpose_tile: bool = False,
    ) -> None:
        shape = tuple(int(d) for d in tile_shape)
        if shape != TILE_SHAPE:
            raise ValueError(
                f"the simulator models the {TILE_SHAPE[0]}x{TILE_SHAPE[1]} tile "
                f"only, got {list(shape)}"
            )
        if transpose_tile:
            raise ValueError("the simulator does not model transposed tiles")
        self._tile_shape = shape

    @property
    def tile_shape(self) -> List[int]:
        return list(self._tile_shape)

    @property
    def face_shape(self) -> List[int]:
        return list(FACE_SHAPE)

    @property
    def num_faces(self) -> int:
        return math.prod(self._tile_shape) // math.prod(FACE_SHAPE)

    @property
    def partial_face(self) -> int:
        """0, since a tile shorter than 32 rows is the only partial-face one.

        An ``int`` and not a ``bool`` because ttnn's is a ``uint32_t``.
        """
        return int(self._tile_shape[0] < TILE_SHAPE[0])

    @property
    def narrow_tile(self) -> int:
        """0, since a tile narrower than 32 columns is the only narrow one."""
        return int(self._tile_shape[1] < TILE_SHAPE[1])

    @property
    def transpose_within_face(self) -> bool:
        """False: the constructor refuses a transposed tile."""
        return False

    @property
    def transpose_of_faces(self) -> bool:
        """False, for the same reason as :attr:`transpose_within_face`."""
        return False

    def get_tile_size(self, dtype: "DType") -> int:
        """Bytes one tile of ``dtype`` occupies, as ttnn reports it.

        Sized from the dtype as declared, so ``ttnn.bfloat16`` gives 2048 bytes
        even though the simulator backs that data with float32 for host
        precision. The declared dtype has to be spelled the ttnn way to get the
        hardware answer: ``torch.bfloat16`` is rebound to ``torch.float32`` under
        float32 promotion (see "Float32 Promotion" in docs/sphinx/simulator.md),
        so passing that spelling reports 4096.

        Raises:
            TypeError: If ``dtype`` is not a dtype. Without this a missing dtype
                would take torch's default and report a float32 tile.
        """
        if not isinstance(dtype, (torch.dtype, _BFloat8BDtype)):
            raise TypeError(
                f"get_tile_size needs the tile's dtype, got "
                f"{type(dtype).__name__}; a tile's size is its geometry times "
                f"the width of what it holds."
            )
        return _dtype_size_in_bytes(dtype, math.prod(self._tile_shape))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        return (
            self._tile_shape == other._tile_shape
            and self.face_shape == other.face_shape
        )

    def __hash__(self) -> int:
        return hash((self._tile_shape, FACE_SHAPE))

    def __repr__(self) -> str:
        return f"Tile with shape: [{self._tile_shape[0]}, {self._tile_shape[1]}]"


class Tensor:
    """TTNN-like Tensor wrapper built on torch.Tensor.

    Exposes `.shape`, `.dtype`, and `.layout`.

    Two shapes are tracked, as ttnn does: :attr:`shape` is the logical one the
    caller supplied, and :attr:`padded_shape` is the storage it is held in,
    tile-aligned and at least rank 2 under TILE_LAYOUT.  They differ whenever a
    logical shape is not tile-aligned -- a `(3, 5)` tensor is stored as
    `(32, 32)` -- with the logical data in the top-left of the store.  The two
    spellings of ``to_torch`` differ accordingly: :meth:`to_torch` hands back
    the padded store, which is what a kernel addresses, while the module-level
    :func:`ttnn.to_torch` un-pads to :attr:`shape`, as ttnn's does.

    **Indexing is tile-space and diverges from ttnn deliberately.**  ttnn's
    ``__getitem__`` indexes elements of the logical shape, as torch does, and
    drops a dimension indexed by an integer.  Here, the layout decides: under
    TILE_LAYOUT one index unit is one 32x32 tile and no dimension is dropped,
    which is how the specification addresses tiled blocks and therefore what
    ``ttl.copy`` needs from its operands; under ROW_MAJOR_LAYOUT indices are
    elements, as ttnn's are.  So on a 64x64 tiled tensor ``t[0:2, 0:2]`` is all
    four of its tiles, the whole 64x64, where ttnn would give a 2x2 element
    view, and ``t[0, :]`` is its first row of tiles, ``(32, 64)``, against
    ttnn's ``(64,)``.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        layout: IndexType = TILE_LAYOUT,
        memory_config: MemoryConfig = DRAM_MEMORY_CONFIG,
        dtype: Any = None,
        logical_shape: Optional[Sequence[int]] = None,
        device: Optional[object] = None,
    ) -> None:
        if tensor.ndim < 1:
            raise ValueError(f"Tensor must have at least 1 dimension, got 0-d scalar")
        self._tensor: torch.Tensor = tensor
        self._layout: IndexType = layout
        # The logical (user-visible) shape, mirroring ttnn.Tensor.shape: the
        # dimensions of the actual data, before the store is padded out to whole
        # tiles and before a low-rank input gains the leading unit dimensions
        # that give it a tileable rank -- a length-N vector is lifted to a 1xN row
        # (see _normalize_rank_for_layout) and then tile-padded from there.
        # ``padded_shape`` reports that stored shape.  ``logical_shape`` defaults to the
        # backing tensor's shape for tensors produced by internal ops (e.g.
        # arithmetic / slicing), whose data is already at physical extent; the
        # creation entry points (from_torch / rand / empty / zeros) pass the
        # caller's original shape so ``.shape`` matches ttnn.
        self._logical_shape: Tuple[int, ...] = (
            tuple(int(d) for d in logical_shape)
            if logical_shape is not None
            else tuple(tensor.shape)
        )
        if memory_config.strategy == ShardingStrategy.ND_SHARDED:
            self.memory_config: MemoryConfig = _maybe_resolve_nd_shard_spec_for_tensor(
                tensor.shape, memory_config
            )
        else:
            self.memory_config: MemoryConfig = memory_config
        self.mesh_shard_info: Optional[MeshShardInfo] = None
        # The device handle a creation entry point was given, or None for a host
        # tensor.  The simulator does not model residency, so this is the handle
        # itself and nothing more; see :meth:`device`.
        self._device: Optional[object] = device
        # _dtype is the declared/logical type; defaults to the tensor's native dtype.
        self._dtype: Any = dtype if dtype is not None else tensor.dtype
        # Cached results of _to_element_key() keyed by the raw user key.
        # Indexing patterns in hot loops (e.g. matmul kernels slicing a tile
        # grid) reuse a small set of integer/slice keys millions of times;
        # memoising the element-key conversion eliminates repeated tuple
        # construction, slice arithmetic, and validation work.  Falls back
        # transparently on TypeError for unhashable keys (e.g. older Python
        # without hashable slices).
        self._ek_cache: Dict[Tuple[Selector, ...], Tuple[Selector, ...]] = {}
        # Tile-alignment validation is a per-tensor invariant: once the shape
        # passes, it always passes.  We defer the first check until the first
        # tile-style access (preserving the original error timing) and then
        # latch the result so subsequent _to_element_key() calls skip it.
        self._tile_alignment_checked: bool = False
        # Set by ttnn.deallocate, after which reaching the data is an error.
        # A plain flag rather than dropping the store, because the point is to
        # catch the use, not to model the release: the guard is a bool test on
        # the tile-indexing path, which runs millions of times in a matmul.
        self._deallocated: bool = False

    def _refuse_deallocated(self) -> NoReturn:
        """Report a use of a tensor whose buffer ttnn.deallocate released."""
        raise RuntimeError(
            f"tensor of shape {self._logical_shape} was deallocated; its buffer "
            f"is released and a device would not read it back"
        )

    def _require_allocated(self, *others: object) -> None:
        """Reject a data operation if this tensor or another operand was released."""
        if self._deallocated:
            self._refuse_deallocated()
        for other in others:
            if isinstance(other, Tensor) and other._deallocated:
                other._refuse_deallocated()

    def device(self) -> object:
        """Return the device handle this tensor was created on.

        Mirrors the device method on ttnn.Tensor, which a kernel calls to route
        a derived tensor to the same device as its input.  Only the creation
        entry points (``from_torch`` / ``rand`` / ``empty`` / ``zeros``) carry a
        handle; the simulator does not propagate residency through operations,
        so a tensor produced by arithmetic or slicing has none and this raises
        rather than name a device the simulator did not place it on.

        Raises:
            RuntimeError: The tensor was not created with a device.
        """
        if self._device is None:
            raise RuntimeError(
                "tensor has no device: it was created without one, or the "
                "simulator does not carry a device through the operation that "
                "produced it"
            )
        return self._device

    @property
    def shape(self) -> Shape:
        """Logical (unpadded) shape, mirroring ``ttnn.Tensor.shape``.

        Reflects the dimensions of the actual data as supplied by the caller,
        which for ``TILE_LAYOUT`` tensors can be smaller (and lower-rank) than
        the tile-aligned :attr:`padded_shape` used for physical storage.
        """
        return Shape(self._logical_shape)

    @property
    def padded_shape(self) -> Shape:
        """Shape after tile-alignment padding.

        Mirrors ``ttnn.Tensor.padded_shape``: the shape of the stored data,
        including any zero padding added to reach ``TILE_SHAPE`` multiples for
        ``TILE_LAYOUT`` tensors and any leading unit dimensions a low-rank input
        gained to reach a tileable rank. A length-N vector is lifted to a 1xN row
        and then padded out with it, so a length-5 vector reports ``(32, 32)``
        while its :attr:`shape` stays ``(5,)``. For ``ROW_MAJOR_LAYOUT`` this
        equals ``shape`` apart from a bare scalar, which is stored as a length-1
        vector.
        """
        return Shape(self._tensor.shape)

    @property
    def tile(self) -> Tile:
        """Tile descriptor, mirroring ``ttnn.Tensor.tile``.

        Every simulated tensor is held in the same 32x32 tile, so this describes
        that one.  Built per access rather than handed out from a shared
        instance, because ttnn returns a value here and code that holds on to one
        must not be holding a description every other tensor shares.
        """
        return Tile()

    @property
    def dtype(self) -> Any:
        """Declared logical dtype (e.g. bfloat8_b, torch.bfloat16, torch.float32)."""
        return self._dtype

    @property
    def underlying_dtype(self) -> torch.dtype:
        """PyTorch dtype used for storage and computation.

        May differ from dtype when float32 promotion is active: bfloat16 and
        float16 tensors are backed by float32, so underlying_dtype returns
        torch.float32 while dtype returns the declared type.  For bfloat8_b
        this is torch.float32 (promotion on) or torch.bfloat16 (off).
        """
        return self._tensor.dtype

    @property
    def layout(self) -> IndexType:
        return self._layout

    @property
    def element_size(self) -> int:
        """Number of bytes per element for this tensor's declared dtype.

        Always reflects the declared (hardware) dtype, not the backing
        storage dtype.  For example, a bfloat16 tensor promoted to float32
        backing still returns 2 here.  For dtypes with a shared exponent
        (e.g. bfloat8_b) this returns only the mantissa byte and does not
        include exponent overhead.  Use size_in_bytes(n) for accurate
        multi-element capacity accounting.
        """
        match self._dtype:
            case _BFloat8BDtype():
                return 1
            case _:
                return _dtype_element_size(self._dtype)

    def size_in_bytes(self, n_elements: int) -> int:
        """Total bytes required to store n_elements of this tensor's declared dtype.

        Always uses the declared (hardware) dtype for byte accounting so that
        L1 capacity checks reflect on-hardware footprint regardless of whether
        float32 promotion is active.  For standard torch dtypes this is
        n_elements * element_size.  For dtypes with shared exponents
        (e.g. bfloat8_b) this includes the exponent overhead.
        """
        return _dtype_size_in_bytes(self._dtype, n_elements)

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
    def _resolve_tile_slice(
        s: slice, tile_count: int, dim_name: str
    ) -> Tuple[int, int]:
        """Resolve a tile-coordinate slice to explicit ``(start, stop)`` tile bounds.

        Open ends follow Python slice semantics: a missing start defaults to 0
        and a missing stop defaults to ``tile_count`` (the full extent along the
        dimension). This lets ``t[i, :]`` / ``t[:, j]`` select whole rows/columns
        of tiles.  The bounds are tiles, not elements: see :class:`Tensor` on how
        that diverges from ttnn.  Steps remain unsupported.

        Raises:
            ValueError: If ``step`` is set.
            IndexError: If the bounds reach outside the tensor.
        """
        if s.step is not None:
            raise ValueError(
                f"Tile slice '{dim_name}' must not have a step value, "
                f"got slice({s.start}, {s.stop}, {s.step}). Only simple slices are supported."
            )
        start = 0 if s.start is None else s.start
        stop = tile_count if s.stop is None else s.stop
        _validate_selector_bounds(start, stop, tile_count, dim_name, "tile")
        return start, stop

    def _to_element_key(self, key: Tuple[Selector, ...]) -> Tuple[Selector, ...]:
        """Translate a coordinate key to an element-space index tuple.

        All integer indices are first normalized to unit slices via
        _normalize_index so that no dimension is ever collapsed.

        For ROW_MAJOR_LAYOUT tensors no further scaling is applied.

        For TILE_LAYOUT tensors the last two (row, col) slices are multiplied
        by TILE_SHAPE to convert from tile-space to element-space.  Batch
        slices are left as-is (implicit tile size 1).

        Results are memoised per ``Tensor`` instance.  Hot loops slice the
        same tile coordinates millions of times; the cache turns the second
        and subsequent calls into a single dict lookup.

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
        cache = self._ek_cache
        try:
            cached = cache.get(key)
        except TypeError:
            # Unhashable key (e.g. legacy Python where ``slice`` is not
            # hashable).  Skip the cache entirely on this call.
            return self._compute_element_key(key)
        if cached is not None:
            return cached
        result = self._compute_element_key(key)
        cache[key] = result
        return result

    def _resolve_element_key(
        self, selectors: Tuple[Selector, ...], unit: str
    ) -> Tuple[Selector, ...]:
        """Fill in and check element-space selectors against their dimensions.

        The dimensions of a key that are not tile-scaled: a whole row-major
        key, or the batch part of a tiled one.  Open ends are resolved to the
        dimension's extent, as :meth:`_resolve_tile_slice` does for the tile
        dimensions, so that every selector this returns carries an explicit
        origin.  ``element_slice_starts`` and the slice origin ``__getitem__``
        accumulates for the locality statistics both read that origin, and an
        open end would leave them without one -- reporting the parent's origin
        for a slice that has moved, or refusing the key outright.
        """
        resolved: List[Selector] = []
        for dim, selector in enumerate(selectors):
            if not isinstance(selector, slice):
                resolved.append(selector)
                continue
            extent = self._tensor.shape[dim]
            start = 0 if selector.start is None else selector.start
            stop = extent if selector.stop is None else selector.stop
            _validate_selector_bounds(start, stop, extent, f"dimension {dim}", unit)
            resolved.append(slice(start, stop, selector.step))
        return tuple(resolved)

    def _compute_element_key(self, key: Tuple[Selector, ...]) -> Tuple[Selector, ...]:
        """Uncached body of :meth:`_to_element_key`.

        Split out so the cached fast path stays a few bytecode ops; this
        method is invoked only on cache misses.
        """
        ndim = len(self._tensor.shape)
        if len(key) != ndim:
            raise ValueError(
                f"Key length {len(key)} does not match tensor rank {ndim}: "
                f"expected exactly {ndim} element(s)"
            )

        normalized = tuple(normalize_selector_to_slice(k) for k in key)

        if self._layout == ROW_MAJOR_LAYOUT:
            # Element-space indexing: no tile scaling needed.
            return self._resolve_element_key(normalized, "element")

        # Tile alignment is a per-tensor invariant; check once and latch.
        if not self._tile_alignment_checked:
            self._validate_tile_alignment()
            self._tile_alignment_checked = True
        if ndim == 1:
            col_tiles = _tile_extent(self._tensor.shape[0], TILE_SHAPE[0])
            start, stop = self._resolve_tile_slice(normalized[0], col_tiles, "col")
            return (slice(start * TILE_SHAPE[0], stop * TILE_SHAPE[0]),)
        *batch_s, row_s, col_s = normalized
        # Batch dimensions carry an implicit tile size of 1, so they are already
        # element-space and only need filling in and checking.
        batch_s = list(self._resolve_element_key(tuple(batch_s), "element"))
        row_tiles = _tile_extent(self._tensor.shape[-2], TILE_SHAPE[0])
        col_tiles = _tile_extent(self._tensor.shape[-1], TILE_SHAPE[1])
        row_start, row_stop = self._resolve_tile_slice(row_s, row_tiles, "row")
        col_start, col_stop = self._resolve_tile_slice(col_s, col_tiles, "col")
        return (
            *batch_s,
            slice(row_start * TILE_SHAPE[0], row_stop * TILE_SHAPE[0]),
            slice(col_start * TILE_SHAPE[1], col_stop * TILE_SHAPE[1]),
        )

    def element_slice_starts(self, key: TensorKey) -> Tuple[Index, ...]:
        """Element-space start offset per dimension for ``key`` (``slice.start`` values).

        Uses the same rules as :meth:`__getitem__`: tile indices for
        ``TILE_LAYOUT`` are converted to element bounds; ``ROW_MAJOR_LAYOUT`` keys
        are already element-space.

        An origin rather than a shape, so it is a tuple of :data:`~sim.typedefs.Index`
        and not a :class:`Shape`; ttnn has no type for one.
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
        """Select a sub-tensor, addressing tiles under TILE_LAYOUT.

        The key is in tile units for a tiled tensor and element units for a
        row-major one, and never drops a dimension -- neither of which is what
        ttnn's element indexing does.  See :class:`Tensor`.

        A selector is an integer or a slice without a step, one per dimension;
        the fancier keys ttnn accepts are refused by name
        (:func:`normalize_selector_to_slice`) rather than half-supported.
        """
        # Python passes a bare int/slice (not a tuple) for single-element indexing.
        if self._deallocated:
            self._refuse_deallocated()
        normalized: Tuple[Selector, ...] = key if isinstance(key, tuple) else (key,)
        ek = self._to_element_key(normalized)
        result = Tensor(
            self._tensor[cast(Any, ek)],
            self._layout,
            self.memory_config,
            dtype=self._dtype,
        )
        _name = getattr(self, "_name", None)
        if _name is not None:
            result._name = _name  # type: ignore
        if TRACE.enabled:
            # Accumulate the element-space origin so locality analysis can find
            # the position of this slice within the original (root) sharded
            # tensor.  ``ek`` was just computed above so derive starts directly
            # instead of calling ``element_slice_starts(normalized)`` which would
            # re-invoke ``_to_element_key``.  Every selector ``ek`` holds has an
            # explicit start, including the open-ended ones (``tensor[:]``),
            # which are resolved against the dimension they index.
            # _element_origin is only read by try_count_locality() in sharding.py,
            # which is called from _copy_trace_fields() inside if TRACE.enabled:
            # guards in copy.py, so tracking it is a no-op when tracing is off.
            parent_origin: Tuple[int, ...] = getattr(
                self, "_element_origin", (0,) * self._tensor.ndim
            )
            starts = [s.start if isinstance(s, slice) else s for s in ek]
            result._element_origin = tuple(  # type: ignore[attr-defined]
                p + s for p, s in zip(parent_origin, starts)
            )
        return result

    def __setitem__(self, key: TensorKey, value: "Tensor") -> None:
        """Write a sub-tensor, addressing it as :meth:`__getitem__` does."""
        if self._deallocated:
            self._refuse_deallocated()
        value._require_allocated()
        normalized: Tuple[Selector, ...] = key if isinstance(key, tuple) else (key,)
        self._tensor[cast(Any, self._to_element_key(normalized))] = value._tensor

    def __repr__(self) -> str:
        # Delegate to torch for value and dtype formatting (handles truncation for large tensors).
        layout_str = (
            f", layout={self._layout.name}" if self._layout != TILE_LAYOUT else ""
        )
        # ``shape`` is the logical one, so that it reads as the tensor's shape
        # does everywhere else; the stored extent is named separately when it
        # differs, since the data shown is the padded store.
        padded = tuple(self._tensor.shape)
        padded_str = f", padded_shape={padded}" if padded != self._logical_shape else ""
        data = "<deallocated>" if self._deallocated else repr(self._tensor)
        return (
            f"Tensor(shape={self._logical_shape}{padded_str}{layout_str}, "
            f"data={data})"
        )

    def to_torch(self) -> torch.Tensor:
        """Return the raw backing torch tensor.

        Returns the underlying storage tensor directly so that callers can
        perform in-place operations.  The backing dtype may differ from the
        declared dtype when float32 promotion is active.

        This is the padded store, so its shape is :attr:`padded_shape` and not
        :attr:`shape`, and it deliberately exposes the padding, because the
        padding is what a kernel sees and what the simulator's tile-level
        accesses address.  The logical data occupies the top-left of the store
        (see ``_pad_to_tile_alignment``), so this is not what ttnn's
        ``to_torch`` returns: use the module-level ``ttnn.to_torch()`` for that,
        which un-pads as ttnn does and is what a caller comparing against a
        torch reference wants.

        Raises:
            RuntimeError: the tensor was deallocated.  This is the chokepoint
                every read of the data reaches, ``_logical_view`` and the
                module-level ``to_torch`` among them.
        """
        if self._deallocated:
            self._refuse_deallocated()
        return self._tensor

    # ---- Dry-run helpers ----

    def _broadcast_logical_shape(self, other: "Tensor") -> Tuple[int, ...]:
        """Logical result shape of an element-wise op, mirroring ttnn broadcasting.

        Broadcasts the operands' *logical* shapes (not the padded storage), so
        the result's ``.shape`` matches what ttnn reports even when the operands
        are non-tile-aligned or low-rank.
        """
        return tuple(torch.broadcast_shapes(tuple(self.shape), tuple(other.shape)))

    def _matmul_logical_shape(self, other: "Tensor") -> Tuple[int, ...]:
        """Logical result shape of ``self @ other`` over the logical shapes.

        Uses meta tensors so torch's batched/broadcast matmul rules (and its
        errors on incompatible dims) apply to the logical shapes without
        allocating storage.
        """
        return tuple(
            torch.matmul(
                torch.empty(tuple(self.shape), device="meta"),
                torch.empty(tuple(other.shape), device="meta"),
            ).shape
        )

    def _promoted_dtype(self, other: "Tensor") -> torch.dtype:
        """The dtype two operands' declared dtypes come to, torch's way.

        Reading the left operand's instead makes ``a + b`` and ``b + a`` report
        different dtypes for the same computation, and the answer is not only
        cosmetic: it is what a dataflow buffer built from the result bills as L1
        (``dfb.capacity_bytes``), so the operands' order would move the hardware
        limit warning.
        """
        return torch.promote_types(self._dtype, other._dtype)

    def _zeros_like(self) -> "Tensor":
        """Return a zero tensor with the same shape, dtype, and layout."""
        self._require_allocated()
        return Tensor(
            torch.zeros_like(self._tensor),
            self._layout,
            dtype=self._dtype,
            logical_shape=self._logical_shape,
        )

    def _zeros_broadcast(self, other: "Tensor") -> "Tensor":
        """Return zeros shaped by broadcasting self and other."""
        self._require_allocated(other)
        out_shape = torch.broadcast_shapes(self._tensor.shape, other._tensor.shape)
        return Tensor(
            torch.zeros(out_shape, dtype=self._tensor.dtype),
            self._layout,
            dtype=self._promoted_dtype(other),
            logical_shape=self._broadcast_logical_shape(other),
        )

    def _zeros_matmul(self, other: "Tensor") -> "Tensor":
        """Return zeros shaped like the real matmul output of self @ other.

        The shape is derived exactly as the non-dry path would: the real path
        is ``self._tensor @ other._tensor`` (``torch.matmul``), so running that
        same op on meta tensors reproduces torch's batched/broadcast matmul
        rules (and raises identically on incompatible dims) without allocating
        storage or computing any values.
        """
        self._require_allocated(other)
        out_shape = torch.matmul(
            self._tensor.to("meta"), other._tensor.to("meta")
        ).shape
        return Tensor(
            torch.zeros(out_shape, dtype=self._tensor.dtype),
            self._layout,
            dtype=self._promoted_dtype(other),
            logical_shape=self._matmul_logical_shape(other),
        )

    # ---- Binary operations (element-wise) ----

    def __add__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise addition."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor + other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor + other,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __sub__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise subtraction."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor - other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor - other,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __mul__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise multiplication."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor * other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor * other,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __truediv__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise true division."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor / other._tensor,
                    self._layout,
                    dtype=_quotient_dtype(self._promoted_dtype(other)),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor / other,
                    self._layout,
                    dtype=_quotient_dtype(self._dtype),
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __floordiv__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise floor division."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor // other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor // other,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __mod__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise modulo."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor % other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor % other,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __pow__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise exponentiation."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_broadcast(other)
                return Tensor(
                    self._tensor**other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._broadcast_logical_shape(other),
                )
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    self._tensor**other,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        self._require_allocated(other)
        match other:
            case Tensor():
                if _is_dry_run():
                    return self._zeros_matmul(other)
                return Tensor(
                    self._tensor @ other._tensor,
                    self._layout,
                    dtype=self._promoted_dtype(other),
                    logical_shape=self._matmul_logical_shape(other),
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __neg__(self) -> "Tensor":
        """Unary negation."""
        self._require_allocated()
        if _is_dry_run():
            return self._zeros_like()
        return Tensor(
            -self._tensor,
            self._layout,
            dtype=self._dtype,
            logical_shape=self._logical_shape,
        )

    def __abs__(self) -> "Tensor":
        """Absolute value."""
        self._require_allocated()
        if _is_dry_run():
            return self._zeros_like()
        return Tensor(
            torch.abs(self._tensor),
            self._layout,
            dtype=self._dtype,
            logical_shape=self._logical_shape,
        )

    # ---- Reverse binary operations ----

    def __radd__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise addition."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other + self._tensor,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rsub__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise subtraction."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other - self._tensor,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rmul__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise multiplication."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other * self._tensor,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rtruediv__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise true division."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other / self._tensor,
                    self._layout,
                    dtype=_quotient_dtype(self._dtype),
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rfloordiv__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise floor division."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other // self._tensor,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rmod__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise modulo."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other % self._tensor,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rpow__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise exponentiation."""
        self._require_allocated()
        match other:
            case float() | int():
                if _is_dry_run():
                    return self._zeros_like()
                return Tensor(
                    other**self._tensor,
                    self._layout,
                    dtype=self._dtype,
                    logical_shape=self._logical_shape,
                )
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented


def _normalize_rank_for_layout(tensor: torch.Tensor, layout: IndexType) -> torch.Tensor:
    """Lift a low-rank tensor to the minimum rank a layout can represent.

    Mirrors ttnn: a ``TILE_LAYOUT`` tensor needs at least two dimensions to
    tile, so 0-D / 1-D inputs get leading unit dimensions prepended (a length-N
    vector becomes a single ``1xN`` row; a bare scalar becomes ``1x1``) before
    tile-alignment padding fills each 32x32 tile.  ``ROW_MAJOR_LAYOUT`` keeps
    rank-1 as-is and only lifts a bare scalar to a length-1 vector.

    tt-metal's nightly reduction suite feeds 0-D / 1-D (and 0-volume) shapes
    straight through ``ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT,
    device=device)`` (tests/ttnn/nightly/unit_tests/operations/reduction/
    test_reduction_ops.py, test_generic_ops), so creation ops and ``from_torch``
    accept them alike.  The lift is storage only: ``.shape`` keeps the rank the
    caller passed, while ``.padded_shape`` reports the lifted, tile-aligned one.
    """
    if layout == TILE_LAYOUT and tensor.ndim < 2:
        return tensor.reshape((1,) * (2 - tensor.ndim) + tuple(tensor.shape))
    if layout == ROW_MAJOR_LAYOUT and tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor


def _pad_to_tile_alignment(tensor: torch.Tensor, layout: IndexType) -> torch.Tensor:
    """Lift rank then pad a user tensor's last two dims to ``TILE_SHAPE`` multiples.

    Low-rank inputs are first normalized via :func:`_normalize_rank_for_layout`
    so 0-D / 1-D tensors are accepted (matching ttnn), then per the TT-Lang
    specification every tile is exactly ``TILE_SHAPE`` (32x32) scalar elements
    (see TTLangSpecification.md, tiled-block section).  Logical shapes that are
    not already tile-aligned have their data placed in the top-left of each
    output tile by spec convention - ``(N, 1)`` column vectors live in column 0,
    ``(1, M)`` row vectors live in row 0, ``(1, 1)`` scalars at position
    ``(0, 0)`` - and the remainder of the tile is padding.  The two-step
    ``block.broadcast`` and ``math.reduce_*`` ops then overwrite that padding
    when needed.  ``ROW_MAJOR_LAYOUT`` tensors are returned untouched apart from
    lifting a bare scalar to a length-1 vector.
    """
    tensor = _normalize_rank_for_layout(tensor, layout)
    if layout != TILE_LAYOUT:
        return tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (-h) % TILE_SHAPE[0]
    pad_w = (-w) % TILE_SHAPE[1]
    if pad_h == 0 and pad_w == 0:
        return tensor
    # torch.nn.functional.pad takes (left, right, top, bottom, ...) starting
    # from the last dim; we pad zero on the right of the innermost dim and
    # the bottom of the next-to-innermost dim.
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)


def _logical_view(tensor: Tensor) -> torch.Tensor:
    """The logical data of a tensor, without its padding.

    The inverse of :func:`_pad_to_tile_alignment`: that function places the
    logical data in the top-left of the stored tiles and zero-fills the rest, so
    the data is recovered by slicing each stored dimension back to its logical
    extent and dropping the unit dimensions a low-rank input was lifted through.
    This is what :func:`to_torch` hands out, and what a wrapped ttnn op computes
    on.  The slice keeps the store's memory, so this is a view for every shape
    the simulator stores, but callers should treat it as read-only: a shape
    torch cannot view would come back as a copy, and ttnn's ``to_torch`` returns
    a host copy in any case.  Use :meth:`Tensor.to_torch` to mutate a tensor.
    """
    logical = tuple(tensor.shape)
    stored = tensor.to_torch()
    lifted = (1,) * (stored.ndim - len(logical)) + logical
    return stored[tuple(slice(0, extent) for extent in lifted)].reshape(logical)


def rand(
    shape: Sequence[int],
    dtype: DType = bfloat16,
    layout: IndexType = TILE_LAYOUT,
    device: object = None,
    memory_config: object = None,
) -> Tensor:
    """Create a random tensor with given shape, dtype, and layout."""
    raw = torch.rand(shape, dtype=promote_dtype(dtype))
    return Tensor(
        _pad_to_tile_alignment(raw, layout),
        layout,
        DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
        dtype=dtype,
        logical_shape=tuple(shape),
        device=device,
    )


def empty(
    shape: Sequence[int],
    dtype: DType = bfloat16,
    layout: IndexType = TILE_LAYOUT,
    device: object = None,
    memory_config: object = None,
) -> Tensor:
    """Create an uninitialized tensor with given shape, dtype, and layout."""
    raw = torch.empty(shape, dtype=promote_dtype(dtype))
    return Tensor(
        _pad_to_tile_alignment(raw, layout),
        layout,
        DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
        dtype=dtype,
        logical_shape=tuple(shape),
        device=device,
    )


def zeros(
    shape: Sequence[int],
    dtype: DType = bfloat16,
    layout: IndexType = TILE_LAYOUT,
    device: object = None,
    memory_config: object = None,
) -> Tensor:
    """Create a zero-filled tensor with given shape, dtype, and layout."""
    raw = torch.zeros(shape, dtype=promote_dtype(dtype))
    return Tensor(
        _pad_to_tile_alignment(raw, layout),
        layout,
        DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
        dtype=dtype,
        logical_shape=tuple(shape),
        device=device,
    )


def to_torch(
    t: Union[Tensor, torch.Tensor],
    mesh_composer: Optional[ConcatMeshToTensor] = None,
) -> torch.Tensor:
    """Convert a simulator Tensor or torch.Tensor to torch.Tensor.

    Returns the tensor's logical data, as ttnn's ``to_torch`` does: the result
    has the tensor's :attr:`~Tensor.shape` and not its
    :attr:`~Tensor.padded_shape`, so tile padding never reaches a caller
    comparing against a torch reference, and ``from_torch`` followed by
    ``to_torch`` round-trips a tensor of any shape.  The result is a copy, as
    ttnn's is, so writing to it does not reach the tensor.  Use
    :meth:`Tensor.to_torch` for the padded store instead, which is what a kernel
    addresses and is the storage itself, so it is also the way to mutate a
    tensor in place.

    When float32 promotion is active the dtype is float32 regardless of the
    declared dtype; external torch code also uses float32 (torch.bfloat16 is
    rebound to torch.float32 at module load time), so comparison with natively
    created tensors works without an additional cast.

    Args:
        t: Tensor to convert.
        mesh_composer: Ignored in the simulator; accepted for API compatibility.

    Returns:
        Plain torch.Tensor (backing dtype, not necessarily the declared dtype).
    """
    match t:
        case Tensor() as tw:
            # A copy, as ttnn's is: on a device the result is host memory, so
            # writing to it cannot reach the tensor.  Sharing the store would
            # let a write that a device would drop take effect here.
            return _logical_view(tw).clone()
        case torch.Tensor() as tt:
            return tt
        case _:
            raise TypeError(f"Unsupported type for to_torch: {type(t)}")


def from_torch(
    tensor: torch.Tensor,
    dtype: Optional[DType] = None,
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
        Tensor whose :attr:`~Tensor.shape` is the input's, backed by the input
        (potentially dtype-converted) torch tensor -- rank-lifted and padded into
        tile-aligned storage when the layout is ``TILE_LAYOUT`` and the input is
        not already aligned, in which case :attr:`~Tensor.padded_shape` and the
        backing tensor are larger than the shape passed in.
    """
    if spec is not None:
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(
                f"tensor shape {tuple(tensor.shape)} does not match spec.shape {spec.shape}"
            )
        layout = spec.layout
        eff_dtype = spec.dtype if dtype is None else dtype
        eff_mc = spec.memory_config
    else:
        eff_dtype = dtype
        eff_mc = memory_config if memory_config is not None else DRAM_MEMORY_CONFIG

    # Preserve the caller's logical shape (mirroring ttnn.Tensor.shape) before
    # rank lifting / tile padding rewrites the storage extent below.
    logical_shape = tuple(tensor.shape)

    # Rank lifting for low-rank (0-D / 1-D) inputs and tile-alignment padding are
    # centralized in _pad_to_tile_alignment so from_torch and the creation ops
    # (rand / empty / zeros) accept the same shapes and produce identical layouts.
    tensor = _pad_to_tile_alignment(tensor, layout)

    match eff_dtype:
        case _ if eff_dtype is not None:
            backing = promote_dtype(eff_dtype)
            converted = tensor if tensor.dtype == backing else tensor.to(backing)
            result = Tensor(
                converted,
                layout,
                memory_config=eff_mc,
                dtype=eff_dtype,
                logical_shape=logical_shape,
                device=device,
            )
        case _:
            result = Tensor(
                tensor,
                layout,
                memory_config=eff_mc,
                logical_shape=logical_shape,
                device=device,
            )

    if isinstance(mesh_mapper, ShardTensorToMesh):
        n = mesh_mapper.mesh.num_devices
        d = mesh_mapper.dim % tensor.ndim
        result.mesh_shard_info = MeshShardInfo(mesh_shape=(1, n), dims=(None, d))
    elif isinstance(mesh_mapper, ShardTensor2dMesh):
        rows, cols = mesh_mapper.mesh_shape
        d0, d1 = mesh_mapper.dims
        norm_d0 = None if d0 is None else d0 % tensor.ndim
        norm_d1 = None if d1 is None else d1 % tensor.ndim
        result.mesh_shard_info = MeshShardInfo(
            mesh_shape=(rows, cols), dims=(norm_d0, norm_d1)
        )
    return result


# Strategy-to-ShardingStrategy mapping for create_sharded_memory_config.
_SHARD_STRATEGY_MAP: dict[ShardStrategy, ShardingStrategy] = {
    ShardStrategy.HEIGHT: ShardingStrategy.HEIGHT_SHARDED,
    ShardStrategy.WIDTH: ShardingStrategy.WIDTH_SHARDED,
    ShardStrategy.BLOCK: ShardingStrategy.BLOCK_SHARDED,
}


def create_sharded_memory_config(
    shape: Sequence[int],
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
            shard_grid: ShardGrid = (core_grid.num_cores,)
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
    statistics collection uses the new layout.  Everything else about the
    tensor -- its shape, dtype and layout -- is the same tensor's.
    """
    result = Tensor(
        tensor.to_torch(),
        tensor.layout,
        memory_config,
        dtype=tensor.dtype,
        logical_shape=tensor.shape,
    )
    if hasattr(tensor, "_name"):
        result._name = tensor._name  # type: ignore[attr-defined]
    return result


def to_layout(
    tensor: Tensor,
    layout: IndexType,
    dtype: Optional[DType] = None,
    memory_config: Optional[MemoryConfig] = None,
    sub_core_grids: Optional[CoreRangeSet] = None,
    pad_value: float = 0.0,
) -> Tensor:
    """Re-store a tensor under a different layout, preserving its logical data.

    Mirrors ttnn.to_layout.  A layout decides the store alone: tile alignment
    pads ``TILE_LAYOUT`` storage and ``ROW_MAJOR_LAYOUT`` carries no padding, so
    converting between them is re-padding the logical view rather than a value
    computation.  That is why this is hand-written instead of golden-served -
    a golden reads the logical tensor and so cannot say what the result stores.

    ``sub_core_grids`` selects the cores ttnn runs the relayout on and does not
    affect the result, so it is accepted and ignored.

    Raises:
        ValueError: pad_value is not zero.  The simulator pads every store with
            zero, per the tiled-block section of the specification.
    """
    if pad_value:
        raise ValueError(f"to_layout pads with zero; got pad_value={pad_value}")
    return from_torch(
        _logical_view(tensor).clone(),
        dtype=tensor.dtype if dtype is None else dtype,
        layout=layout,
        memory_config=tensor.memory_config if memory_config is None else memory_config,
    )


def _operand_tensors(values: Iterable[Any]) -> Iterator[Tensor]:
    """The tensor operands among ``values``, reaching through lists and tuples.

    ttnn passes the operands of a join (``concat``, ``stack``) as one sequence,
    so a scan of the top level alone reports that such a call has no tensor
    operands at all.
    """
    for value in values:
        match value:
            case Tensor():
                yield value
            case list() | tuple():
                yield from _operand_tensors(value)


def _has_padded_operand(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    """Whether any tensor operand stores more than its logical extent."""
    return any(
        tuple(t.shape) != tuple(t.padded_shape)
        for t in _operand_tensors(list(args) + list(kwargs.values()))
    )


def _golden_logical_result(
    golden_fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """Result of a golden-wrapped op computed on its inputs' logical data.

    ttnn's operations are defined on the logical tensor; tile padding is how the
    simulator stores it, per :func:`_pad_to_tile_alignment`, which puts the
    logical data in the top-left of the stored tiles and zeroes the rest.
    Running the op on the padded store instead makes the padding part of the
    computation, which gets both halves of a result wrong: its shape (a padded
    ``(3, 5)`` by ``(5, 7)`` matmul reporting ``(32, 32)`` rather than
    ``(3, 7)``), and, for anything that is not elementwise, its values -- a mean
    divided by 1024 elements instead of 15, a softmax normalized over 1009
    zeros it was never given.  It also loses the invariant above for ops that
    move data: concatenating a padded ``(3, 5)`` onto a padded ``(2, 5)`` leaves
    the second operand's rows at row 32 of the store, where nothing can read
    them back as logical data.

    Running on the logical data instead answers all three: the result is what
    ttnn computes, its shape is the shape of that result, and re-padding it
    restores the top-left invariant so the store stays readable.  The cost is
    lower than the padded run's, since the logical data is the smaller tensor.

    Returns ``None`` when this cannot be done, leaving the caller its padded
    run: when no input is padded (the two runs are then the same computation),
    when the op will not run on logical extents (an argument derived from the
    padded ones), or when it does not return a single tensor.

    Operands are found through lists and tuples, not just at the top level:
    ttnn passes the operands of a join (``concat``, ``stack``) as one sequence,
    and leaving those wrapped hands the golden a simulator ``Tensor`` where it
    wants a torch one, which it declines -- sending the very ops that most need
    the logical run to the padded one instead.
    """
    if not _has_padded_operand(args, kwargs):
        return None

    def unpadded(arg: Any) -> Any:
        match arg:
            case Tensor():
                return _logical_view(arg)
            case list() | tuple():
                return type(arg)(unpadded(item) for item in arg)
            case _:
                return arg

    try:
        result = golden_fn(
            *(unpadded(a) for a in args),
            **{k: unpadded(v) for k, v in kwargs.items()},
        )
    except Exception:
        # Deliberately broad: this is arbitrary third-party code being offered
        # inputs of a size it was not called with, and every way it can decline
        # -- an unsupported extent, an argument that only makes sense at padded
        # sizes -- leads here, to the padded run the caller would have made
        # anyway.  Letting any of them escape would fail a supported call.
        return None
    return result if isinstance(result, torch.Tensor) else None


def _elementwise_logical_shape(
    result: torch.Tensor, inputs: Sequence[Any]
) -> Optional[Tuple[int, ...]]:
    """Logical shape for an elementwise op result, mirroring ttnn.

    Broadcasts the ``Tensor`` inputs' logical shapes so a module-level op (e.g.
    ``ttnn.multiply``) reports the same logical ``.shape`` as the equivalent
    operator (``a * b``) and as real ttnn.  This is the fallback for the calls
    :func:`_golden_logical_result` leaves to the padded run, which are the ones
    whose operands carry no padding for it to differ over, plus the few it
    cannot run.  Returns ``None`` -- i.e. keep the physical shape as the logical
    default -- when there are no ``Tensor`` inputs, or when the op was not
    elementwise, so shape-changing ops still report their physical shape.  Not
    elementwise covers two cases: operands that do not broadcast against each
    other at all (``linear``'s ``(32, 64)`` and ``(64, 128)``), and operands that
    do broadcast but whose broadcast does not match the torch result.

    A ``None`` here also decides that a padded run is not answerable at all,
    in :func:`_create_golden_wrapper`: an op that broadcast the padded extents
    onto its result left the padding in the pad region, and one that did not
    moved it into the result.
    """
    tensors = [t for t in inputs if isinstance(t, Tensor)]
    if not tensors:
        return None
    try:
        logical = tuple(torch.broadcast_shapes(*[tuple(t.shape) for t in tensors]))
        padded = tuple(
            torch.broadcast_shapes(*[tuple(t.padded_shape) for t in tensors])
        )
    except RuntimeError:
        # torch reports non-broadcastable operands by raising rather than by
        # returning, and every golden-wrapped op is routed through here, so
        # without this the shape bookkeeping of an op like ``linear`` fails the
        # call itself.  Non-broadcastable operands are not an elementwise op's,
        # which is the same answer the comparison below reaches for the operands
        # that do broadcast.
        return None
    return logical if tuple(result.shape) == padded else None


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise add (simulator shim for ttnn.add)."""
    return a + b


def multiply(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiply (simulator shim for ttnn.multiply)."""
    return a * b


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiply (simulator shim for ttnn.matmul)."""
    return a @ b


def relu(a: Tensor) -> Tensor:
    """Element-wise ReLU (simulator shim for ttnn.relu)."""
    if _is_dry_run():
        return a._zeros_like()
    return Tensor(
        torch.relu(a.to_torch()),
        a.layout,
        dtype=a.dtype,
        logical_shape=a._logical_shape,
    )


def abs(a: Tensor) -> Tensor:
    """Element-wise absolute value (simulator shim for ttnn.abs)."""
    return a.__abs__()


def exp(a: Tensor, fast_and_approximate_mode: bool = False) -> Tensor:
    """Element-wise exponential (simulator shim for ttnn.exp).

    ``fast_and_approximate_mode`` is accepted for ttnn API compatibility and
    ignored; the simulator always computes the exact ``torch.exp``.
    """
    if _is_dry_run():
        return a._zeros_like()
    return Tensor(
        torch.exp(a.to_torch()),
        a.layout,
        dtype=a.dtype,
        logical_shape=a._logical_shape,
    )


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
    """Sum-reduce across simulated devices.

    The partition structure is read from the tensor's :attr:`~Tensor.mesh_shard_info`
    attribute, which is set by :func:`from_torch` when a :class:`ShardTensorToMesh`
    or :class:`ShardTensor2dMesh` mapper is provided.

    The correct output for the all-reduce collective is: sum each group of
    corresponding slices element-wise across all partitions, then give every
    partition that same sum.

    For 2D meshes, ``cluster_axis`` selects which mesh axis to reduce across:

    * ``cluster_axis=0`` — reduce across the row axis (``msi.mesh_shape[0]``
      devices, partitioned along ``msi.dims[0]``).
    * ``cluster_axis=1`` — reduce across the column axis (``msi.mesh_shape[1]``
      devices, partitioned along ``msi.dims[1]``).
    * ``cluster_axis=None`` — reduce across all active mesh axes sequentially.

    Args:
        input_tensor: Input tensor (must have been created with a mesh mapper).
        cluster_axis: Which mesh axis to reduce across (0 or 1).  ``None``
            reduces across all active axes.
        mesh_device: Ignored (accepted for API compatibility).
        memory_config: Optional output memory config.
        dtype: Optional output dtype.
        **kwargs: Additional keyword arguments accepted for API compatibility.

    Returns:
        Tensor where every partition contains the element-wise sum across the
        selected devices.
    """
    msi = input_tensor.mesh_shard_info
    if msi is None:
        raise ValueError("Mesh device is required for all_reduce operation")

    t = input_tensor.to_torch()

    axes = [cluster_axis] if cluster_axis is not None else range(2)
    active_axes = [k for k in axes if msi.dims[k] is not None and msi.mesh_shape[k] > 1]

    result = t
    for k in active_axes:
        d = cast(int, msi.dims[k]) % result.ndim
        n = msi.mesh_shape[k]
        if result.shape[d] % n != 0:
            raise ValueError(
                f"Tensor size {result.shape[d]} along dim {d} is not divisible "
                f"by {n} devices on mesh axis {k}"
            )
        shard = result.shape[d] // n
        # Sum corresponding slices across all n partitions along this mesh axis.
        reduced = sum(result.narrow(d, i * shard, shard) for i in range(n))
        # Every partition gets the same reduced result.
        result = torch.cat([reduced] * n, dim=d).contiguous()  # type: ignore[arg-type]

    if dtype is not None and result.dtype != dtype:
        result = result.to(dtype)

    out_memory_config = (
        memory_config if memory_config is not None else input_tensor.memory_config
    )
    # A reduce leaves the shape alone, so the logical one carries over.
    result_tensor = Tensor(
        result,
        input_tensor.layout,
        out_memory_config,
        dtype=dtype if dtype is not None else input_tensor.dtype,
        logical_shape=input_tensor.shape,
    )
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
    """Gather shards from simulated devices along ``dim``.

    The partition structure is read from the tensor's :attr:`~Tensor.mesh_shard_info`
    attribute, which is set by :func:`from_torch` when a :class:`ShardTensorToMesh`
    or :class:`ShardTensor2dMesh` mapper is provided.

    Each device contributes its local shard.  After the gather every device holds
    an identical result: all shards concatenated along ``dim``.  The simulator
    represents n identical copies by stacking them along the shard dimension,
    matching what ``ttnn.to_torch(..., mesh_composer=ConcatMeshToTensor(...))``
    would produce.

    For 2D meshes, ``cluster_axis`` selects which mesh axis to gather across:

    * ``cluster_axis=0`` — gather across the row axis (``msi.mesh_shape[0]``
      devices, partitioned along ``msi.dims[0]``).
    * ``cluster_axis=1`` — gather across the column axis (``msi.mesh_shape[1]``
      devices, partitioned along ``msi.dims[1]``).
    * ``cluster_axis=None`` — gather across all active mesh axes sequentially,
      applying the same ``dim`` for each.

    Args:
        input_tensor: Input tensor (must have been created with a mesh mapper).
        dim: Dimension along which to concatenate the gathered shards.
        cluster_axis: Which mesh axis to gather across (0 or 1).  ``None``
            gathers across all active axes.
        mesh_device: Ignored (accepted for API compatibility).
        memory_config: Optional output memory config.
        **kwargs: Additional keyword arguments accepted for API compatibility.

    Returns:
        Tensor where every partition contains all shards concatenated along
        ``dim``.
    """
    msi = input_tensor.mesh_shard_info
    if msi is None:
        raise ValueError("Mesh device is required for all_gather operation")

    t = input_tensor.to_torch()

    axes = [cluster_axis] if cluster_axis is not None else range(2)
    active_axes = [k for k in axes if msi.dims[k] is not None and msi.mesh_shape[k] > 1]

    result = t
    gather_dim = dim % t.ndim
    for k in active_axes:
        shard_dim = cast(int, msi.dims[k]) % result.ndim
        n = msi.mesh_shape[k]
        if result.shape[shard_dim] % n != 0:
            raise ValueError(
                f"Tensor size {result.shape[shard_dim]} along dim {shard_dim} is not divisible "
                f"by {n} devices on mesh axis {k}"
            )
        shard_size = result.shape[shard_dim] // n
        # Each device's shard, sliced along shard_dim.
        shards = [
            result.narrow(shard_dim, i * shard_size, shard_size) for i in range(n)
        ]
        # All n devices get the same result: every shard concatenated along gather_dim.
        gathered = torch.cat(shards, dim=gather_dim)
        # Stack n identical copies along shard_dim so the simulator tensor
        # represents all devices holding the gathered result.
        result = torch.cat([gathered] * n, dim=shard_dim).contiguous()

    out_memory_config = (
        memory_config if memory_config is not None else input_tensor.memory_config
    )
    result_tensor = Tensor(
        result, input_tensor.layout, out_memory_config, dtype=input_tensor.dtype
    )
    result_tensor.mesh_shard_info = msi
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

    Operates on the logical tensor, as ttnn's does, and stores the result
    padded again: a size-1 dimension of the logical shape is a whole tile of
    the store, so squeezing the store would find nothing to remove and leave
    the dimension in place.

    Args:
        input_tensor: Input tensor
        dim: If specified, only squeeze this dimension if it has size 1.
             If None, squeeze all dimensions of size 1.

    Returns:
        Tensor with singleton dimensions removed
    """
    logical = _logical_view(input_tensor)
    result = logical.squeeze() if dim is None else logical.squeeze(dim)
    return from_torch(
        result,
        dtype=input_tensor.dtype,
        layout=input_tensor.layout,
        memory_config=input_tensor.memory_config,
    )


def reshape(
    input_tensor: Tensor,
    shape: Sequence[int],
    *,
    memory_config: Optional[MemoryConfig] = None,
    pad_value: Optional[float] = None,
    sub_core_grids: Optional[CoreRangeSet] = None,
) -> Tensor:
    """Give a tensor a new logical shape, keeping its element order.

    Mirrors ttnn.reshape.  Reshapes the logical tensor and stores the result
    padded again, as :func:`squeeze` does: the store's padding is not part of
    the element order the new shape indexes, so reshaping the store would fold
    padding into the result's logical data.  A single ``-1`` is inferred, as
    torch does.

    ``sub_core_grids`` selects the cores ttnn runs on and does not affect the
    result, so it is accepted and ignored.

    Raises:
        ValueError: pad_value is neither None nor zero.  The simulator pads
            every store with zero, per the tiled-block section of the
            specification.
    """
    if pad_value:
        raise ValueError(f"reshape pads with zero; got pad_value={pad_value}")
    return from_torch(
        _logical_view(input_tensor).reshape(tuple(int(d) for d in shape)),
        dtype=input_tensor.dtype,
        layout=input_tensor.layout,
        memory_config=(
            input_tensor.memory_config if memory_config is None else memory_config
        ),
    )


def copy(input_a: Tensor, input_b: Tensor) -> Tensor:
    """Write input_a's logical data into input_b in place and return it.

    Mirrors ttnn.copy, which writes through to an existing tensor rather than
    allocating one, so callers holding ``input_b`` observe the result.  Only
    input_b's logical extent is written; its tile padding keeps whatever it
    already held, matching where :func:`_pad_to_tile_alignment` places logical
    data.

    Raises:
        ValueError: the two tensors do not have the same logical shape, which
            ttnn.copy also requires.
    """
    if tuple(input_a.shape) != tuple(input_b.shape):
        raise ValueError(
            f"copy shape mismatch: source {tuple(input_a.shape)} into destination "
            f"{tuple(input_b.shape)}"
        )
    stored = input_b.to_torch()
    lifted = (1,) * (stored.ndim - len(input_b.shape)) + tuple(input_b.shape)
    region = tuple(slice(0, extent) for extent in lifted)
    stored[region] = _logical_view(input_a).reshape(lifted).to(stored.dtype)
    return input_b


def deallocate(tensor: Tensor, force: bool = True) -> None:
    """Release a tensor's buffer, after which reading it is an error.

    Mirrors ttnn.deallocate.  The simulator does not model the allocator, so
    nothing is freed here; what is modelled is the consequence a kernel can
    observe, that the data is gone.  Reading a deallocated tensor raises rather
    than returning the values it happened to hold, because a device would not
    return them either and a simulator that did would let a use-after-free
    produce a clean answer here and an arbitrary one on hardware.

    ``force=False`` leaves the tensor alone.  ttnn then deallocates only when
    the buffer has a single reference, which the simulator cannot determine;
    invalidating anyway would fail a kernel whose tensor ttnn would have kept.

    Raises:
        RuntimeError: the tensor was already deallocated.
    """
    if not force:
        return
    if tensor._deallocated:
        tensor._refuse_deallocated()
    tensor._deallocated = True


def _fill_pad_region(tensor: Tensor, value: float) -> Tensor:
    """Write ``value`` into everything a tensor stores outside its logical extent."""
    stored = tensor.to_torch()
    lifted = (1,) * (stored.ndim - len(tensor.shape)) + tuple(tensor.shape)
    if tuple(stored.shape) == lifted:
        return tensor
    kept = torch.zeros(stored.shape, dtype=torch.bool)
    kept[tuple(slice(0, extent) for extent in lifted)] = True
    stored[~kept] = value
    return tensor


def _slice(
    input_tensor: Tensor,
    slice_start: Sequence[int],
    slice_end: Sequence[int],
    slice_step: Optional[Sequence[int]] = None,
    *,
    memory_config: Optional[MemoryConfig] = None,
    output_tensor: Optional[Tensor] = None,
    pad_value: Optional[float] = None,
    sub_core_grids: Optional[CoreRangeSet] = None,
) -> Tensor:
    """Take a strided range of a tensor along every dimension.

    Mirrors ttnn.slice, whose ``slice_end`` is exclusive and whose
    ``slice_step`` defaults to 1 on every dimension.  The range is taken from
    the logical data and the result stored padded again, as :func:`reshape`
    does: the store's padding does not lie at the indices the caller is naming,
    so slicing the store would take padding for data.

    Padding of the result is filled with ``pad_value``, or with NaN when the
    caller does not give one.  ttnn documents the implicit tile padding of a
    slice as undefined by default, and NaN is the only value that reports being
    read: filling with zero would let a kernel that addresses padding it was
    never promised produce a clean answer here and an arbitrary one on a
    device.  An integer store has no NaN and keeps the zero it was padded with.

    ``sub_core_grids`` selects the cores ttnn runs on and does not affect the
    result, so it is accepted and ignored.

    Raises:
        ValueError: an index list does not have one entry per dimension, an
            index lies outside the dimension it addresses, a step is not
            positive, a step other than 1 is asked of a bfloat8_b tensor
            (which ttnn does not support), or ``output_tensor`` does not have
            the shape the slice produces.
    """
    logical = _logical_view(input_tensor)
    rank = logical.ndim
    start = [int(v) for v in slice_start]
    end = [int(v) for v in slice_end]
    step = [1] * rank if slice_step is None else [int(v) for v in slice_step]
    if not len(start) == len(end) == len(step) == rank:
        raise ValueError(
            f"slice of a rank-{rank} tensor needs {rank} indices per list; got "
            f"start={len(start)}, end={len(end)}, step={len(step)}"
        )
    for axis, (first, last, stride) in enumerate(zip(start, end, step)):
        extent = logical.shape[axis]
        if stride < 1:
            raise ValueError(f"slice step must be positive; got {stride} on dim {axis}")
        if not 0 <= first < extent:
            raise ValueError(
                f"slice start {first} is outside dim {axis} of extent {extent}"
            )
        if not 0 < last <= extent:
            raise ValueError(
                f"slice end {last} is outside dim {axis} of extent {extent}"
            )
    if any(stride != 1 for stride in step) and isinstance(
        input_tensor.dtype, _BFloat8BDtype
    ):
        raise ValueError("ttnn.slice does not stride a bfloat8_b tensor")

    taken = logical[
        tuple(slice(f, l, s) for f, l, s in zip(start, end, step))
    ].contiguous()
    if output_tensor is not None:
        if tuple(output_tensor.shape) != tuple(taken.shape):
            raise ValueError(
                f"slice produces {tuple(taken.shape)} but output_tensor has "
                f"{tuple(output_tensor.shape)}"
            )
        return copy(from_torch(taken, layout=output_tensor.layout), output_tensor)
    result = from_torch(
        taken,
        dtype=input_tensor.dtype,
        layout=input_tensor.layout,
        memory_config=(
            input_tensor.memory_config if memory_config is None else memory_config
        ),
    )
    if pad_value is not None:
        return _fill_pad_region(result, float(pad_value))
    if not result.to_torch().dtype.is_floating_point:
        return result
    return _fill_pad_region(result, float("nan"))


def _result_storage(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Storage a golden-wrapped op's result must have.

    A golden function computes values and nothing else: they declare ``**_``
    and drop the dtype, layout, memory config and device they were given, so
    ``ttnn.ones(shape, dtype=...)`` comes back float32 whatever was asked for,
    and the wrapper has to record the request the golden discarded.

    Explicit requests match by type rather than by position or name, because
    ttnn spells the same request either way -- ``typecast(t, dtype)``
    positionally, ``ones(shape, dtype=dtype)`` by keyword -- and each of these
    four is a type no operand shares, so a scan cannot mistake one for an
    operand.  Derived tensors inherit anything the call leaves unspecified from
    their first tensor operand, as ttnn's clone, like, and elementwise operations
    do.
    """
    asked: Dict[str, Any] = {}
    for value in list(args) + list(kwargs.values()):
        match value:
            case torch.dtype() | _BFloat8BDtype():
                asked.setdefault("dtype", value)
            case IndexType():
                asked.setdefault("layout", value)
            case MemoryConfig():
                asked.setdefault("memory_config", value)
            case Device() | MeshDevice():
                asked.setdefault("device", value)
    source = next(_operand_tensors(list(args) + list(kwargs.values())), None)
    if source is not None:
        asked.setdefault("layout", source.layout)
        asked.setdefault("memory_config", source.memory_config)
        if source._device is not None:
            asked.setdefault("device", source._device)
    return asked


def _inherit_result_dtype(
    result: torch.Tensor,
    asked: Dict[str, Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Carry an operand's declared dtype when the golden preserves its store.

    Comparisons and similar operations return a new native dtype, so inheriting
    the first operand unconditionally would turn their boolean result into a
    float tensor.  A unique declared dtype whose backing dtype matches the
    golden result identifies clone, like, unary, and same-dtype arithmetic
    results without overriding a genuinely type-changing operation.
    """
    if "dtype" in asked:
        return asked
    candidates = {
        tensor.dtype
        for tensor in _operand_tensors(list(args) + list(kwargs.values()))
        if tensor.underlying_dtype == result.dtype
    }
    if len(candidates) == 1:
        return {**asked, "dtype": candidates.pop()}
    return asked


def _tensor_from_golden(
    result: torch.Tensor,
    logical_shape: Optional[Sequence[int]],
    asked: Dict[str, Any],
    pad: bool,
) -> Tensor:
    """Store a golden's result as the call that produced it asked.

    ``pad`` says the result is a created tensor rather than a derived one, so
    the logical extent the golden returned is the only extent there is and the
    store has to be padded out to leave the top-left invariant every other
    tensor here holds.  A derived result is stored exactly as it arrives: it
    already matches the extent of the operands it came from, which a caller
    that built them from raw torch data is entitled to have left unpadded.
    """
    dtype = asked.get("dtype")
    layout = asked.get("layout", TILE_LAYOUT)
    if dtype is not None:
        result = result.to(promote_dtype(dtype))
    return Tensor(
        _pad_to_tile_alignment(result, layout) if pad else result,
        layout,
        dtype=dtype,
        logical_shape=(
            tuple(result.shape) if logical_shape is None else tuple(logical_shape)
        ),
        **{k: asked[k] for k in ("memory_config", "device") if k in asked},
    )


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
        # Compute on the logical data where the inputs carry padding, and store
        # the result padded again, so a wrapped op behaves as ttnn's does and
        # leaves a tensor indistinguishable from a created one.
        asked = _result_storage(args, kwargs)
        match operation_name:
            case "tilize":
                asked["layout"] = TILE_LAYOUT
            case "untilize":
                asked["layout"] = ROW_MAJOR_LAYOUT
        logical_result = _golden_logical_result(golden_fn, args, kwargs)
        if logical_result is not None:
            return _tensor_from_golden(
                logical_result,
                tuple(logical_result.shape),
                _inherit_result_dtype(logical_result, asked, args, kwargs),
                pad=True,
            )

        # Falling back to the padded run makes the padding part of the
        # computation, which is only harmless when there is none.  Refusing the
        # rest is what lets an op be served on the strength of its logical run
        # alone, rather than on a hand-maintained list of the ops whose padded
        # run is known to be wrong: a wrong answer here is silent, and the
        # caller cannot tell which run produced it.

        # Convert Tensor arguments to torch.Tensor, reaching through the
        # sequence a join passes its operands in as _golden_logical_result
        # does, so an unpadded join is served here rather than declined.
        def convert_arg(arg: Any) -> Any:
            match arg:
                case Tensor():
                    return arg.to_torch()
                case list() | tuple():
                    return type(arg)(convert_arg(item) for item in arg)
                case _:
                    return arg

        torch_args = tuple(convert_arg(arg) for arg in args)
        torch_kwargs = {k: convert_arg(v) for k, v in kwargs.items()}

        # Call golden function
        result = golden_fn(*torch_args, **torch_kwargs)

        # Wrap result in Tensor if it's a torch.Tensor, carrying the logical
        # shape where the elementwise rule can supply one.
        match result:
            case torch.Tensor():
                operands = list(args) + list(kwargs.values())
                logical = _elementwise_logical_shape(result, operands)
                # An op that broadcast its operands' padded extents to its
                # result left the padding where it found it, in the pad region,
                # so the padded run answered what the logical one would have.
                # Anything else moved the padding into the result, and is
                # refused rather than returned: a wrong answer here is silent,
                # and the caller cannot tell which of the two runs produced it.
                # Refusing on this rather than on a name is what lets an op be
                # served on the strength of its own behaviour.
                if logical is None and _has_padded_operand(args, kwargs):
                    raise NotImplementedError(
                        f"ttnn.{operation_name} cannot be simulated on operands "
                        f"that carry tile padding: its golden function declined "
                        f"the logical extents, and on the padded store it does "
                        f"not leave the padding in the pad region."
                    )
                return _tensor_from_golden(
                    result,
                    logical,
                    _inherit_result_dtype(result, asked, args, kwargs),
                    pad=not any(True for _ in _operand_tensors(operands)),
                )
            case _:
                return result

    # Set proper function name and docstring
    wrapper.__name__ = operation_name
    wrapper.__doc__ = (
        f"Wrapper for ttnn.{operation_name} using golden function implementation."
    )

    return wrapper


# Operations the simulator leaves unavailable rather than serve from a golden
# function, because a golden run cannot say where their result lands and
# answering with a tensor laid out wrongly is worse than the AttributeError
# reaching one of these raises, which names the gap.  Each is here for one of
# two reasons, and both are narrower than they once were: _tensor_from_golden
# now carries a call's dtype, layout, memory config and device onto the result,
# and _golden_logical_result runs an op on logical data and re-pads the result,
# which between them serve every op whose only gap was one of those.
#
# Rearranging an operand's data where the logical run cannot stand in (pad,
# bitcast): both are defined on the store rather than on the logical data, so
# running them on a logical view answers a different question than the call
# asked.  pad adds explicit padding, which the logical run would conflate with
# the tile padding it just stripped; bitcast reinterprets the bytes of a dtype,
# which the logical extent does not determine.
#
# Placement the simulator does not model (to_device, from_device, reallocate,
# reshard, sharded_to_interleaved, interleaved_to_sharded, and the _partial
# spellings): there is no residency or allocation here for a golden to compute,
# so these want hand-written metadata ops, as to_memory_config and to_layout
# already are.
#
# Two exceptions belong to neither: arange's golden returns int64 where ttnn
# defaults to bfloat16, and empty_like's golden demands a fill_value its ttnn
# signature makes optional, so both would fail a call that ttnn accepts.
_DECIDES_THE_STORE = {
    "arange",
    "bitcast",
    "empty_like",
    "from_device",
    "interleaved_to_sharded",
    "interleaved_to_sharded_partial",
    "pad",
    "reallocate",
    "reshard",
    "sharded_to_interleaved",
    "sharded_to_interleaved_partial",
    "to_device",
}

# Operations this module implements itself but cannot name in globals(),
# because the name is a builtin the module calls: defining `slice` above would
# shadow the builtin for every call in this file, `_logical_view` and `copy`
# among them.  Held here for the same reason _GOLDEN_WRAPPERS is, and served
# ahead of the golden path, which has nothing to offer for these names anyway.
_HAND_WRITTEN: Dict[str, Callable[..., Any]] = {"slice": _slice}

# Wrappers built by __getattr__, kept out of globals() so that a wrapped op
# named after a builtin this module calls (sum, min, max) cannot shadow the
# builtin: a module's globals are searched before the builtins, but only
# attribute access on the module consults __getattr__.
_GOLDEN_WRAPPERS: Dict[str, Callable[..., Any]] = {}


def __getattr__(name: str) -> Any:
    """Serve a ttnn operation this module does not define from its golden function.

    Python calls this only for names that are not already module attributes, so
    everything defined above wins without needing to be listed anywhere, and a
    name in :data:`_DECIDES_THE_STORE` stays unavailable.

    Raises:
        AttributeError: ttnn is not installed, the name is not a callable ttnn
            operation, it has no golden function, or the simulator decides that
            operation's storage itself.
    """
    hand_written = _HAND_WRITTEN.get(name)
    if hand_written is not None:
        return hand_written
    if (
        not TTNN_AVAILABLE
        or name.startswith("_")
        or name in _DECIDES_THE_STORE
        or not callable(getattr(ttnn, name, None))
    ):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    wrapper = _GOLDEN_WRAPPERS.get(name)
    if wrapper is not None:
        return wrapper
    try:
        golden_fn = ttnn.get_golden_function(getattr(ttnn, name))
    except (RuntimeError, AttributeError):
        # RuntimeError: the operation has no golden function.  AttributeError:
        # the attribute is not an operation at all (an enum, a class).
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    wrapper = _create_golden_wrapper(name, golden_fn)  # type: ignore[arg-type]
    _GOLDEN_WRAPPERS[name] = wrapper
    return wrapper


def __dir__() -> List[str]:
    """Names this module offers, including the golden-backed ones.

    ``dir()`` on a module reports its globals, which the golden-backed
    operations are deliberately not in; without this they would be invisible to
    tab completion and to code that inspects the module.
    """
    names = set(globals()) | set(_HAND_WRITTEN)
    if TTNN_AVAILABLE:
        names.update(
            name
            for name in dir(ttnn)
            if not name.startswith("_") and name not in _DECIDES_THE_STORE
        )
    return sorted(names)
