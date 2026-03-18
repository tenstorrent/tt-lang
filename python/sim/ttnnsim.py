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
- Core coordinate and range classes for multicore operations
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union, cast

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


# Memory config placeholder (no-op in simulator)
L1_MEMORY_CONFIG = None
DRAM_MEMORY_CONFIG = None

# Type aliases for binary operations
Scalar = Union[float, int]
TensorOrScalar = Union["Tensor", float, int]


class CoreCoord:
    """Coordinate representation for a core in a grid.

    Attributes:
        x: X coordinate (column) of the core
        y: Y coordinate (row) of the core
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"CoreCoord(x={self.x}, y={self.y})"

    def __eq__(self, other: object) -> bool:
        match other:
            case CoreCoord():
                return self.x == other.x and self.y == other.y
            case _:
                return False


class CoreRange:
    """Represents a rectangular range of cores from start to end (inclusive).

    Attributes:
        start: Starting core coordinate (inclusive)
        end: Ending core coordinate (inclusive)
    """

    def __init__(self, start: CoreCoord, end: CoreCoord):
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"CoreRange(start={self.start}, end={self.end})"

    def num_cores(self) -> Count:
        """Calculate the total number of cores in this range."""
        x_range = self.end.x - self.start.x + 1
        y_range = self.end.y - self.start.y + 1
        return x_range * y_range


class CoreRangeSet:
    """Set of core ranges representing a collection of cores.

    This can represent non-contiguous sets of cores by combining
    multiple CoreRange objects.

    Attributes:
        _ranges: List of CoreRange objects
    """

    def __init__(self, ranges: List[CoreRange]):
        self._ranges = ranges

    def ranges(self) -> List[CoreRange]:
        """Get the list of core ranges."""
        return self._ranges

    def num_cores(self) -> Count:
        """Calculate the total number of cores across all ranges."""
        return sum(r.num_cores() for r in self._ranges)

    def __repr__(self) -> str:
        return f"CoreRangeSet(ranges={self._ranges})"


# Dtype aliases
bfloat16 = torch.bfloat16
float32 = torch.float32


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


def tile_count_from_tensor(t: "Tensor") -> int:
    """Return the number of tiles a Tensor represents.

    For 2-D+ tensors the last two element dimensions are divided by TILE_SHAPE
    (treating H==1 or W==1 as degenerate single-tile dimensions); leading
    dimensions are batch dimensions each contributing independently.

    For 1-D tensors the single dimension is divided by TILE_SHAPE[0] (treating
    size==1 as a degenerate single-tile dimension); there are no batch dims.
    """
    import math

    s = t.shape
    if len(s) == 1:
        w = s[0]
        tk = 1 if w == 1 else w // TILE_SHAPE[0]
        return tk
    h, w = s[-2], s[-1]
    tm = 1 if h == 1 else h // TILE_SHAPE[0]
    tk = 1 if w == 1 else w // TILE_SHAPE[1]
    batch = s[:-2]
    return math.prod((*batch, tm, tk))


class Tensor:
    """TTNN-like Tensor wrapper built on torch.Tensor.

    Exposes `.shape` and keeps underlying storage in `_tensor`.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.ndim < 1:
            raise ValueError(f"Tensor must have at least 1 dimension, got 0-d scalar")
        self._tensor: torch.Tensor = tensor

    @property
    def shape(self) -> Shape:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def _validate_tile_alignment(self) -> None:
        """Validate that this tensor supports tile-style indexing.

        For 2-D+ tensors the last two dimensions must be tile-aligned (or
        degenerate); leading batch dimensions may have any size.
        For 1-D tensors the single dimension must be a multiple of TILE_SHAPE[0]
        (or exactly 1).

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
    def _normalize_tile_index(selector: Selector) -> slice:
        """Convert an integer tile index to a unit slice, or return slice as-is."""
        match selector:
            case int():
                return slice(selector, selector + 1)
            case _:
                return selector

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
        """Translate a tile-coordinate key to an element-space index tuple.

        The last two elements of the key are tile-row and tile-col coordinates,
        scaled by TILE_SHAPE to produce element-space slices.  Preceding batch
        elements are passed through unchanged (implicit tile size 1, so
        tile-space and element-space are identical for those dimensions).

        Args:
            key: Tuple whose length must exactly match the tensor's rank.
                For a 1-D tensor: 1 element.  For an N-D tensor (N >= 2): N
                elements, where the last two are tile-row and tile-col
                coordinates and the preceding elements are batch indices.

        Returns:
            Tuple suitable for indexing the underlying torch.Tensor directly.

        Raises:
            ValueError: If key length does not match tensor rank, the tensor
                is not tile-aligned, or a tile slice has missing or stepped
                bounds.
        """
        self._validate_tile_alignment()
        ndim = len(self._tensor.shape)
        if len(key) != ndim:
            raise ValueError(
                f"Key length {len(key)} does not match tensor rank {ndim}: "
                f"expected exactly {ndim} element(s)"
            )
        if ndim == 1:
            col_s = self._normalize_tile_index(key[0])
            self._validate_tile_slice(col_s, "col")
            return (slice(col_s.start * TILE_SHAPE[0], col_s.stop * TILE_SHAPE[0]),)
        *batch, row_k, col_k = key
        row_s = self._normalize_tile_index(row_k)
        col_s = self._normalize_tile_index(col_k)
        self._validate_tile_slice(row_s, "row")
        self._validate_tile_slice(col_s, "col")
        return (
            *batch,
            slice(row_s.start * TILE_SHAPE[0], row_s.stop * TILE_SHAPE[0]),
            slice(col_s.start * TILE_SHAPE[1], col_s.stop * TILE_SHAPE[1]),
        )

    def __getitem__(self, key: TensorKey) -> "Tensor":
        # Python passes a bare int/slice (not a tuple) for single-element indexing.
        normalized: Tuple[Selector, ...] = key if isinstance(key, tuple) else (key,)
        result = Tensor(self._tensor[cast(Any, self._to_element_key(normalized))])
        if hasattr(self, "_name"):
            result._name = self._name  # type: ignore
        return result

    def __setitem__(self, key: TensorKey, value: "Tensor") -> None:
        normalized: Tuple[Selector, ...] = key if isinstance(key, tuple) else (key,)
        self._tensor[cast(Any, self._to_element_key(normalized))] = value._tensor

    def __repr__(self) -> str:
        # Delegate to torch for value and dtype formatting (handles truncation for large tensors).
        return f"Tensor(shape={tuple(self._tensor.shape)}, data={repr(self._tensor)})"

    def to_torch(self) -> torch.Tensor:
        """Public accessor for the underlying torch tensor."""
        return self._tensor

    # ---- Binary operations (element-wise) ----

    def __add__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise addition."""
        match other:
            case Tensor():
                return Tensor(self._tensor + other._tensor)
            case float() | int():
                return Tensor(self._tensor + other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __sub__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise subtraction."""
        match other:
            case Tensor():
                return Tensor(self._tensor - other._tensor)
            case float() | int():
                return Tensor(self._tensor - other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __mul__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise multiplication."""
        match other:
            case Tensor():
                return Tensor(self._tensor * other._tensor)
            case float() | int():
                return Tensor(self._tensor * other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __truediv__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise true division."""
        match other:
            case Tensor():
                return Tensor(self._tensor / other._tensor)
            case float() | int():
                return Tensor(self._tensor / other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __floordiv__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise floor division."""
        match other:
            case Tensor():
                return Tensor(self._tensor // other._tensor)
            case float() | int():
                return Tensor(self._tensor // other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __mod__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise modulo."""
        match other:
            case Tensor():
                return Tensor(self._tensor % other._tensor)
            case float() | int():
                return Tensor(self._tensor % other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __pow__(self, other: TensorOrScalar) -> "Tensor":
        """Element-wise exponentiation."""
        match other:
            case Tensor():
                return Tensor(self._tensor**other._tensor)
            case float() | int():
                return Tensor(self._tensor**other)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        match other:
            case Tensor():
                return Tensor(self._tensor @ other._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __neg__(self) -> "Tensor":
        """Unary negation."""
        return Tensor(-self._tensor)

    def __abs__(self) -> "Tensor":
        """Absolute value."""
        return Tensor(torch.abs(self._tensor))

    # ---- Reverse binary operations ----

    def __radd__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise addition."""
        match other:
            case float() | int():
                return Tensor(other + self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rsub__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise subtraction."""
        match other:
            case float() | int():
                return Tensor(other - self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rmul__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise multiplication."""
        match other:
            case float() | int():
                return Tensor(other * self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rtruediv__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise true division."""
        match other:
            case float() | int():
                return Tensor(other / self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rfloordiv__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise floor division."""
        match other:
            case float() | int():
                return Tensor(other // self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rmod__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise modulo."""
        match other:
            case float() | int():
                return Tensor(other % self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented

    def __rpow__(self, other: Scalar) -> "Tensor":
        """Reverse element-wise exponentiation."""
        match other:
            case float() | int():
                return Tensor(other**self._tensor)
            case _:  # type: ignore[reportUnnecessaryComparison]
                return NotImplemented


def rand(
    shape: Shape, dtype: torch.dtype = bfloat16, layout: Any = TILE_LAYOUT
) -> Tensor:
    """Create a random tensor with given shape and dtype.

    Layout is a placeholder and not used in the simulator, but kept for API compatibility.
    """
    # Use torch.rand; for bfloat16 we cast after creation
    t = torch.rand(shape, dtype=torch.float32)
    t = t.to(dtype)
    return Tensor(t)


def empty(
    shape: Shape, dtype: torch.dtype = bfloat16, layout: Any = TILE_LAYOUT
) -> Tensor:
    """Create an uninitialized tensor with given shape and dtype."""
    t = torch.empty(shape, dtype=dtype)
    return Tensor(t)


def to_torch(t: Union[Tensor, torch.Tensor]) -> torch.Tensor:
    """Convert a simulator Tensor or torch.Tensor to torch.Tensor."""
    match t:
        case Tensor() as tw:
            # Use the public accessor to avoid private attribute usage warnings
            return tw.to_torch()
        case torch.Tensor() as tt:
            return tt
        case _:
            raise TypeError(f"Unsupported type for to_torch: {type(t)}")


def from_torch(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    layout: Any = None,
    device: Optional[Device] = None,
    memory_config: Any = None,
) -> Tensor:
    """Convert a torch.Tensor to a TTNN simulator Tensor.

    Accepts additional keyword arguments for API compatibility with TTNN
    (layout, device, memory_config), but these are no-ops in the simulator.

    Args:
        tensor: Input torch tensor to wrap
        dtype: Optional dtype to convert to (defaults to tensor's dtype)
        layout: Layout parameter (no-op in simulator)
        device: Device parameter (no-op in simulator)
        memory_config: Memory config parameter (no-op in simulator)

    Returns:
        Tensor wrapping the input (potentially converted) torch tensor
    """
    # Convert dtype if specified
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    return Tensor(tensor)


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
