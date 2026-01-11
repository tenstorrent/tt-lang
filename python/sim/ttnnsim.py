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

from typing import Any, List, Optional, Tuple, Union, cast

import torch

from .constants import TILE_SHAPE
from .tensoraccessor import TensorAccessor
from .typedefs import Count, IndexType, Shape

# Public constants (mirror TTL constants)
TILE_SIZE: int = TILE_SHAPE[0]
TILE_LAYOUT = IndexType.TILE

# Memory config placeholder (no-op in simulator)
L1_MEMORY_CONFIG = None

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
        if not isinstance(other, CoreCoord):
            return False
        return self.x == other.x and self.y == other.y


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


def open_device(device_id: int = 0) -> Device:
    """Open a simulated device (no-op)."""
    return Device(device_id)


def close_device(device: Device) -> None:
    """Close a simulated device (no-op)."""
    # Nothing to do in simulator
    return None


class Tensor:
    """TTNN-like Tensor wrapper built on torch.Tensor.

    Exposes `.shape` and keeps underlying storage in `_tensor`.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor: torch.Tensor = tensor
        # Accessor is created lazily only when tile-style indexing is used
        self._accessor: Optional[TensorAccessor] = None

    @property
    def shape(self) -> Shape:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def __getitem__(self, key: Any) -> "Tensor":
        # If key looks like tile-style indexing (two slices), use TensorAccessor
        if isinstance(key, tuple):
            key_t = cast(Tuple[Any, ...], key)
            if len(key_t) == 2:
                row_key = key_t[0]
                col_key = key_t[1]

                # Check if both are integers (tile indexing like a[m, k])
                if isinstance(row_key, int) and isinstance(col_key, int):
                    # Check if tensor is tile-aligned; if not, use regular indexing
                    if (
                        len(self._tensor.shape) != 2
                        or self._tensor.shape[0] % TILE_SHAPE[0] != 0
                        or self._tensor.shape[1] % TILE_SHAPE[1] != 0
                    ):
                        return Tensor(self._tensor.__getitem__(cast(Any, key)))
                    # Use tile indexing for tile-aligned tensors
                    row_slice: slice = slice(row_key, row_key + 1)
                    col_slice: slice = slice(col_key, col_key + 1)
                    self._ensure_accessor()
                    assert self._accessor is not None
                    return Tensor(self._accessor.get_tiles(row_slice, col_slice))

                # Check if both are slices (slice indexing)
                if isinstance(row_key, slice) and isinstance(col_key, slice):
                    self._ensure_accessor()
                    assert self._accessor is not None
                    return Tensor(
                        self._accessor.get_tiles(
                            cast(slice, row_key), cast(slice, col_key)
                        )
                    )

        return Tensor(self._tensor.__getitem__(cast(Any, key)))

    def __setitem__(self, key: Any, value: Union["Tensor", torch.Tensor, Any]) -> None:
        # If setting via tile-style indexing, route through accessor
        if isinstance(key, tuple):
            key_t = cast(Tuple[Any, ...], key)
            if len(key_t) == 2:
                row_key = key_t[0]
                col_key = key_t[1]

                # Check if both are integers (tile indexing like a[m, k])
                if isinstance(row_key, int) and isinstance(col_key, int):
                    # Check if tensor is tile-aligned; if not, use regular indexing
                    if (
                        len(self._tensor.shape) != 2
                        or self._tensor.shape[0] % TILE_SHAPE[0] != 0
                        or self._tensor.shape[1] % TILE_SHAPE[1] != 0
                    ):
                        match value:
                            case Tensor() as tval:
                                self._tensor.__setitem__(cast(Any, key), tval._tensor)
                            case torch.Tensor() as tt:
                                self._tensor.__setitem__(cast(Any, key), tt)
                            case _:
                                self._tensor.__setitem__(cast(Any, key), value)
                        return
                    # Use tile indexing for tile-aligned tensors
                    row_slice: slice = slice(row_key, row_key + 1)
                    col_slice: slice = slice(col_key, col_key + 1)
                    self._ensure_accessor()
                    assert self._accessor is not None
                    match value:
                        case Tensor() as tval:
                            self._accessor.set_tiles(row_slice, col_slice, tval._tensor)
                        case torch.Tensor() as tt:
                            self._accessor.set_tiles(row_slice, col_slice, tt)
                        case _:
                            self._accessor.set_tiles(row_slice, col_slice, value)
                    return

                # Check if both are slices (slice indexing)
                if isinstance(row_key, slice) and isinstance(col_key, slice):
                    self._ensure_accessor()
                    assert self._accessor is not None
                    match value:
                        case Tensor() as tval:
                            self._accessor.set_tiles(
                                cast(slice, row_key), cast(slice, col_key), tval._tensor
                            )
                        case torch.Tensor() as tt:
                            self._accessor.set_tiles(
                                cast(slice, row_key), cast(slice, col_key), tt
                            )
                        case _:
                            self._accessor.set_tiles(
                                cast(slice, row_key), cast(slice, col_key), value
                            )
                    return

        match value:
            case Tensor() as tval:
                self._tensor.__setitem__(cast(Any, key), tval._tensor)
            case torch.Tensor() as tt:
                self._tensor.__setitem__(cast(Any, key), tt)
            case _:
                self._tensor.__setitem__(cast(Any, key), value)

    def __repr__(self) -> str:
        return f"Tensor(shape={tuple(self._tensor.shape)}, dtype={self._tensor.dtype})"

    def to_torch(self) -> torch.Tensor:
        """Public accessor for the underlying torch tensor."""
        return self._tensor

    def _ensure_accessor(self) -> None:
        """Create a TensorAccessor for tile-based indexing when possible.

        Raises ValueError if the underlying tensor is not compatible with
        tile-based access (i.e., not 2D or not multiple of tile dims).
        """
        if self._accessor is not None:
            return

        # Only create accessor when the tensor dimensions align with TILE_SHAPE
        if len(self._tensor.shape) != 2:
            raise ValueError("Tile-style indexing requires a 2D tensor")

        if (
            self._tensor.shape[0] % TILE_SHAPE[0] != 0
            or self._tensor.shape[1] % TILE_SHAPE[1] != 0
        ):
            raise ValueError(
                "Tensor shape is not a multiple of tile shape; cannot create TensorAccessor"
            )

        self._accessor = TensorAccessor(self._tensor, index_type=IndexType.TILE)

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
    shape: Tuple[int, ...], dtype: torch.dtype = bfloat16, layout: Any = TILE_LAYOUT
) -> Tensor:
    """Create a random tensor with given shape and dtype.

    Layout is a placeholder and not used in the simulator, but kept for API compatibility.
    """
    # Use torch.rand; for bfloat16 we cast after creation
    t = torch.rand(shape, dtype=torch.float32)
    t = t.to(dtype)
    return Tensor(t)


def empty(
    shape: Tuple[int, ...], dtype: torch.dtype = bfloat16, layout: Any = TILE_LAYOUT
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


def isclose(
    a: Tensor,
    b: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Tensor:
    """
    Element-wise comparison of two tensors, returning a boolean tensor indicating
    whether |a - b| <= atol + rtol * |b|.

    Accepts either sim.ttnnsim.Tensor or torch.Tensor (or objects coercible to
    torch tensors). Returns a sim.ttnnsim.Tensor wrapping a torch.bool tensor.

    Args:
        a, b: operands to compare (ttnn.Tensor or torch.Tensor or array-like)
        rtol: relative tolerance
        atol: absolute tolerance
        equal_nan: if True, NaNs in the same position are treated as equal

    Behavior follows numpy/torch isclose semantics.
    """

    # Normalize inputs to torch.Tensor
    ta = a.to_torch()
    tb = b.to_torch()

    # Promote to a floating dtype for safe relative comparison if needed
    if not ta.is_floating_point() or not tb.is_floating_point():
        promoted = torch.float32
    else:
        promoted = torch.promote_types(ta.dtype, tb.dtype)

    ta = ta.to(promoted)
    tb = tb.to(promoted)

    # Compute closeness
    diff = torch.abs(ta - tb)
    tol = atol + rtol * torch.abs(tb)
    result = diff <= tol

    if equal_nan:
        both_nan = torch.isnan(ta) & torch.isnan(tb)
        result = result | both_nan

    # Wrap result in ttnn.Tensor for public API consistency
    return Tensor(result)


def repeat(input_tensor: Tensor, repetition_vector: Shape) -> Tensor:
    """Repeat the input tensor according to the repetition vector.

    Returns a new tensor filled with repetition of input_tensor according to
    the number of times specified in repetition_vector.

    Note: This function is not fully defined after the original TTNN API.
    The original API includes additional keyword arguments (e.g., memory_config)
    which are not implemented in this simulator version.

    Args:
        input_tensor (Tensor): The input tensor to repeat.
        repetition_vector (Shape): The number of repetitions for each dimension.

    Returns:
        Tensor: The output tensor with repeated values.

    Example:
        >>> a = ttnn.rand((2, 3), dtype=ttnn.float32)
        >>> b = ttnn.repeat(a, (2, 3))  # Shape becomes (4, 9)
    """

    # Convert input tensor to torch
    t = input_tensor.to_torch()

    # Use torch.repeat to perform the repetition
    repeated_t = t.repeat(*repetition_vector)

    # Wrap result back in simulator Tensor
    return Tensor(repeated_t)


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
    if isinstance(core_grid, CoreCoord):
        # Create a CoreRangeSet from the grid dimensions
        num_cores = core_grid.x * core_grid.y
        all_cores = CoreRangeSet(
            [CoreRange(CoreCoord(0, 0), CoreCoord(core_grid.x - 1, core_grid.y - 1))]
        )
        grid_size = (core_grid.x, core_grid.y)
    else:
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
        if isinstance(core_grid, CoreCoord) and grid_size:
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
        else:
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
        if isinstance(core_grid, CoreCoord) and grid_size:
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
                core_group_1 = CoreRangeSet([CoreRange(c, c) for c in group_1_cores])
            else:
                core_group_1 = CoreRangeSet([])

            if group_2_cores:
                core_group_2 = CoreRangeSet([CoreRange(c, c) for c in group_2_cores])
            else:
                core_group_2 = CoreRangeSet([])
        else:
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
                core_group_1 = CoreRangeSet([CoreRange(c, c) for c in group_1_cores])
            else:
                core_group_1 = CoreRangeSet([])

            if group_2_cores:
                core_group_2 = CoreRangeSet([CoreRange(c, c) for c in group_2_cores])
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
