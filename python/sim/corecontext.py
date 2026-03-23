# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Core coordinate and grid utilities for kernel execution contexts.

Provides functions to query the current core index and grid size from within
a running kernel, and to convert multi-dimensional core coordinates to a
linear index.
"""

import inspect
from typing import Any, List, Union

from .typedefs import CoreCoord, Index, Shape, Size


def _get_from_frame(var_name: str, error_msg: str) -> Any:
    """Helper to walk up the call stack and find a variable.

    Searches through the call stack (locals first, then globals) to find
    a variable by name. This is used by functions like grid_size(), core(),
    and flatten_core_index() to access context variables like 'grid' and '_core'.

    Args:
        var_name: Name of the variable to search for
        error_msg: Error message to raise if not found

    Returns:
        The value of the variable if found

    Raises:
        RuntimeError: If the variable is not found in any frame
    """
    frame = inspect.currentframe()
    # Start from the caller's caller frame (skip _get_from_frame and the immediate caller)
    current_frame = frame.f_back.f_back if frame and frame.f_back else None

    while current_frame:
        # Check locals first (takes precedence)
        if var_name in current_frame.f_locals:
            return current_frame.f_locals[var_name]
        # Then check globals
        if var_name in current_frame.f_globals:
            return current_frame.f_globals[var_name]
        current_frame = current_frame.f_back

    raise RuntimeError(error_msg)


def flatten_core_index(core_coord: CoreCoord) -> Index:
    """Flatten a CoreCoord to a linear Index.

    Args:
        core_coord: A CoreCoord which can be a single Index or a tuple of Indices

    Returns:
        A linear Index (single integer)

    Example:
        >>> flatten_core_index(5)  # Already linear
        5
        >>> # With grid (8, 8), core (2, 3) -> 2 * 8 + 3 = 19
        >>> flatten_core_index((2, 3))
        19
    """
    match core_coord:
        case int():
            return core_coord
        case _:
            # Convert to linear index using grid dimensions
            grid = _get_from_frame(
                "grid",
                "grid not available - function must be called within a kernel context",
            )

            coords = list(core_coord)

            # Calculate linear index: for (y, x) with grid (h, w), linear = y * w + x
            # For 3D: (z, y, x) with grid (d, h, w), linear = z * h * w + y * w + x
            linear = coords[0]
            for i in range(1, len(coords)):
                linear = linear * grid[i] + coords[i]

            return int(linear)


def grid_size(dims: Size = 2) -> Union[Size, Shape]:
    """Get the grid size from the execution context.

    Returns the size of the grid in the specified dimensionality.
    - If requested dims < actual grid dims: highest rank dimensions are flattened
    - If requested dims > actual grid dims: lowest rank dimensions are padded with 1s

    Args:
        dims: Number of dimensions to return (must be positive). Defaults to 2.

    Returns:
        Size if dims == 1, otherwise Tuple[Size, ...] of length dims

    Raises:
        ValueError: If dims is not positive
        RuntimeError: If called outside of a kernel function context

    Example:
        # For grid=(8, 8):
        grid_size(dims=1) -> 64 (flattened)
        grid_size(dims=2) -> (8, 8)
        grid_size(dims=3) -> (8, 8, 1) (padded)
    """
    if dims <= 0:
        raise ValueError(f"dims must be positive, got {dims}")

    grid = _get_from_frame(
        "grid", "grid not available - function must be called within a kernel context"
    )

    grid_dims = len(grid)

    if dims == grid_dims:
        result = tuple(grid)
    elif dims < grid_dims:
        if dims == 1:
            flattened = 1
            for d in grid:
                flattened *= d
            result = (flattened,)
        else:
            kept = tuple(grid[: dims - 1])
            flattened = 1
            for i in range(dims - 1, grid_dims):
                flattened *= grid[i]
            result = kept + (flattened,)
    else:  # dims > grid_dims
        padding = (1,) * (dims - grid_dims)
        result = tuple(grid) + padding

    if dims == 1:
        return result[0]
    else:
        return result


def node(dims: Size = 2) -> CoreCoord:
    """Get the current core coordinates from injected context.

    Args:
        dims: Number of dimensions for the core coordinates. Default is 2

    Returns:
        CoreCoord: The core coordinates (int for 1D, tuple for > 1D)

    Raises:
        RuntimeError: If called outside of a Program context
    """
    cid = _get_from_frame(
        "_core", "core not available - function must be called within Program context"
    )

    grid = _get_from_frame(
        "grid", "grid not available - function must be called within a kernel context"
    )

    coords: List[Index] = []

    for s in reversed(grid):
        coords.append(cid % s)
        cid = cid // s
    coords.reverse()

    # If dims < len(grid), flatten the first dimension(s)
    if dims < len(coords):
        flattened = coords[0]
        for i in range(1, len(coords) - dims + 1):
            flattened = flattened * grid[i] + coords[i]
        coords = [flattened] + coords[len(coords) - dims + 1 :]

    # Pad with zeros if dims > len(grid)
    while len(coords) < dims:
        coords.append(0)

    if dims == 1:
        return coords[0]
    else:
        return tuple(coords)
