# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kernel generation and grid management utilities.

This module provides decorators and utilities for generating kernels with
specified grid configurations and granularity settings.
"""

import inspect
from typing import Any, Callable, List, Tuple, Union, cast
import types

from .typedefs import CoreIndex, Index, Shape, Size


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


def flatten_core_index(core_idx: CoreIndex) -> Index:
    """Flatten a CoreIndex to a linear Index.

    Args:
        core_idx: A CoreIndex which can be a single Index or a tuple of Indices

    Returns:
        A linear Index (single integer)

    Example:
        >>> flatten_core_index(5)  # Already linear
        5
        >>> # With grid (8, 8), core (2, 3) -> 2 * 8 + 3 = 19
        >>> flatten_core_index((2, 3))
        19
    """
    match core_idx:
        case int():
            return core_idx
        case _:
            # Convert to linear index using grid dimensions
            grid = _get_from_frame(
                "grid",
                "grid not available - function must be called within a kernel context",
            )

            coords = list(core_idx)

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
        grid_size(dims=3) -> (8, 8, 1) (padded)
    """
    if dims <= 0:
        raise ValueError(f"dims must be positive, got {dims}")

    grid = _get_from_frame(
        "grid", "grid not available - function must be called within a kernel context"
    )

    grid_dims = len(grid)

    if dims == grid_dims:
        # Requested dims matches grid dims
        result = tuple(grid)
    elif dims < grid_dims:
        # Flatten: keep first (dims-1) dimensions, multiply the rest into the last dimension
        # For grid=(8, 8, 8) and dims=2: keep first 1, flatten rest -> (8, 8*8) = (8, 64)
        # For grid=(8, 8) and dims=1: keep first 0, flatten all -> (8*8,) = (64,)
        if dims == 1:
            # Flatten all dimensions
            flattened = 1
            for d in grid:
                flattened *= d
            result = (flattened,)
        else:
            # Keep first (dims-1) dimensions, flatten the rest
            kept = tuple(grid[: dims - 1])
            flattened = 1
            for i in range(dims - 1, grid_dims):
                flattened *= grid[i]
            result = kept + (flattened,)
    else:  # dims > grid_dims
        # Pad at the end with 1s
        # For grid=(8, 8) and dims=3: return (8, 8, 1)
        padding = (1,) * (dims - grid_dims)
        result = tuple(grid) + padding

    # Return int if dims=1, otherwise tuple
    if dims == 1:
        return result[0]
    else:
        return result


def core(dims: Size = 2) -> CoreIndex:
    """Get the current core index from injected context.

    Args:
        dims: Number of dimensions for the core index. Default is 2

    Returns:
        CoreIndex: The core index (int for 1D, tuple for > 1D)

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
        # Calculate flattened first dimension
        flattened = coords[0]
        for i in range(1, len(coords) - dims + 1):
            flattened = flattened * grid[i] + coords[i]
        # Keep the remaining dimensions
        coords = [flattened] + coords[len(coords) - dims + 1 :]

    # Pad with zeros if dims > len(grid)
    while len(coords) < dims:
        coords.append(0)

    if dims == 1:
        return coords[0]
    else:
        return cast(Tuple[Index, Index, *tuple[Index, ...]], tuple(coords))


def kernel(
    grid: Union[str, Shape] = "auto", granularity: Size = 4
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that generates a kernel with specified grid and granularity.

    Args:
        grid: Grid specification. If 'auto', defaults to (8, 8)
        granularity: Number of tiles to process in a batch

    Returns:
        Decorated function with grid and granularity configuration

    Example:
        @kernel(grid="auto", granularity=2)
        def my_kernel(a, b, out):
            # grid and granularity are available as variables here
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create a new function with grid and granularity in its closure
        # This is achieved by modifying the function's globals to include these variables

        # Set grid to (8, 8) if 'auto'
        actual_grid = (8, 8) if grid == "auto" else grid

        # Create new globals dict that includes grid and granularity
        new_globals = func.__globals__.copy()
        new_globals["grid"] = actual_grid
        new_globals["granularity"] = granularity

        # Create a new function with the modified globals
        modified_func = types.FunctionType(
            func.__code__,
            new_globals,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Call the modified function (grid and granularity are already in globals)
            return modified_func(*args, **kwargs)

        # Store the decorator parameters for later access
        setattr(
            wrapper, "__pykernel_config__", {"grid": grid, "granularity": granularity}
        )
        setattr(wrapper, "granularity", granularity)  # Make granularity accessible
        return wrapper

    return decorator
