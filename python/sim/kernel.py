# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kernel generation and grid management utilities.

This module provides decorators and utilities for generating kernels with
specified grid configurations and granularity settings.
"""

from typing import Any, Callable, Union, Tuple, List, cast
import inspect
from .typedefs import Shape, Size, CoreIndex, Index


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
    if isinstance(core_idx, int):
        return core_idx

    # It's a tuple - convert to linear index using grid dimensions
    grid = grid_size()
    coords = list(core_idx)

    # Calculate linear index: for (y, x) with grid (h, w), linear = y * w + x
    # For 3D: (z, y, x) with grid (d, h, w), linear = z * h * w + y * w + x
    linear = coords[0]
    for i in range(1, len(coords)):
        linear = linear * grid[i] + coords[i]

    return int(linear)


def grid_size() -> Tuple[int, ...]:
    """Get the grid size from the execution context.

    Returns:
        Tuple of grid dimensions (e.g., (height, width) for 2D grid)

    Raises:
        RuntimeError: If called outside of a kernel function context

    Example:
        grid_h, grid_w = ttl.grid_size()
    """
    frame = inspect.currentframe()

    # Walk up the stack to find the frame with 'grid' variable
    current_frame = frame
    while current_frame:
        if "grid" in current_frame.f_globals:
            return current_frame.f_globals["grid"]
        if "grid" in current_frame.f_locals:
            return current_frame.f_locals["grid"]
        current_frame = current_frame.f_back

    raise RuntimeError(
        "grid not available - function must be called within a kernel context"
    )


def core(dims: Size = 2) -> CoreIndex:
    """Get the current core index from injected context.

    Args:
        dims: Number of dimensions for the core index. Default is 2

    Returns:
        CoreIndex: The core index (int for 1D, tuple for > 1D)

    Raises:
        RuntimeError: If called outside of a Program context
    """

    frame = inspect.currentframe()

    # Walk up the call stack to find the frame with '_core'
    cid = None
    while frame:
        if "_core" in frame.f_locals:
            cid = frame.f_locals["_core"]
        if "_core" in frame.f_globals:
            cid = frame.f_globals["_core"]
        frame = frame.f_back

    if cid is None:
        raise RuntimeError(
            "core not available - function must be called within Program context"
        )

    grid = grid_size()

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
        import types

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
