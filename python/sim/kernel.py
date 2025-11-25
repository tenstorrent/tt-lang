# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kernel generation and grid management utilities.

This module provides decorators and utilities for generating kernels with
specified grid configurations and granularity settings.
"""

from typing import Any, Callable, Union
from .typedefs import Shape, Size


def pykernel_gen(
    grid: Union[str, Shape] = "auto", granularity: Size = 4
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that generates a kernel with specified grid and granularity.

    Args:
        grid: Grid specification. If 'auto', defaults to (2, 2)
        granularity: Number of tiles to process in a batch

    Returns:
        Decorated function with grid and granularity configuration

    Example:
        @pykernel_gen(grid="auto", granularity=2)
        def my_kernel(a, b, out):
            # grid and granularity are available as variables here
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create a new function with grid and granularity in its closure
        # This is achieved by modifying the function's globals to include these variables
        import types

        # Set grid to (2, 2) if 'auto'
        actual_grid = (2, 2) if grid == "auto" else grid

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
