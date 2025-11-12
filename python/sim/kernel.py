# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kernel generation and grid management utilities.

This module provides decorators and utilities for generating kernels with
specified grid configurations and granularity settings.
"""

import inspect
from typing import Any, Callable, Union, Tuple


def pykernel_gen(
    grid: Union[str, Tuple[int, int]] = "auto", granularity: int = 4
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
        def my_kernel(a, b, out, grid=None):
            # kernel implementation
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Set grid to (2, 2) if 'auto'
            actual_grid = (2, 2) if grid == "auto" else grid

            # Inject granularity into the function's local scope
            frame = inspect.currentframe()
            if frame and frame.f_back:
                frame.f_back.f_locals["granularity"] = granularity

            # Call the original function with the processed grid
            return func(*args, grid=actual_grid, **kwargs)

        # Store the decorator parameters for later access
        setattr(
            wrapper, "__pykernel_config__", {"grid": grid, "granularity": granularity}
        )
        setattr(wrapper, "granularity", granularity)  # Make granularity accessible
        return wrapper

    return decorator
