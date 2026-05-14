# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kernel generation and grid management utilities.

This module provides decorators and utilities for generating kernels with
specified grid configurations.
"""

import types
from typing import Any, Callable, Optional, Union, cast

from .blockstate import ThreadType
from .typedefs import Shape
from .context import get_context, cleanup_run_context


def set_default_grid(grid: Shape) -> None:
    """Set the default grid size used when kernel specifies grid='auto'.

    Args:
        grid: Tuple of (rows, cols) specifying the grid size

    Example:
        set_default_grid((4, 4))  # Use 4x4 grid for 'auto'
    """
    get_context().config.default_auto_grid = grid


def get_default_grid() -> Shape:
    """Get the current default grid size for grid='auto'.

    Returns:
        Tuple of (rows, cols) specifying the default grid size
    """
    return get_context().config.default_auto_grid


def operation(
    grid: Union[str, Shape] = "auto",
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    **unknown: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that generates a kernel with specified grid.

    fp32_dest_acc_en and dst_full_sync_en are accepted for compatibility with
    compiler-side code but have no effect in the simulator.  Any other
    unrecognised keyword argument raises TypeError to catch user errors early.

    Args:
        grid: Grid specification. If 'auto' or 'full', uses the default grid
            (configurable via set_default_grid()).
        fp32_dest_acc_en: Ignored; accepted for compiler compatibility.
        dst_full_sync_en: Ignored; accepted for compiler compatibility.

    Returns:
        Decorated function with grid configuration

    Example:
        @ttl.operation(grid="auto")
        def my_operation(a, b, out):
            # grid is available as a variable here
            pass
    """

    if unknown:
        raise TypeError(
            f"ttl.operation() received unexpected keyword argument(s): "
            f"{', '.join(sorted(unknown))}"
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Set grid to default if 'auto' or 'full'
        actual_grid: Shape = cast(
            Shape,
            (
                get_context().config.default_auto_grid
                if grid in ("auto", "full")
                else grid
            ),
        )

        # Create new globals dict that includes grid
        new_globals = func.__globals__.copy()
        new_globals["grid"] = actual_grid

        # Create a new function with the modified globals
        modified_func = types.FunctionType(
            func.__code__,
            new_globals,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import here to avoid circular dependency
            from .decorators import clear_thread_registry, get_registered_threads
            from .program import Program
            from .pipe import build_pipenets, discover_pipe_nets_from_closures

            # Clear thread registry and resource counters before kernel execution
            clear_thread_registry()
            get_context().kernel_dfb_count = 0
            get_context().kernel_l1_bytes = 0

            # Call the modified function (grid is already in globals)
            # This executes the kernel body which defines and registers threads
            modified_func(*args, **kwargs)

            # Get registered threads
            threads = get_registered_threads()

            # All device kernels must register compute, dm0, and dm1.
            if len(threads) != 3:
                raise ValueError(
                    f"Kernel must define exactly 3 kernels (compute, dm0, dm1), got {len(threads)}"
                )

            # Sort threads by type to ensure consistent ordering regardless of definition order
            # Program expects: compute, dm0, dm1
            compute_threads = [
                t
                for t in threads
                if getattr(t, "thread_type", None) == ThreadType.COMPUTE
            ]
            dm_threads = [
                t for t in threads if getattr(t, "thread_type", None) == ThreadType.DM
            ]

            if len(compute_threads) != 1:
                raise ValueError(
                    f"Kernel must define exactly 1 compute kernel, got {len(compute_threads)}"
                )
            if len(dm_threads) != 2:
                raise ValueError(
                    f"Kernel must define exactly 2 datamovement kernels, got {len(dm_threads)}"
                )

            # Arrange in expected order: compute, dm0, dm1
            ordered_threads = [compute_threads[0], dm_threads[0], dm_threads[1]]

            # Build the operation-level PipeNet graph for validation.
            thread_funcs = [getattr(t, "__wrapped__", None) for t in ordered_threads]
            pipe_nets = discover_pipe_nets_from_closures(modified_func, *thread_funcs)
            pipenets = build_pipenets(pipe_nets)
            pipenets.validate()

            # Execute the program with grid parameter.  After the run,
            # clean up execution-specific state so subsequent runs start
            # from a clean slate.  This is the outermost session boundary:
            # thread_registry was already consumed by get_registered_threads()
            # above, so clearing it here is safe.
            try:
                program = Program(*ordered_threads, grid=actual_grid)
                program(*args, **kwargs)
            finally:
                cleanup_run_context()

        # Store the decorator parameters for later access
        setattr(wrapper, "__pykernel_config__", {"grid": grid})
        return wrapper

    return decorator
