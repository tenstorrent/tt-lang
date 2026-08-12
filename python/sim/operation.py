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

from ttl.constants import validate_math_fidelity

from .typedefs import Shape
from .context import get_context, cleanup_run_context
from .kernel import KernelKind


def set_default_grid(grid: Shape) -> None:
    """Set the default grid size used when kernel specifies grid='full'.

    Args:
        grid: Tuple of (rows, cols) specifying the grid size

    Example:
        set_default_grid((4, 4))  # Use 4x4 grid for 'full'
    """
    get_context().config.default_full_grid = grid


def get_default_grid() -> Shape:
    """Get the current default grid size for grid='full'.

    Returns:
        Tuple of (rows, cols) specifying the default grid size
    """
    return get_context().config.default_full_grid


def _validate_grid(grid: Shape) -> None:
    """Reject a grid that names no node, before a body is ever run.

    A grid with a dimension of zero has no nodes, so nothing runs the operation
    body and the per-node state the run reads is never built; the run then fails
    on a missing node rather than on the grid that has none.  A negative
    dimension is worse, because the node count is a product: ``(-1, -2)`` counts
    two nodes.

    Raises:
        ValueError: If ``grid`` is empty or has a dimension below one.
    """
    if not grid:
        raise ValueError("ttl.operation() grid must have at least one dimension")
    bad = [(axis, size) for axis, size in enumerate(grid) if size < 1]
    if bad:
        listed = ", ".join(f"dimension {axis} is {size}" for axis, size in bad)
        raise ValueError(
            f"ttl.operation() grid {tuple(grid)} names no node: {listed}. "
            "Every grid dimension must be one or more."
        )


def operation(
    grid: Union[str, Shape] = "full",
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    math_fidelity: Optional[str] = None,
    **unknown: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that generates a kernel with specified grid.

    Compute configuration arguments are accepted for compatibility with
    compiler-side code but have no effect in the simulator. Any other
    unrecognised keyword argument raises TypeError to catch user errors early.

    The decorated function's interface is checked against the specification's
    rules for an operation (no parameter defaults, no ``*args`` / ``**kwargs``, no
    return), by the same code the compiler checks with.

    Args:
        grid: Grid specification. If 'auto' or 'full', uses the default grid
            (configurable via set_default_grid()).
        fp32_dest_acc_en: Ignored; accepted for compiler compatibility.
        dst_full_sync_en: Ignored; accepted for compiler compatibility.
        math_fidelity: Ignored; accepted for compiler compatibility.

    Returns:
        Decorated function with grid configuration

    Raises:
        TypeError: If an unrecognised keyword argument is passed.
        ValueError: If ``math_fidelity`` is unsupported, or the grid names no
            node (see :func:`_validate_grid`).

    Example:
        @ttl.operation(grid="full")
        def my_operation(a, b, out):
            # grid is available as a variable here
            pass
    """

    if unknown:
        raise TypeError(
            f"ttl.operation() received unexpected keyword argument(s): "
            f"{', '.join(sorted(unknown))}"
        )

    validate_math_fidelity(math_fidelity)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Set grid to default if 'auto' or 'full'
        actual_grid: Shape = cast(
            Shape,
            (
                get_context().config.default_full_grid
                if grid in ("auto", "full")
                else grid
            ),
        )

        _validate_grid(actual_grid)

        # Create new globals dict that includes grid
        new_globals = func.__globals__.copy()
        new_globals["grid"] = actual_grid

        # A thread-unified operation body (no hand-written @ttl.compute /
        # @ttl.datamovement kernels) is rewritten into an equivalent
        # multi-kernel function by reusing the compiler's thread-assignment
        # splitter; the rest of this decorator then runs it unchanged. A
        # multi-kernel body keeps the original code (and its source lines).
        from .unified_operation import (
            build_multikernel_function,
            is_unified_body,
            validate_operation_interface,
        )

        # The interface rules apply to every operation, so they are checked before
        # anything is done with the body, and with the compiler's own wording.
        validate_operation_interface(func)

        if is_unified_body(func):
            try:
                modified_func = build_multikernel_function(func, new_globals)
            except ValueError as error:
                raise ValueError(f"@ttl.operation({func.__name__}): {error}") from error
        else:
            # Create a new function with the modified globals
            modified_func = types.FunctionType(
                func.__code__,
                new_globals,
                func.__name__,
                func.__defaults__,
                func.__closure__,
            )

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import here to avoid circular dependency.
            from .program import run_operation

            # Execute the operation single-phase: the body is re-run once per
            # node with that node's context injected (so node-dependent setup
            # such as ttl.node() / ttl.grid_size() works), producing per-node
            # DataflowBuffers and kernels; run_operation then aggregates the
            # discovered PipeNets, schedules the active nodes, and runs them.
            # cleanup_run_context() resets execution-specific state afterwards
            # so subsequent runs start from a clean slate.
            try:
                run_operation(modified_func, actual_grid, args, kwargs)
            finally:
                cleanup_run_context()

        # Store the decorator parameters for later access
        setattr(wrapper, "__pykernel_config__", {"grid": grid})
        return wrapper

    return decorator
