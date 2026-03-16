# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Function decorators for compute and data movement operations.

This module provides decorators for marking functions as compute or data movement
operations within the simulation framework, along with the BindableTemplate protocol
and the rebind_func_with_ctx utility used by Program to bind decorated functions to
per-core execution contexts.
"""

import types
from types import CellType, FunctionType
from typing import Any, Callable, Dict, List

from .blockstate import ThreadType
from .context import get_context
from .typedefs import BindableTemplate


def _make_cell(value: Any) -> CellType:
    """Create a real closure cell holding `value`."""

    def inner() -> Any:
        return value

    assert inner.__closure__ is not None
    return inner.__closure__[0]


def rebind_func_with_ctx(func: FunctionType, ctx: Dict[str, Any]) -> FunctionType:
    """
    Create a new function from `func` but with:
      - globals = func.__globals__ + ctx
      - closure cells rebuilt from ctx when possible
    so that names like `out_dfb` that were captured will now point to the per-core objects.
    """
    freevars = func.__code__.co_freevars
    orig_closure = func.__closure__ or ()
    orig_cell_map: Dict[str, CellType] = {
        name: cell for name, cell in zip(freevars, orig_closure)
    }

    new_cells: List[CellType] = []
    for name in freevars:
        if name in ctx:
            new_cells.append(_make_cell(ctx[name]))
        else:
            # fall back to original cell if we don't have an override
            new_cells.append(orig_cell_map[name])

    # merge globals with ctx so globals-based lookups also see per-core state
    new_globals: Dict[str, Any] = dict(func.__globals__)
    new_globals.update(ctx)

    new_func = types.FunctionType(
        func.__code__, new_globals, func.__name__, func.__defaults__, tuple(new_cells)
    )
    return new_func


def _register_thread(thread_template: BindableTemplate) -> None:
    """Register a thread template during decoration."""
    get_context().thread_registry.append(thread_template)


def clear_thread_registry() -> None:
    """Clear the thread registry before kernel execution."""
    get_context().thread_registry.clear()


def get_registered_threads() -> List[BindableTemplate]:
    """Get all registered threads and clear the registry."""
    registry = get_context().thread_registry
    threads = list(registry)
    registry.clear()
    return threads


def compute() -> Callable[[FunctionType], BindableTemplate]:
    """
    Decorator to mark a function as a compute operation.

    The decorated function will be executed on compute cores and can access
    the core context including dataflow buffers and core index.

    Returns:
        A BindableTemplate that can be bound to specific execution contexts
    """

    def decorator(func: FunctionType) -> BindableTemplate:
        class ComputeTemplate:
            __name__ = func.__name__
            __wrapped__ = func  # Standard convention from functools.wraps
            thread_type = ThreadType.COMPUTE  # ThreadType enum for type safety

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                # rebuild function with per-core closure
                bound_func = rebind_func_with_ctx(func, ctx)
                return bound_func

        template = ComputeTemplate()
        _register_thread(template)
        return template

    return decorator


def datamovement() -> Callable[[FunctionType], BindableTemplate]:
    """
    Decorator to mark a function as a data movement operation.

    The decorated function will handle data transfers between memory and
    dataflow buffers, and can access the core context.

    Returns:
        A BindableTemplate that can be bound to specific execution contexts
    """

    def decorator(func: FunctionType) -> BindableTemplate:
        class DMTemplate:
            __name__ = func.__name__
            __wrapped__ = func  # Standard convention from functools.wraps
            thread_type = ThreadType.DM  # ThreadType enum for type safety

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                bound_func = rebind_func_with_ctx(func, ctx)
                return bound_func

        template = DMTemplate()
        _register_thread(template)
        return template

    return decorator
