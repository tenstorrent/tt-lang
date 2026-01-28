# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Function decorators for compute and data movement operations.

This module provides decorators for marking functions as compute or data movement
operations within the simulation framework.
"""

from types import FunctionType
from typing import Any, Callable, Dict, List

from .program import BindableTemplate, rebind_func_with_ctx

# Thread registry for automatic collection of @compute and @datamovement threads
_thread_registry: List[BindableTemplate] = []


def _register_thread(thread_template: BindableTemplate) -> None:
    """Register a thread template during decoration."""
    _thread_registry.append(thread_template)


def _clear_thread_registry() -> None:
    """Clear the thread registry before kernel execution."""
    _thread_registry.clear()


def _get_registered_threads() -> List[BindableTemplate]:
    """Get all registered threads and clear the registry."""
    threads = list(_thread_registry)
    _thread_registry.clear()
    return threads


def compute() -> Callable[[FunctionType], BindableTemplate]:
    """
    Decorator to mark a function as a compute operation.

    The decorated function will be executed on compute cores and can access
    the core context including circular buffers and core index.

    Returns:
        A BindableTemplate that can be bound to specific execution contexts
    """

    def decorator(func: FunctionType) -> BindableTemplate:
        class ComputeTemplate:
            __name__ = func.__name__
            __wrapped__ = func  # Standard convention from functools.wraps
            thread_type = "compute"  # Identifier for sorting

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                # rebuild function with per-core closure
                bound_func = rebind_func_with_ctx(func, ctx)

                def runner() -> Any:
                    return bound_func()

                # Store original function on runner for cooperative mode
                runner.__wrapped__ = func  # type: ignore[reportFunctionMemberAccess]
                return runner

        template = ComputeTemplate()
        _register_thread(template)
        return template

    return decorator


def datamovement() -> Callable[[FunctionType], BindableTemplate]:
    """
    Decorator to mark a function as a data movement operation.

    The decorated function will handle data transfers between memory and
    circular buffers, and can access the core context.

    Returns:
        A BindableTemplate that can be bound to specific execution contexts
    """

    def decorator(func: FunctionType) -> BindableTemplate:
        class DMTemplate:
            __name__ = func.__name__
            __wrapped__ = func  # Standard convention from functools.wraps
            thread_type = "datamovement"  # Identifier for sorting

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                bound_func = rebind_func_with_ctx(func, ctx)

                def runner() -> Any:
                    return bound_func()

                # Store original function on runner for cooperative mode
                runner.__wrapped__ = func  # type: ignore[reportFunctionMemberAccess]
                return runner

        template = DMTemplate()
        _register_thread(template)
        return template

    return decorator
