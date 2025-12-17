# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Function decorators for compute and data movement operations.

This module provides decorators for marking functions as compute or data movement
operations within the simulation framework.
"""

from typing import Any, Callable, Dict
from types import FunctionType

from .program import BindableTemplate, rebind_func_with_ctx


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
            _func = func  # Store original function for inspect.getsource()

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                # rebuild function with per-core closure
                bound_func = rebind_func_with_ctx(func, ctx)

                def runner() -> Any:
                    return bound_func()

                # Store original function on runner for cooperative mode
                runner._func = func  # type: ignore[reportFunctionMemberAccess]
                return runner

        return ComputeTemplate()

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
            _func = func  # Store original function for inspect.getsource()

            def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
                bound_func = rebind_func_with_ctx(func, ctx)

                def runner() -> Any:
                    return bound_func()

                # Store original function on runner for cooperative mode
                runner._func = func  # type: ignore[reportFunctionMemberAccess]
                return runner

        return DMTemplate()

    return decorator
