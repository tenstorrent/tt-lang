# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Program execution framework for multi-core simulation.

This module provides the core execution framework for running compute and data movement
functions across multiple cores with proper context binding and error handling.
"""

import types
import copy
import threading
import traceback
import inspect
from typing import Any, Callable, Dict, List, Tuple, Protocol
from types import CellType, FunctionType

import torch

from .cb import CircularBuffer
from .cbapi import CBAPI
from .tensoraccessor import TensorAccessor
from .typedefs import CoreIndex


# Protocol for templates that have a bind method
class BindableTemplate(Protocol):
    """Protocol for templates that can be bound to a specific execution context."""

    __name__: str

    def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
        """Bind the template to a specific execution context."""
        ...


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
    so that names like `out_cb` that were captured will now point to the per-core objects.
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


def core(dims: int = 2) -> int:
    """Get the current core index from injected context.

    Args:
        dims: Number of dimensions for the core index. Only dims=1 is supported.
              Default is 2 for backward compatibility but will raise an error.

    Returns:
        int: The linear index of the current core in the grid (only when dims=1)

    Raises:
        RuntimeError: If called outside of a Program context
        ValueError: If dims != 1
    """
    if dims != 1:
        raise ValueError(
            f"core() only supports dims=1, got dims={dims}. "
            "Use core(dims=1) to get the linear core index."
        )

    frame = inspect.currentframe()

    # Check the calling frame's globals for the injected '_core' variable
    if frame and frame.f_back and "_core" in frame.f_back.f_globals:
        core_index: CoreIndex = frame.f_back.f_globals["_core"]
        return core_index

    raise RuntimeError(
        "core not available - function must be called within Program context"
    )


def Program(*funcs: BindableTemplate) -> Any:
    """Program class that combines compute and data movement functions."""

    class ProgramImpl:
        def __init__(self, *functions: BindableTemplate):
            self.functions = functions
            self.context: Dict[str, Any] = {}

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                # capture locals and globals from the caller (eltwise_add)
                # Check globals first for decorator-injected variables, then locals
                self.context = {}
                # Add relevant items from globals (like grid, granularity from decorators)
                caller_globals = frame.f_back.f_globals
                for key in ["grid", "granularity"]:
                    if key in caller_globals:
                        self.context[key] = caller_globals[key]
                # Add locals (which take precedence)
                self.context.update(frame.f_back.f_locals)

            grid = self.context.get("grid", (1, 1))
            total_cores = grid[0] * grid[1]

            compute_func_tmpl, dm0_tmpl, dm1_tmpl = self.functions

            # collect user-facing errors here
            errors: List[str] = []

            for core in range(total_cores):
                # build per-core context
                memo: Dict[int, Any] = {}
                core_context: Dict[str, Any] = {}
                api = CBAPI()  # new CBAPI per core

                for key, value in self.context.items():
                    match value:
                        case torch.Tensor() | TensorAccessor():
                            core_context[key] = value
                            memo[id(value)] = value
                        case CircularBuffer():
                            # create a fresh CB for this core
                            core_context[key] = CircularBuffer(
                                shape=value.shape,
                                buffer_factor=value.buffer_factor,
                                api=api,
                            )
                        case _:
                            core_context[key] = copy.deepcopy(value, memo)

                # also make the core number visible
                core_context["_core"] = core

                # bind per-core
                core_dm0 = dm0_tmpl.bind(core_context)
                core_compute = compute_func_tmpl.bind(core_context)
                core_dm1 = dm1_tmpl.bind(core_context)

                # run the three in parallel threads, because CB ops are blocking
                # we store (stage_name, exception, traceback_str)
                thread_results: List[Tuple[str, Exception, str]] = []

                def run_func_in_thread(
                    name: str, func_factory: Callable[[], Any]
                ) -> None:
                    try:
                        func_factory()  # Execute the function factory directly
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        thread_results.append((name, e, tb_str))

                t_dm0 = threading.Thread(
                    target=run_func_in_thread, args=(f"core{core}-dm0", core_dm0)
                )
                t_comp = threading.Thread(
                    target=run_func_in_thread,
                    args=(f"core{core}-compute", core_compute),
                )
                t_dm1 = threading.Thread(
                    target=run_func_in_thread, args=(f"core{core}-dm1", core_dm1)
                )

                # start all three
                t_dm0.start()
                t_comp.start()
                t_dm1.start()

                # wait for all to finish
                t_dm0.join()
                t_comp.join()
                t_dm1.join()

                # check if any failed
                if thread_results:
                    for name, e, tb_str in thread_results:
                        # print a user-readable header
                        print(f"\n❌ {name} failed on core {core}")
                        print(f"   error type   : {type(e).__name__}")
                        print(f"   error message: {e}")
                        print("   traceback:")
                        print(tb_str)
                        print("-" * 50)

                        # add to final aggregation (short)
                        errors.append(f"{name} on core {core}: {type(e).__name__}: {e}")

            if errors:
                raise RuntimeError("One or more cores failed:\n" + "\n".join(errors))

            return None

    return ProgramImpl(*funcs)
