# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Program execution framework for multi-core simulation.

This module provides the core execution framework for running compute and data movement
functions across multiple cores with proper context binding and error handling.
"""

import copy
import inspect
import types
from types import CellType, FunctionType
from typing import Any, Callable, Dict, List, Optional, Protocol

from .block import ThreadType
from .cb import CircularBuffer
from .cbapi import CBAPI
from .greenlet_scheduler import GreenletScheduler, set_scheduler
from .ttnnsim import Tensor
from .typedefs import Shape


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


def Program(*funcs: BindableTemplate, grid: Shape) -> Any:
    """Program class that combines compute and data movement functions.

    Args:
        *funcs: Compute and data movement function templates
        grid: Grid size tuple
    """

    class ProgramImpl:
        def __init__(
            self,
            *functions: BindableTemplate,
        ):
            self.functions = functions
            self.context: Dict[str, Any] = {"grid": grid}

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                # Capture caller's locals for any remaining context variables
                # Don't reset context - grid was already set in __init__
                self.context.update(frame.f_back.f_locals)

            # Extract closure variables from thread functions and add to context
            # This ensures variables like CBs that were defined in the kernel function
            # are available for per-core copying
            for tmpl in self.functions:
                if hasattr(tmpl, "__wrapped__"):
                    func = tmpl.__wrapped__
                    if func.__code__.co_freevars and func.__closure__:
                        for var_name, cell in zip(
                            func.__code__.co_freevars, func.__closure__
                        ):
                            try:
                                # Only add if not already in context
                                if var_name not in self.context:
                                    self.context[var_name] = cell.cell_contents
                            except ValueError:
                                # Cell is empty (variable not yet bound)
                                pass

            grid = self.context.get("grid", (1, 1))
            # Calculate total cores for any dimension grid
            total_cores = 1
            for dim_size in grid:
                total_cores *= dim_size

            compute_func_tmpl, dm0_tmpl, dm1_tmpl = self.functions

            # Run in cooperative mode
            self._run_cooperative(total_cores, compute_func_tmpl, dm0_tmpl, dm1_tmpl)

        def _build_core_context(self, core: int) -> Dict[str, Any]:
            """Build per-core context with copied circular buffers and other state.

            Args:
                core: Core number to build context for

            Returns:
                Dictionary containing per-core context with fresh CircularBuffers
            """
            memo: Dict[int, Any] = {}
            core_context: Dict[str, Any] = {}
            api = CBAPI()  # new CBAPI per core

            for key, value in self.context.items():
                # Skip module objects (e.g., local imports like `from python.sim import ttnn`)
                if isinstance(value, types.ModuleType):
                    core_context[key] = value
                    continue

                match value:
                    case Tensor():
                        core_context[key] = value
                        memo[id(value)] = value
                    case CircularBuffer():
                        # create a fresh CB for this core
                        new_cb = CircularBuffer(
                            element=value.element,
                            shape=value.shape,
                            buffer_factor=value.buffer_factor,
                            api=api,
                        )
                        # Store the variable name for debugging
                        new_cb._name = key
                        core_context[key] = new_cb
                    case _:
                        core_context[key] = copy.deepcopy(value, memo)

            # also make the core number visible
            core_context["_core"] = core
            # Also inject grid into core context for grid_size() function
            core_context["grid"] = self.context.get("grid", (1, 1))

            return core_context

        def _run_cooperative(
            self,
            total_cores: int,
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> None:
            """Cooperative scheduling execution mode using greenlets."""

            # Create scheduler
            scheduler = GreenletScheduler()
            set_scheduler(scheduler)

            try:
                # Track all per-core contexts for validation
                all_core_contexts: List[Dict[str, Any]] = []

                for core in range(total_cores):
                    # Build per-core context
                    core_context = self._build_core_context(core)
                    all_core_contexts.append(core_context)

                    # Add threads to scheduler
                    for name, tmpl in [
                        ("compute", compute_func_tmpl),
                        ("dm0", dm0_tmpl),
                        ("dm1", dm1_tmpl),
                    ]:
                        # Get ThreadType directly from template's thread_type attribute
                        thread_type = getattr(tmpl, "thread_type", None)
                        if not isinstance(thread_type, ThreadType):
                            raise RuntimeError(
                                f"Template {tmpl} has invalid thread_type '{thread_type}'. "
                                f"Expected ThreadType enum (COMPUTE or DM)."
                            )

                        # Bind template to core context
                        bound_func = tmpl.bind(core_context)

                        # Add to scheduler
                        thread_name = f"core{core}-{name}"
                        scheduler.add_thread(thread_name, bound_func, thread_type)

                # Run scheduler
                scheduler.run()

                # Validate all CircularBuffers have no pending blocks
                self._validate_circular_buffers(all_core_contexts)
            finally:
                # Clear scheduler
                set_scheduler(None)

        def _validate_circular_buffers(
            self, all_core_contexts: List[Dict[str, Any]]
        ) -> None:
            """Validate that all CircularBuffers have no pending blocks at end of execution.

            Args:
                all_core_contexts: List of per-core contexts containing CircularBuffers

            Raises:
                RuntimeError: If any CircularBuffer has pending blocks
            """
            errors = []
            for core_idx, core_context in enumerate(all_core_contexts):
                for key, value in core_context.items():
                    if isinstance(value, CircularBuffer):
                        try:
                            value.validate_no_pending_blocks()
                        except RuntimeError as e:
                            errors.append(f"core{core_idx}.{key}: {e}")

            if errors:
                raise RuntimeError(
                    "Kernel execution completed with incomplete CircularBuffer operations:\n"
                    + "\n".join(errors)
                )

    return ProgramImpl(*funcs)
