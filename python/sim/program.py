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
from typing import Any, Dict, List

from .dfb import DFBAPI, DataflowBuffer
from .decorators import BindableTemplate
from .blockstate import ThreadType
from .greenlet_scheduler import GreenletScheduler, set_scheduler
from .ttnnsim import Tensor
from .typedefs import Shape


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
            # This ensures variables like DFBs that were defined in the kernel function
            # are available for per-core copying
            for tmpl in self.functions:
                if hasattr(tmpl, "__wrapped__"):
                    func = getattr(tmpl, "__wrapped__")
                    if hasattr(func, "__code__") and hasattr(func, "__closure__"):
                        code = func.__code__
                        closure = func.__closure__
                        if code.co_freevars and closure:
                            for var_name, cell in zip(code.co_freevars, closure):
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
                Dictionary containing per-core context with fresh DataflowBuffers
            """
            memo: Dict[int, Any] = {}
            core_context: Dict[str, Any] = {}
            api = DFBAPI()  # new DFBAPI per core

            for key, value in self.context.items():
                # Skip module objects (e.g., local imports like `from python.sim import ttnn`)
                match value:
                    case types.ModuleType():
                        core_context[key] = value
                        continue
                    case _:
                        pass

                match value:
                    case Tensor():
                        core_context[key] = value
                        memo[id(value)] = value
                    case DataflowBuffer():
                        # create a fresh DFB for this core
                        new_dfb = DataflowBuffer(
                            element=value.element,
                            shape=value.shape,
                            buffer_factor=value.buffer_factor,
                            api=api,
                        )
                        # Store the variable name for debugging
                        setattr(new_dfb, "_name", key)
                        core_context[key] = new_dfb
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
                        match thread_type:
                            case ThreadType.COMPUTE | ThreadType.DM:
                                pass
                            case _:
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

                # Validate all DataflowBuffers have no pending blocks
                self._validate_circular_buffers(all_core_contexts)
            finally:
                # Clear scheduler
                set_scheduler(None)

        def _validate_circular_buffers(
            self, all_core_contexts: List[Dict[str, Any]]
        ) -> None:
            """Validate that all DataflowBuffers have no pending blocks at end of execution.

            Args:
                all_core_contexts: List of per-core contexts containing DataflowBuffers

            Raises:
                RuntimeError: If any DataflowBuffer has pending blocks
            """
            errors: List[str] = []
            for core_idx, core_context in enumerate(all_core_contexts):
                for key, value in core_context.items():
                    match value:
                        case DataflowBuffer():
                            try:
                                value.validate_no_pending_blocks()
                            except RuntimeError as e:
                                errors.append(f"core{core_idx}.{key}: {e}")
                        case _:
                            pass

            if errors:
                raise RuntimeError(
                    "Kernel execution completed with incomplete DataflowBuffer operations:\n"
                    + "\n".join(errors)
                )

    return ProgramImpl(*funcs)
