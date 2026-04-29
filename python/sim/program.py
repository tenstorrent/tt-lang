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
import warnings
from typing import Any, Dict, List

from greenlet import getcurrent

from .dfb import DataflowBuffer
from .typedefs import BindableTemplate, Shape
from .blockstate import ThreadType
from .context import get_context
from .greenlet_scheduler import GreenletScheduler, set_scheduler
from .ttnnsim import Tensor
from .debug_print import ttlang_print
from .trace import trace


def set_max_dfbs(limit: int) -> None:
    """Set the maximum number of DataflowBuffers per core.

    Args:
        limit: Maximum number of CBs per core (must be non-negative)

    Raises:
        ValueError: If limit is negative

    Example:
        set_max_dfbs(64)  # Allow up to 64 CBs per core
    """
    if limit < 0:
        raise ValueError(f"max_dfbs must be non-negative, got {limit}")
    get_context().config.max_dfbs = limit


def get_max_dfbs() -> int:
    """Get the current maximum number of DataflowBuffers per core.

    Returns:
        Current CB limit per core
    """
    return get_context().config.max_dfbs


def set_max_l1_bytes(limit: int) -> None:
    """Set the maximum L1 memory per core (in bytes).

    The L1 memory used by a core is the sum of capacity_bytes across all of its
    DataflowBuffers. Kernel execution issues a warning if the total CB capacity
    on any core exceeds this limit. Defaults to 1336 KiB (Blackhole/Wormhole
    L1 size minus reserved program space).

    Args:
        limit: Maximum L1 bytes per core (must be positive)

    Raises:
        ValueError: If limit is not positive

    Example:
        set_max_l1_bytes(1_572_864)  # 1.5 MB
    """
    if limit <= 0:
        raise ValueError(f"max_l1_bytes must be positive, got {limit}")
    get_context().config.max_l1_bytes = limit


def get_max_l1_bytes() -> int:
    """Get the current L1 memory limit per core in bytes.

    Returns:
        Current L1 limit in bytes
    """
    return get_context().config.max_l1_bytes


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
            """Build per-core context with fresh DataflowBuffers and deep-copied state.

            Args:
                core: Core number to build context for

            Returns:
                Dictionary containing per-core context with fresh DataflowBuffers
            """
            memo: Dict[int, Any] = {}
            core_context: Dict[str, Any] = {}

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
                        setattr(value, "_name", key)
                        core_context[key] = value
                        memo[id(value)] = value
                    case DataflowBuffer():
                        # Create a fresh DFB for this core.
                        new_dfb = DataflowBuffer(
                            likeness_tensor=value.likeness_tensor,
                            shape=value.shape,
                            block_count=value.block_count,
                        )
                        setattr(new_dfb, "_name", key)
                        core_context[key] = new_dfb
                    case _:
                        core_context[key] = copy.deepcopy(value, memo)

            core_context["_core"] = core
            core_context["grid"] = self.context.get("grid", (1, 1))

            # Inject custom print function for debug printing
            core_context["print"] = ttlang_print

            return core_context

        def _run_cooperative(
            self,
            total_cores: int,
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> None:
            """Cooperative scheduling execution mode using greenlets."""

            # Warn if the number of DataflowBuffers exceeds the hardware limit.
            dfb_count = get_context().kernel_dfb_count
            max_dfbs = get_max_dfbs()
            if dfb_count > max_dfbs:
                warnings.warn(
                    f"Kernel defines {dfb_count} dataflow buffers, "
                    f"but the hardware limit is {max_dfbs}. "
                    f"Reduce the number of ttl.make_dataflow_buffer_like() calls.",
                    stacklevel=2,
                )

            # Warn if total L1 capacity exceeds the configured limit.
            total_l1_bytes = get_context().kernel_l1_bytes
            max_l1 = get_max_l1_bytes()
            if total_l1_bytes > max_l1:
                warnings.warn(
                    f"Total DataflowBuffer capacity per core ({total_l1_bytes} bytes) "
                    f"exceeds the L1 memory limit of {max_l1} bytes.",
                    stacklevel=2,
                )

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
                    for tmpl in [compute_func_tmpl, dm0_tmpl, dm1_tmpl]:
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

                        # Wrap to tag the greenlet with its linear core index so
                        # locality analysis in copy.py can read it via getcurrent().
                        def _tagged(fn=bound_func, c=core):
                            getcurrent()._sim_core = c  # type: ignore[attr-defined]
                            fn()

                        # Add to scheduler
                        thread_name = f"core{core}-{tmpl.__name__}"
                        scheduler.add_thread(thread_name, _tagged, thread_type)

                # Emit operation_start for each node before the scheduler runs.
                for core in range(total_cores):
                    trace("operation_start", node=core)

                # Run scheduler
                scheduler.run()

                # Emit operation_end for each node now that all kernels completed.
                for core in range(total_cores):
                    trace("operation_end", node=core)

                # Validate all DataflowBuffers have no pending blocks
                self._validate_dataflow_buffers(all_core_contexts)
            finally:
                # Clear scheduler
                set_scheduler(None)

        def _validate_dataflow_buffers(
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
