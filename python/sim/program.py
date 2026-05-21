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
from typing import Any, Callable, Dict, List

from greenlet import getcurrent

from .dfb import DataflowBuffer
from .typedefs import BindableTemplate, Shape
from .blockstate import ThreadType
from .context import get_context
from .greenlet_scheduler import (
    GreenletScheduler,
    KernelId,
    set_scheduler,
)
from .ttnnsim import Tensor
from .analysis import (
    collect_reachable_analyses,
    install_copy_wait_hooks,
    PatternViolation,
    ThreadAnalysis,
)
from .diagnostics import print_diagnostic_error
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


def Program(*funcs: BindableTemplate, grid: Shape, pipenets: Any = None) -> Any:
    """Program class that combines compute and data movement functions.

    Args:
        *funcs: Compute and data movement function templates
        grid: Grid size tuple
        pipenets: Optional OperationPipeNets used to compute the active
            set of nodes. When None, every node participates.
    """

    class ProgramImpl:
        def __init__(
            self,
            *functions: BindableTemplate,
        ):
            self.functions = functions
            self.context: Dict[str, Any] = {"grid": grid}
            self.pipenets = pipenets

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

            # Run in cooperative mode.
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
                # Skip module objects (e.g., local imports like `from ttl.sim import ttnn`)
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
                    f"exceeds the L1 memory limit of {max_l1} bytes. "
                    f"Memory is accounted using declared dtypes, so this reflects "
                    f"the on-hardware footprint of the kernel.",
                    stacklevel=2,
                )

            # Create scheduler
            scheduler = GreenletScheduler()
            set_scheduler(scheduler)

            # Analyse all three thread functions (and any reachable helpers)
            # once before iterating over cores.  A shared visited set prevents
            # duplicate analysis when helpers are called by more than one thread.
            ctx = get_context()
            _empty = ThreadAnalysis(injection_points=(), bare_copy_linenos=frozenset())
            _visited: set[int] = set()
            injection_map: dict[types.CodeType, ThreadAnalysis] = {}
            all_violations: List[PatternViolation] = []

            for tmpl in [compute_func_tmpl, dm0_tmpl, dm1_tmpl]:
                analyses = collect_reachable_analyses(tmpl.__wrapped__, _visited)
                injection_map.update(analyses)
                top = analyses.get(tmpl.__wrapped__.__code__, _empty)
                ctx.injection_points_cache[tmpl.__wrapped__] = top
                all_violations.extend(top.violations)

            # Report any unsupported copy patterns before running the kernel.
            # All violations are collected first so the user sees every problem.
            if all_violations:
                for v in all_violations:
                    print_diagnostic_error(
                        v.func_name,
                        v.message,
                        v.source_file,
                        v.lineno,
                        v.col,
                    )
                n = len(all_violations)
                raise RuntimeError(
                    f"Found {n} unsupported pattern{'s' if n > 1 else ''} in thread "
                    "function(s). See errors above for details."
                )

            # Compute the PipeNet active set: linear node indices that
            # participate in any pipe as source or destination. Inactive nodes
            # skip every kernel thread, mirroring the compiler's scf.if guard.
            grid = self.context.get("grid", (1, 1))
            active_nodes = (
                self.pipenets.active_node_set(tuple(grid))
                if self.pipenets is not None
                else None
            )

            def _is_active(node: int) -> bool:
                return active_nodes is None or node in active_nodes

            try:
                # Track all per-core contexts for validation
                all_core_contexts: List[Dict[str, Any]] = []

                for core in range(total_cores):
                    # Skip cores that are not in any PipeNet's active set.
                    if not _is_active(core):
                        continue

                    # Build per-core context
                    core_context = self._build_core_context(core)
                    all_core_contexts.append(core_context)

                    # Add threads to scheduler (one compute + two DM per core).
                    # Identity is (core, kind, __name__); the two DM kernels on
                    # a core must have distinct __name__s -- the scheduler
                    # rejects duplicates with a user-facing error.
                    for tmpl in (compute_func_tmpl, dm0_tmpl, dm1_tmpl):
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
                        def _tagged(
                            fn: Callable[[], Any] = bound_func,
                            c: int = core,
                            core_ctx: Dict[str, Any] = core_context,
                        ) -> None:
                            getcurrent()._sim_core = c  # type: ignore[attr-defined]
                            fn()
                            # Auto-push/pop any blocks still pending when the thread
                            # function returns normally (final-iteration cleanup).
                            # This must not run during exception propagation, so it
                            # is placed after fn() rather than in a finally block.
                            for _val in core_ctx.values():
                                if isinstance(_val, DataflowBuffer):
                                    _val.auto_push_block()
                                    _val.auto_pop_block()

                        scheduler.add_thread(
                            KernelId(core, thread_type, tmpl.__name__),
                            _tagged,
                        )

                # Install injection hooks for all discovered code objects (thread
                # functions, nested defs, and module-scope helpers).
                install_copy_wait_hooks(injection_map)

                # Iterator over the cores that actually run threads.
                # Inactive cores (filtered above) are not traced.
                active_cores = [c for c in range(total_cores) if _is_active(c)]

                # Emit operation_start for each node before the scheduler runs.
                for core in active_cores:
                    trace("operation_start", node=core)

                # Run scheduler; if any thread raises, the exception propagates
                # immediately and the validation below is intentionally skipped.
                # Reporting a "simulator bug" for unpushed blocks only makes sense
                # when all threads completed normally (auto-push/pop should have fired).
                scheduler.run()

                # Emit operation_end for each node now that all kernels completed.
                for core in active_cores:
                    trace("operation_end", node=core)

                # Validate all DataflowBuffers have no pending blocks.
                # Only reached on normal exit from the scheduler.
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
