# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Greenlet-based cooperative scheduler for multi-node simulation.

This module provides a cooperative scheduler using greenlets instead of
yield transformations. Each compute or datamovement kernel runs in its own greenlet,
and blocking operations (wait/reserve) switch back to the scheduler.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from greenlet import greenlet

from .blockstate import KernelType
from .context import get_context, set_current_kernel_type, clear_current_kernel_type
from .diagnostics import (
    print_diagnostic_error,
    find_user_code_location,
    is_simulator_frame,
    format_node_ranges,
)
from .trace import get_dfb_name, trace


@dataclass(frozen=True)
class KernelId:
    """Stable identity for a cooperative scheduled kernel (scheduler dict key).

    Identity is ``(linear_node, kind, func_name)``: the linear node index, the
    kernel role (compute or data movement), and the decorated function's
    ``__name__``. The function name is part of identity because a node can host
    up to two data movement kernels; their ``__name__`` distinguishes them. The
    scheduler enforces uniqueness on registration -- two kernels with the same
    triple on the same node is rejected with a user-facing error.
    """

    linear_node: int
    kind: KernelType
    func_name: str

    def __post_init__(self) -> None:
        if self.linear_node < 0:
            raise ValueError(
                f"linear_node must be non-negative; got {self.linear_node!r}"
            )
        if not self.func_name:
            raise ValueError("func_name must be a non-empty string")


def kernel_display_name(kernel_id: KernelId) -> str:
    """Return the user-facing kernel label (``node0-mm_compute`` style)."""
    return f"node{kernel_id.linear_node}-{kernel_id.func_name}"


def set_scheduler_algorithm(algorithm: str) -> None:
    """Set the scheduling algorithm.

    Args:
        algorithm: Either 'greedy' or 'fair'
    """
    if algorithm not in ("greedy", "fair"):
        raise ValueError(f"Invalid scheduler algorithm: {algorithm}")
    get_context().config.scheduler_algorithm = algorithm


def get_scheduler_algorithm() -> str:
    """Get the current scheduling algorithm."""
    return get_context().config.scheduler_algorithm


class GreenletScheduler:
    """
    Cooperative scheduler using greenlets for per-node kernel execution.

    The scheduler maintains a collection of greenlets (one per registered kernel)
    and runs them in round-robin fashion. When a kernel blocks (e.g., on wait/reserve),
    it switches back to the scheduler, which tries other kernels.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        # Active greenlets: kernel_id -> (greenlet, blocking_obj, operation, kernel_type, block_location, raw_loc)
        # raw_loc is Optional[Tuple[str, int]] = (filename, lineno) for pretty-printing
        self._active: Dict[
            KernelId,
            Tuple[greenlet, Any, str, KernelType, str, Optional[Tuple[str, int]]],
        ] = {}
        # Completed greenlets (internal bookkeeping)
        self._completed: List[KernelId] = []
        # Main greenlet for the scheduler
        self._main_greenlet: Optional[greenlet] = None
        # Currently executing scheduled kernel
        self._current_kernel_id: Optional[KernelId] = None
        # Last run timestamp for fair scheduling (kernel_id -> timestamp)
        self._last_run: Dict[KernelId, int] = {}
        # Global timestamp counter
        self._timestamp: int = 0
        # Track if kernel has ever made progress (passed at least one block_if_needed check)
        self._has_made_progress: Dict[KernelId, bool] = {}

    def add_kernel(
        self,
        kernel_id: KernelId,
        func: Callable[[], None],
    ) -> None:
        """Add a scheduled kernel (greenlet) to the scheduler.

        Args:
            kernel_id: Stable kernel identity. Its ``kind`` field doubles as the
                kernel role (COMPUTE or DM); two kernels with the same
                ``(linear_node, kind, func_name)`` triple is rejected.
            func: Kernel entry function to execute.

        Raises:
            RuntimeError: If a kernel with this identity is already registered.
                Most commonly fired when two data movement kernels on the same
                node have the same ``__name__``; rename one of them.
        """
        if kernel_id in self._active:
            label = kernel_display_name(kernel_id)
            raise RuntimeError(
                f"Duplicate kernel registration: {label!r} "
                f"({kernel_id.kind.name}) is already scheduled on "
                f"node{kernel_id.linear_node}. Two {kernel_id.kind.name} kernels "
                f"on the same node must have distinct function names; rename "
                f"one of them."
            )

        # Create greenlet that wraps the function
        def wrapped_func() -> None:
            trace("kernel_start")
            func()
            trace("kernel_end")
            # Kernel completed successfully
            self._mark_completed(kernel_id)

        g = greenlet(wrapped_func)
        # Initially not blocked (will start when scheduled)
        self._active[kernel_id] = (g, None, "", kernel_id.kind, "", None)
        # Initialize last run time to 0 (never run)
        self._last_run[kernel_id] = 0
        # Kernel hasn't made progress yet
        self._has_made_progress[kernel_id] = False

    def block_current_kernel(self, blocking_obj: Any, operation: str) -> None:
        """Block the current scheduled kernel on an operation.

        This is called by wait()/reserve() operations to yield control back
        to the scheduler.

        Args:
            blocking_obj: Object being waited on (DataflowBuffer or CopyTransaction)
            operation: Operation name ("wait" or "reserve")
        """
        if self._current_kernel_id is None:
            raise RuntimeError(
                "block_current_kernel called outside of scheduler context "
                "(no kernel is currently scheduled)"
            )

        # Capture location where blocking occurred
        filename, lineno = find_user_code_location()
        location_str = f" at {filename}:{lineno}"
        raw_loc: Optional[Tuple[str, int]] = (filename, lineno)

        # Update active entry with blocking info and location
        g, _, _, kernel_type, _, _ = self._active[self._current_kernel_id]
        self._active[self._current_kernel_id] = (
            g,
            blocking_obj,
            operation,
            kernel_type,
            location_str,
            raw_loc,
        )

        # Switch back to scheduler
        if self._main_greenlet is None:
            raise RuntimeError("Main greenlet not set")

        trace("kernel_block", op=operation, on=get_dfb_name(blocking_obj))
        self._main_greenlet.switch()
        trace("kernel_unblock")

    def _mark_completed(self, kernel_id: KernelId) -> None:
        """Mark a kernel as completed and remove from active set.

        Args:
            kernel_id: Kernel identity
        """
        if kernel_id in self._active:
            del self._active[kernel_id]
        self._completed.append(kernel_id)
        # Clean up last run time
        if kernel_id in self._last_run:
            del self._last_run[kernel_id]

    def mark_kernel_progress(self) -> None:
        """Mark that the current scheduled kernel has made progress.

        This is called by block_if_needed when a kernel successfully proceeds
        past a blocking check without actually blocking.

        Raises:
            RuntimeError: If no kernel is scheduled or the name is missing from progress tracking
        """
        if self._current_kernel_id is None:
            raise RuntimeError(
                "mark_kernel_progress called but no kernel is currently scheduled. "
                "This indicates a bug in the scheduler."
            )
        if self._current_kernel_id not in self._has_made_progress:
            label = kernel_display_name(self._current_kernel_id)
            raise RuntimeError(
                f"Kernel {label!r} not found in progress tracking. "
                "This indicates a bug in the scheduler."
            )
        self._has_made_progress[self._current_kernel_id] = True

    def get_current_kernel_id(self) -> Optional[KernelId]:
        """Return the identity of the currently executing kernel, if any."""
        return self._current_kernel_id

    def get_current_kernel_name(self) -> Optional[str]:
        """Get the display name of the currently executing kernel.

        Returns:
            Kernel display name (e.g., ``node0-mm_reader``), or None if none
            is executing.
        """
        if self._current_kernel_id is None:
            return None
        return kernel_display_name(self._current_kernel_id)

    @property
    def tick(self) -> int:
        """Current logical tick (number of scheduler activations elapsed)."""
        return self._timestamp

    def _format_and_raise_kernel_error(
        self,
        name: str,
        exception: Exception,
        include_traceback: bool = False,
    ) -> None:
        """Format kernel runtime error with source location and re-raise.

        Args:
            name: Scheduled kernel name (e.g., node0-compute)
            exception: The exception that was raised
            include_traceback: Whether to include full traceback in fallback

        Raises:
            RuntimeError: Always raises with formatted error message
        """
        # Extract source location from exception traceback
        import traceback

        tb = traceback.extract_tb(exception.__traceback__)
        source_file = None
        source_line = None
        source_col = None

        for frame in tb:
            # Skip simulator internal frames
            if not is_simulator_frame(frame.filename):
                source_file = frame.filename
                source_line = frame.lineno
                source_col = getattr(frame, "colno", None) or 1
                break

        # Assert we found user code in traceback
        assert source_file is not None and source_line is not None, (
            f"No user code found in exception traceback for {name}. "
            "This indicates a bug in the scheduler or test setup."
        )

        # Print error with diagnostic formatting
        print_diagnostic_error(
            name,
            f"{type(exception).__name__}: {exception}",
            source_file,
            source_line,
            source_col or 1,
        )

        # Re-raise with kernel name included
        error_msg = f"{name}: {type(exception).__name__}: {exception}"
        raise RuntimeError(error_msg) from exception

    def _initialization_phase(self) -> None:
        """Run all kernels sequentially until they first block.

        This initialization ensures all kernels have blocking_obj set,
        so can_{operation}() checks work correctly in the fair scheduler.

        Timestamps are only given to kernels that made progress (passed at least
        one block_if_needed check). Kernels that blocked on their first check
        keep ts=0, giving them priority in fair scheduling.
        """

        for kernel_id in list(self._active.keys()):
            g, blocking_obj, _, kernel_type, _, _ = self._active[kernel_id]

            # All kernels should start unblocked in init phase
            if blocking_obj is not None:
                label = kernel_display_name(kernel_id)
                raise RuntimeError(
                    f"Kernel {label!r} is already blocked at init phase start. "
                    "This indicates a bug in the scheduler."
                )

            # Set current kernel context
            self._current_kernel_id = kernel_id
            set_current_kernel_type(kernel_type)

            try:
                # Run kernel until it blocks or completes
                g.switch()

                # Update timestamp only if kernel made progress
                made_progress = self._has_made_progress.get(kernel_id, False)

                if g.dead:
                    self._mark_completed(kernel_id)
                elif made_progress:
                    # Kernel passed one or more block_if_needed checks - give it a timestamp
                    self._timestamp += 1
                    self._last_run[kernel_id] = self._timestamp
                # Kernels that blocked on their first check keep ts=0

            except Exception as e:
                # Kernel raised an error during initialization
                clear_current_kernel_type()
                self._current_kernel_id = None

                # Format and raise error with source location
                self._format_and_raise_kernel_error(kernel_display_name(kernel_id), e)

            clear_current_kernel_type()

        self._current_kernel_id = None

    def _get_fair_kernel_order(self) -> List[KernelId]:
        """Get kernels sorted by least recently run.

        Kernels that can potentially make progress (not blocked or can unblock)
        are sorted by their last run timestamp in ascending order.

        Returns:
            List of kernel ids in least-recently-run order
        """
        # Get all active kernels with their last run times
        kernel_times: List[Tuple[int, KernelId]] = []
        for kernel_id in self._active.keys():
            last_run = self._last_run.get(kernel_id, 0)
            kernel_times.append((last_run, kernel_id))

        # Sort by timestamp (ascending), then by node, kind, and name for stability
        kernel_times.sort(
            key=lambda x: (
                x[0],
                x[1].linear_node,
                x[1].kind.value,
                x[1].func_name,
            )
        )

        return [tid for _, tid in kernel_times]

    def run(self) -> None:
        """Run all kernels until completion or deadlock is detected."""
        # Store main greenlet for switching back from kernels
        self._main_greenlet = greenlet.getcurrent()

        # Determine scheduling algorithm
        algorithm = get_scheduler_algorithm()

        # Phase 1: Initialization - run all kernels until they first block
        # This ensures all kernels have blocking_obj set so can_{operation}() checks work
        if algorithm == "fair":
            self._initialization_phase()

        # Phase 2: Main scheduling loop with fairness
        # Run all kernels until completion or deadlock
        while self._active:
            any_progress = False

            # Select kernels to try based on algorithm
            if algorithm == "fair":
                # Fair: Try kernels in order of least recently run
                kernel_candidates = self._get_fair_kernel_order()
            else:
                # Greedy: Try kernels in arbitrary order (as they appear in dict)
                kernel_candidates = list(self._active.keys())

            # Try to advance each kernel in the selected order
            for kernel_id in kernel_candidates:
                if kernel_id not in self._active:
                    # Kernel may have completed during this iteration
                    continue

                g, blocking_obj, blocked_op, kernel_type, location, _ = self._active[
                    kernel_id
                ]

                # If kernel is blocked, check if it can proceed
                if blocking_obj is not None:
                    can_method = getattr(blocking_obj, f"can_{blocked_op}", None)
                    if can_method is None or not can_method():
                        # Still blocked
                        continue

                    # Unblocked! Clear blocking state
                    self._active[kernel_id] = (g, None, "", kernel_type, "", None)

                # Set current kernel for block_current_kernel()
                self._current_kernel_id = kernel_id

                # Run kernel until it blocks or completes

                set_current_kernel_type(kernel_type)
                try:
                    if g.dead:
                        # Kernel already completed (marked by wrapped_func)
                        if kernel_id in self._active:
                            del self._active[kernel_id]
                        continue

                    # Switch to the greenlet
                    g.switch()
                    any_progress = True

                    # Always update timestamp after kernel runs
                    # The pre-check already prevented kernels that can't make progress from running
                    self._timestamp += 1
                    self._last_run[kernel_id] = self._timestamp

                    # If greenlet is dead, it completed
                    if g.dead and kernel_id in self._active:
                        # Should have been marked by wrapped_func, but double-check
                        self._mark_completed(kernel_id)
                except Exception as e:
                    # Kernel raised an error - preserve traceback for debugging
                    clear_current_kernel_type()
                    self._current_kernel_id = None

                    # Format and raise error with source location
                    # Include full traceback for main loop errors (more debugging info)
                    self._format_and_raise_kernel_error(
                        kernel_display_name(kernel_id),
                        e,
                        include_traceback=True,
                    )
                finally:
                    clear_current_kernel_type()

                self._current_kernel_id = None

            # Deadlock detection
            if not any_progress and self._active:
                # Group kernels by (operation, object, location)
                from collections import defaultdict

                blocked_groups: dict[tuple[str, str, str], list[str]] = defaultdict(
                    list
                )
                # Track raw (filename, lineno) per group for pretty printing
                blocked_raw_locs: dict[
                    tuple[str, str, str], Optional[Tuple[str, int]]
                ] = {}

                for kernel_id, (
                    g,
                    blocking_obj,
                    op,
                    _,
                    location,
                    raw_loc,
                ) in self._active.items():
                    obj_desc = self._get_obj_description(blocking_obj)
                    key = (op, obj_desc, location)
                    node_id = f"node{kernel_id.linear_node}"
                    blocked_groups[key].append(node_id)
                    if key not in blocked_raw_locs:
                        blocked_raw_locs[key] = raw_loc

                # Format and print grouped messages with pretty source context
                print("\nDeadlock detected: all generators blocked")
                for (op, obj_desc, location), node_ids in blocked_groups.items():
                    # Remove duplicates and sort for consistent output
                    unique_nodes = sorted(set(node_ids), key=lambda x: (len(x), x))

                    if len(unique_nodes) == 1:
                        nodes_label = unique_nodes[0]
                    else:
                        node_numbers: list[int] = [int(n[4:]) for n in unique_nodes]
                        nodes_label = f"nodes: {format_node_ranges(node_numbers)}"

                    raw_loc = blocked_raw_locs.get((op, obj_desc, location))
                    if raw_loc:
                        filename, lineno = raw_loc
                        print_diagnostic_error(
                            "deadlock",
                            f"blocked on {op}(){obj_desc} ({nodes_label})",
                            filename,
                            lineno,
                            1,
                        )
                    else:
                        print(
                            f"  blocked on {op}(){obj_desc}{location} ({nodes_label})"
                        )

                raise RuntimeError(
                    "Deadlock detected: all generators blocked"
                ) from RuntimeError("deadlock")

    def _get_obj_description(self, obj: Any) -> str:
        """Get a brief description of an object for debugging output."""
        if obj is None:
            return ""

        class_name = type(obj).__name__
        match class_name:
            case "Block":
                return " on Block"
            case "DataflowBuffer":
                name = getattr(obj, "_name", None)
                return f" on DataflowBuffer({name})" if name else " on DataflowBuffer"
            case "Pipe":
                src = getattr(obj, "src", "?")
                dst = getattr(obj, "dst", "?")
                return f" on Pipe({src}->{dst})"
            case "Tensor":
                return " on Tensor"
            case _:
                return f" on {class_name}"


def get_scheduler() -> GreenletScheduler:
    """Get the current scheduler instance.

    Returns:
        Current scheduler instance

    Raises:
        RuntimeError: If no scheduler is active
    """
    scheduler = get_context().scheduler
    if scheduler is None:
        raise RuntimeError(
            "No active scheduler. This should only be called from within a kernel."
        )
    return scheduler


def set_scheduler(scheduler: Optional[GreenletScheduler]) -> None:
    """Set the current scheduler instance."""
    get_context().scheduler = scheduler


def get_current_node_id() -> str:
    """Return the current node label for simulator-internal diagnostics.

    Not part of the public ``ttl`` API. Used by simulator modules (e.g. math
    warnings, debug print) to attribute messages to a node.

    Returns:
        Node ID like "node0".

    Raises:
        RuntimeError: If there is no active scheduler, or no kernel is currently
            scheduled. That indicates a simulator bug, not user misuse.
    """
    scheduler = get_scheduler()
    tid = scheduler.get_current_kernel_id()
    if tid is None:
        raise RuntimeError(
            "get_current_node_id() called with no kernel "
            "currently scheduled. Please report this as a bug."
        )
    return f"node{tid.linear_node}"


def block_if_needed(obj: Any, operation: str) -> None:
    """Block the current kernel if the operation cannot proceed, or yield for fair scheduling.

    For greedy scheduler:
    - Only blocks if the operation cannot proceed (can_{operation}() returns False)

    For fair scheduler:
    - Always yields at synchronization points to give other kernels a chance
    - Checks if operation can proceed and blocks if it can't
    - If it can proceed, yields anyway but will resume immediately when scheduled

    Args:
        obj: Object with can_{operation}() method to check
        operation: Operation name (e.g., "wait", "reserve")
    """
    can_method = getattr(obj, f"can_{operation}")
    scheduler = get_scheduler()
    algorithm = get_scheduler_algorithm()

    if algorithm == "fair":
        # Fair scheduler: always yield at synchronization points
        scheduler.mark_kernel_progress()
        # Always yield to give other kernels a chance
        scheduler.block_current_kernel(obj, operation)
        # When we resume, check again if we can proceed (in case state changed)
        if not can_method():
            scheduler.block_current_kernel(obj, operation)
    else:
        # Greedy scheduler: only block if we can't proceed
        if not can_method():
            scheduler.block_current_kernel(obj, operation)
        else:
            scheduler.mark_kernel_progress()
