# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Greenlet-based cooperative scheduler for multi-core simulation.

This module provides a cooperative scheduler using greenlets instead of
yield transformations. Each compute or datamovement kernel runs in its own greenlet,
and blocking operations (wait/reserve) switch back to the scheduler.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from greenlet import greenlet

from .blockstate import ThreadType
from .context import get_context, set_current_thread_type, clear_current_thread_type
from .diagnostics import (
    print_diagnostic_error,
    find_user_code_location,
    is_simulator_frame,
    format_core_ranges,
)
from .trace import get_dfb_name, trace


@dataclass(frozen=True)
class KernelThreadId:
    """Stable identity for a cooperative scheduled kernel (scheduler dict key).

    ``linear_core`` is the linear core index used by ``program`` when registering
    threads. ``suffix`` is the kernel template name (e.g. ``compute``, ``dm0``).

    User-visible strings match the historical format ``core{linear_core}-{suffix}``;
    use :func:`kernel_thread_display_name` for traces and diagnostics.
    """

    linear_core: int
    suffix: str

    def __post_init__(self) -> None:
        if self.linear_core < 0:
            raise ValueError(
                f"linear_core must be non-negative; got {self.linear_core!r}"
            )
        if not self.suffix:
            raise ValueError("suffix must be a non-empty string")


def kernel_thread_display_name(thread_id: KernelThreadId) -> str:
    """Return the user-facing kernel label (``core0-compute`` style)."""
    return f"core{thread_id.linear_core}-{thread_id.suffix}"


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
    Cooperative scheduler using greenlets for per-core kernel execution.

    The scheduler maintains a collection of greenlets (one per registered kernel)
    and runs them in round-robin fashion. When a kernel blocks (e.g., on wait/reserve),
    it switches back to the scheduler, which tries other kernels.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        # Active greenlets: thread_id -> (greenlet, blocking_obj, operation, thread_type, block_location, raw_loc)
        # raw_loc is Optional[Tuple[str, int]] = (filename, lineno) for pretty-printing
        self._active: Dict[
            KernelThreadId,
            Tuple[greenlet, Any, str, ThreadType, str, Optional[Tuple[str, int]]],
        ] = {}
        # Completed greenlets (internal bookkeeping)
        self._completed: List[KernelThreadId] = []
        # Main greenlet for the scheduler
        self._main_greenlet: Optional[greenlet] = None
        # Currently executing scheduled kernel
        self._current_thread_id: Optional[KernelThreadId] = None
        # Last run timestamp for fair scheduling (thread_id -> timestamp)
        self._last_run: Dict[KernelThreadId, int] = {}
        # Global timestamp counter
        self._timestamp: int = 0
        # Track if thread has ever made progress (passed at least one block_if_needed check)
        self._has_made_progress: Dict[KernelThreadId, bool] = {}

    def add_thread(
        self,
        thread_id: KernelThreadId,
        func: Callable[[], None],
        thread_type: ThreadType,
    ) -> None:
        """Add a scheduled kernel (greenlet) to the scheduler.

        Args:
            thread_id: Stable kernel identity (linear core index and template suffix).
            func: Kernel entry function to execute
            thread_type: Kernel role (COMPUTE or DM)
        """

        # Create greenlet that wraps the function
        def wrapped_func() -> None:
            trace("kernel_start")
            func()
            trace("kernel_end")
            # Thread completed successfully
            self._mark_completed(thread_id)

        g = greenlet(wrapped_func)
        # Initially not blocked (will start when scheduled)
        self._active[thread_id] = (g, None, "", thread_type, "", None)
        # Initialize last run time to 0 (never run)
        self._last_run[thread_id] = 0
        # Thread hasn't made progress yet
        self._has_made_progress[thread_id] = False

    def block_current_thread(self, blocking_obj: Any, operation: str) -> None:
        """Block the current scheduled kernel on an operation.

        This is called by wait()/reserve() operations to yield control back
        to the scheduler.

        Args:
            blocking_obj: Object being waited on (DataflowBuffer or CopyTransaction)
            operation: Operation name ("wait" or "reserve")
        """
        if self._current_thread_id is None:
            raise RuntimeError(
                "block_current_thread called outside of scheduler context "
                "(no kernel is currently scheduled)"
            )

        # Capture location where blocking occurred
        filename, lineno = find_user_code_location()
        location_str = f" at {filename}:{lineno}"
        raw_loc: Optional[Tuple[str, int]] = (filename, lineno)

        # Update active entry with blocking info and location
        g, _, _, thread_type, _, _ = self._active[self._current_thread_id]
        self._active[self._current_thread_id] = (
            g,
            blocking_obj,
            operation,
            thread_type,
            location_str,
            raw_loc,
        )

        # Switch back to scheduler
        if self._main_greenlet is None:
            raise RuntimeError("Main greenlet not set")

        trace("kernel_block", op=operation, on=get_dfb_name(blocking_obj))
        self._main_greenlet.switch()
        trace("kernel_unblock")

    def _mark_completed(self, thread_id: KernelThreadId) -> None:
        """Mark a kernel as completed and remove from active set.

        Args:
            thread_id: Kernel identity
        """
        if thread_id in self._active:
            del self._active[thread_id]
        self._completed.append(thread_id)
        # Clean up last run time
        if thread_id in self._last_run:
            del self._last_run[thread_id]

    def mark_thread_progress(self) -> None:
        """Mark that the current scheduled kernel has made progress.

        This is called by block_if_needed when a kernel successfully proceeds
        past a blocking check without actually blocking.

        Raises:
            RuntimeError: If no kernel is scheduled or the name is missing from progress tracking
        """
        if self._current_thread_id is None:
            raise RuntimeError(
                "mark_thread_progress called but no kernel is currently scheduled. "
                "This indicates a bug in the scheduler."
            )
        if self._current_thread_id not in self._has_made_progress:
            label = kernel_thread_display_name(self._current_thread_id)
            raise RuntimeError(
                f"Kernel {label!r} not found in progress tracking. "
                "This indicates a bug in the scheduler."
            )
        self._has_made_progress[self._current_thread_id] = True

    def get_current_kernel_thread_id(self) -> Optional[KernelThreadId]:
        """Return the identity of the currently executing kernel, if any."""
        return self._current_thread_id

    def get_current_thread_name(self) -> Optional[str]:
        """Get the name of the currently executing scheduled kernel.

        Returns:
            Kernel name (e.g., core0-dm), or None if none is executing
        """
        if self._current_thread_id is None:
            return None
        return kernel_thread_display_name(self._current_thread_id)

    @property
    def tick(self) -> int:
        """Current logical tick (number of scheduler activations elapsed)."""
        return self._timestamp

    def _format_and_raise_thread_error(
        self,
        name: str,
        exception: Exception,
        include_traceback: bool = False,
    ) -> None:
        """Format kernel runtime error with source location and re-raise.

        Args:
            name: Scheduled kernel name (e.g., core0-compute)
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
        """Run all threads sequentially until they first block.

        This initialization ensures all threads have blocking_obj set,
        so can_{operation}() checks work correctly in the fair scheduler.

        Timestamps are only given to threads that made progress (passed at least
        one block_if_needed check). Threads that blocked on their first check
        keep ts=0, giving them priority in fair scheduling.
        """

        for thread_id in list(self._active.keys()):
            g, blocking_obj, _, thread_type, _, _ = self._active[thread_id]

            # All threads should start unblocked in init phase
            if blocking_obj is not None:
                label = kernel_thread_display_name(thread_id)
                raise RuntimeError(
                    f"Kernel {label!r} is already blocked at init phase start. "
                    "This indicates a bug in the scheduler."
                )

            # Set current thread context
            self._current_thread_id = thread_id
            set_current_thread_type(thread_type)

            try:
                # Run thread until it blocks or completes
                g.switch()

                # Update timestamp only if thread made progress
                made_progress = self._has_made_progress.get(thread_id, False)

                if g.dead:
                    self._mark_completed(thread_id)
                elif made_progress:
                    # Thread passed one or more block_if_needed checks - give it a timestamp
                    self._timestamp += 1
                    self._last_run[thread_id] = self._timestamp
                # Threads that blocked on their first check keep ts=0

            except Exception as e:
                # Thread raised an error during initialization
                clear_current_thread_type()
                self._current_thread_id = None

                # Format and raise error with source location
                self._format_and_raise_thread_error(
                    kernel_thread_display_name(thread_id), e
                )

            clear_current_thread_type()

        self._current_thread_id = None

    def _get_fair_thread_order(self) -> List[KernelThreadId]:
        """Get threads sorted by least recently run.

        Threads that can potentially make progress (not blocked or can unblock)
        are sorted by their last run timestamp in ascending order.

        Returns:
            List of thread ids in least-recently-run order
        """
        # Get all active threads with their last run times
        thread_times: List[Tuple[int, KernelThreadId]] = []
        for thread_id in self._active.keys():
            last_run = self._last_run.get(thread_id, 0)
            thread_times.append((last_run, thread_id))

        # Sort by timestamp (ascending), then by display name for stability
        thread_times.sort(key=lambda x: (x[0], kernel_thread_display_name(x[1])))

        return [tid for _, tid in thread_times]

    def run(self) -> None:
        """Run all threads until completion or deadlock is detected."""
        # Store main greenlet for switching back from threads
        self._main_greenlet = greenlet.getcurrent()

        # Determine scheduling algorithm
        algorithm = get_scheduler_algorithm()

        # Phase 1: Initialization - run all threads until they first block
        # This ensures all threads have blocking_obj set so can_{operation}() checks work
        if algorithm == "fair":
            self._initialization_phase()

        # Phase 2: Main scheduling loop with fairness
        # Run all threads until completion or deadlock
        while self._active:
            any_progress = False

            # Select threads to try based on algorithm
            if algorithm == "fair":
                # Fair: Try threads in order of least recently run
                thread_candidates = self._get_fair_thread_order()
            else:
                # Greedy: Try threads in arbitrary order (as they appear in dict)
                thread_candidates = list(self._active.keys())

            # Try to advance each thread in the selected order
            for thread_id in thread_candidates:
                if thread_id not in self._active:
                    # Thread may have completed during this iteration
                    continue

                g, blocking_obj, blocked_op, thread_type, location, _ = self._active[
                    thread_id
                ]

                # If thread is blocked, check if it can proceed
                if blocking_obj is not None:
                    can_method = getattr(blocking_obj, f"can_{blocked_op}", None)
                    if can_method is None or not can_method():
                        # Still blocked
                        continue

                    # Unblocked! Clear blocking state
                    self._active[thread_id] = (g, None, "", thread_type, "", None)

                # Set current thread for block_current_thread()
                self._current_thread_id = thread_id

                # Run thread until it blocks or completes

                set_current_thread_type(thread_type)
                try:
                    if g.dead:
                        # Thread already completed (marked by wrapped_func)
                        if thread_id in self._active:
                            del self._active[thread_id]
                        continue

                    # Switch to the greenlet
                    g.switch()
                    any_progress = True

                    # Always update timestamp after thread runs
                    # The pre-check already prevented threads that can't make progress from running
                    self._timestamp += 1
                    self._last_run[thread_id] = self._timestamp

                    # If greenlet is dead, it completed
                    if g.dead and thread_id in self._active:
                        # Should have been marked by wrapped_func, but double-check
                        self._mark_completed(thread_id)
                except Exception as e:
                    # Thread raised an error - preserve traceback for debugging
                    clear_current_thread_type()
                    self._current_thread_id = None

                    # Format and raise error with source location
                    # Include full traceback for main loop errors (more debugging info)
                    self._format_and_raise_thread_error(
                        kernel_thread_display_name(thread_id),
                        e,
                        include_traceback=True,
                    )
                finally:
                    clear_current_thread_type()

                self._current_thread_id = None

            # Deadlock detection
            if not any_progress and self._active:
                # Group threads by (operation, object, location)
                from collections import defaultdict

                blocked_groups: dict[tuple[str, str, str], list[str]] = defaultdict(
                    list
                )
                # Track raw (filename, lineno) per group for pretty printing
                blocked_raw_locs: dict[
                    tuple[str, str, str], Optional[Tuple[str, int]]
                ] = {}

                for thread_id, (
                    g,
                    blocking_obj,
                    op,
                    _,
                    location,
                    raw_loc,
                ) in self._active.items():
                    obj_desc = self._get_obj_description(blocking_obj)
                    key = (op, obj_desc, location)
                    core_id = f"core{thread_id.linear_core}"
                    blocked_groups[key].append(core_id)
                    if key not in blocked_raw_locs:
                        blocked_raw_locs[key] = raw_loc

                # Format and print grouped messages with pretty source context
                print("\nDeadlock detected: all generators blocked")
                for (op, obj_desc, location), core_ids in blocked_groups.items():
                    # Remove duplicates and sort for consistent output
                    unique_cores = sorted(set(core_ids), key=lambda x: (len(x), x))

                    if len(unique_cores) == 1:
                        cores_label = unique_cores[0]
                    else:
                        core_numbers: list[int] = [
                            int(core_id[4:]) for core_id in unique_cores
                        ]
                        cores_label = f"cores: {format_core_ranges(core_numbers)}"

                    raw_loc = blocked_raw_locs.get((op, obj_desc, location))
                    if raw_loc:
                        filename, lineno = raw_loc
                        print_diagnostic_error(
                            "deadlock",
                            f"blocked on {op}(){obj_desc} ({cores_label})",
                            filename,
                            lineno,
                            1,
                        )
                    else:
                        print(
                            f"  blocked on {op}(){obj_desc}{location} ({cores_label})"
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


def get_current_core_id() -> str:
    """Return the current core label for simulator-internal diagnostics.

    Not part of the public ``ttl`` API. Used by simulator modules (e.g. math
    warnings, debug print) to attribute messages to a core.

    Returns:
        Core ID like "core0".

    Raises:
        RuntimeError: If there is no active scheduler, or no kernel is currently
            scheduled. That indicates a simulator bug, not user misuse.
    """
    scheduler = get_scheduler()
    tid = scheduler.get_current_kernel_thread_id()
    if tid is None:
        raise RuntimeError(
            "get_current_core_id() called with no kernel "
            "currently scheduled. Please report this as a bug."
        )
    return f"core{tid.linear_core}"


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
        scheduler.mark_thread_progress()
        # Always yield to give other threads a chance
        scheduler.block_current_thread(obj, operation)
        # When we resume, check again if we can proceed (in case state changed)
        if not can_method():
            scheduler.block_current_thread(obj, operation)
    else:
        # Greedy scheduler: only block if we can't proceed
        if not can_method():
            scheduler.block_current_thread(obj, operation)
        else:
            scheduler.mark_thread_progress()
