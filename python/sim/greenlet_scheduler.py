# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Greenlet-based cooperative scheduler for multi-core simulation.

This module provides a cooperative scheduler using greenlets instead of
yield transformations. Each thread (compute/DM) runs in its own greenlet,
and blocking operations (wait/reserve) switch back to the scheduler.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from greenlet import greenlet

from .blockstate import ThreadType
from .context import get_context, set_current_thread_type, clear_current_thread_type
from .diagnostics import (
    print_diagnostic_error,
    find_user_code_location,
    is_simulator_frame,
    format_core_ranges,
    extract_core_id_from_thread_name,
)
from .trace import get_dfb_name, trace


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
    Cooperative scheduler using greenlets for thread execution.

    The scheduler maintains a collection of greenlets (threads) and runs them
    in round-robin fashion. When a thread blocks (e.g., on wait/reserve),
    it switches back to the scheduler, which tries other threads.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        # Active greenlets: name -> (greenlet, blocking_obj, operation, thread_type, block_location, raw_loc)
        # raw_loc is Optional[Tuple[str, int]] = (filename, lineno) for pretty-printing
        self._active: Dict[
            str, Tuple[greenlet, Any, str, ThreadType, str, Optional[Tuple[str, int]]]
        ] = {}
        # Completed greenlets
        self._completed: List[str] = []
        # Main greenlet for the scheduler
        self._main_greenlet: Optional[greenlet] = None
        # Current greenlet being executed
        self._current_name: Optional[str] = None
        # Last run timestamp for fair scheduling (thread_name -> timestamp)
        self._last_run: Dict[str, int] = {}
        # Global timestamp counter
        self._timestamp: int = 0
        # Track if thread has ever made progress (passed at least one block_if_needed check)
        self._has_made_progress: Dict[str, bool] = {}

    def add_thread(
        self,
        name: str,
        func: Callable[[], None],
        thread_type: ThreadType,
    ) -> None:
        """Add a thread to the scheduler.

        Args:
            name: Thread identifier (e.g., "core0-compute")
            func: Thread function to execute
            thread_type: Thread type (COMPUTE or DM)
        """

        # Create greenlet that wraps the function
        def wrapped_func() -> None:
            trace("kernel_start")
            func()
            trace("kernel_end")
            # Thread completed successfully
            self._mark_completed(name)

        g = greenlet(wrapped_func)
        # Initially not blocked (will start when scheduled)
        self._active[name] = (g, None, "", thread_type, "", None)
        # Initialize last run time to 0 (never run)
        self._last_run[name] = 0
        # Thread hasn't made progress yet
        self._has_made_progress[name] = False

    def block_current_thread(self, blocking_obj: Any, operation: str) -> None:
        """Block the current thread on an operation.

        This is called by wait()/reserve() operations to yield control back
        to the scheduler.

        Args:
            blocking_obj: Object being waited on (DataflowBuffer or CopyTransaction)
            operation: Operation name ("wait" or "reserve")
        """
        if self._current_name is None:
            raise RuntimeError(
                "block_current_thread called outside of scheduler context"
            )

        # Capture location where blocking occurred
        filename, lineno = find_user_code_location()
        location_str = f" at {filename}:{lineno}"
        raw_loc: Optional[Tuple[str, int]] = (filename, lineno)

        # Update active entry with blocking info and location
        g, _, _, thread_type, _, _ = self._active[self._current_name]
        self._active[self._current_name] = (
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

    def _mark_completed(self, name: str) -> None:
        """Mark a thread as completed and remove from active set.

        Args:
            name: Thread identifier
        """
        if name in self._active:
            del self._active[name]
        self._completed.append(name)
        # Clean up last run time
        if name in self._last_run:
            del self._last_run[name]

    def mark_thread_progress(self) -> None:
        """Mark that the current thread has made progress.

        This is called by block_if_needed when a thread successfully proceeds
        past a blocking check without actually blocking.

        Raises:
            RuntimeError: If no current thread is set or thread not found
        """
        if self._current_name is None:
            raise RuntimeError(
                "mark_thread_progress called but no current thread is set. "
                "This indicates a bug in the scheduler."
            )
        if self._current_name not in self._has_made_progress:
            raise RuntimeError(
                f"Thread {self._current_name} not found in progress tracking. "
                "This indicates a bug in the scheduler."
            )
        self._has_made_progress[self._current_name] = True

    def get_current_thread_name(self) -> Optional[str]:
        """Get the name of the currently executing thread.

        Returns:
            Current thread name, or None if no thread is executing
        """
        return self._current_name

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
        """Format thread error with source location and re-raise.

        Args:
            name: Thread name
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

        # Re-raise with thread name included
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

        for name in list(self._active.keys()):
            g, blocking_obj, _, thread_type, _, _ = self._active[name]

            # All threads should start unblocked in init phase
            if blocking_obj is not None:
                raise RuntimeError(
                    f"Thread {name} is already blocked at init phase start. "
                    "This indicates a bug in the scheduler."
                )

            # Set current thread context
            self._current_name = name
            set_current_thread_type(thread_type)

            try:
                # Run thread until it blocks or completes
                g.switch()

                # Update timestamp only if thread made progress
                made_progress = self._has_made_progress.get(name, False)

                if g.dead:
                    self._mark_completed(name)
                elif made_progress:
                    # Thread passed one or more block_if_needed checks - give it a timestamp
                    self._timestamp += 1
                    self._last_run[name] = self._timestamp
                # Threads that blocked on their first check keep ts=0

            except Exception as e:
                # Thread raised an error during initialization
                clear_current_thread_type()
                self._current_name = None

                # Format and raise error with source location
                self._format_and_raise_thread_error(name, e)

            clear_current_thread_type()

        self._current_name = None

    def _get_fair_thread_order(self) -> List[str]:
        """Get threads sorted by least recently run.

        Threads that can potentially make progress (not blocked or can unblock)
        are sorted by their last run timestamp in ascending order.

        Returns:
            List of thread names in least-recently-run order
        """
        # Get all active threads with their last run times
        thread_times: List[Tuple[int, str]] = []
        for name in self._active.keys():
            last_run = self._last_run.get(name, 0)
            thread_times.append((last_run, name))

        # Sort by timestamp (ascending), then by name for stability
        thread_times.sort(key=lambda x: (x[0], x[1]))

        # Return just the thread names
        return [name for _, name in thread_times]

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
            for name in thread_candidates:
                if name not in self._active:
                    # Thread may have completed during this iteration
                    continue

                g, blocking_obj, blocked_op, thread_type, location, _ = self._active[
                    name
                ]

                # If thread is blocked, check if it can proceed
                if blocking_obj is not None:
                    can_method = getattr(blocking_obj, f"can_{blocked_op}", None)
                    if can_method is None or not can_method():
                        # Still blocked
                        continue

                    # Unblocked! Clear blocking state
                    self._active[name] = (g, None, "", thread_type, "", None)

                # Set current thread for block_current_thread()
                self._current_name = name

                # Run thread until it blocks or completes

                set_current_thread_type(thread_type)
                try:
                    if g.dead:
                        # Thread already completed (marked by wrapped_func)
                        if name in self._active:
                            del self._active[name]
                        continue

                    # Switch to the greenlet
                    g.switch()
                    any_progress = True

                    # Always update timestamp after thread runs
                    # The pre-check already prevented threads that can't make progress from running
                    self._timestamp += 1
                    self._last_run[name] = self._timestamp

                    # If greenlet is dead, it completed
                    if g.dead and name in self._active:
                        # Should have been marked by wrapped_func, but double-check
                        self._mark_completed(name)
                except Exception as e:
                    # Thread raised an error - preserve traceback for debugging
                    clear_current_thread_type()
                    self._current_name = None

                    # Format and raise error with source location
                    # Include full traceback for main loop errors (more debugging info)
                    self._format_and_raise_thread_error(name, e, include_traceback=True)
                finally:
                    clear_current_thread_type()

                self._current_name = None

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

                for name, (
                    g,
                    blocking_obj,
                    op,
                    _,
                    location,
                    raw_loc,
                ) in self._active.items():
                    obj_desc = self._get_obj_description(blocking_obj)
                    key = (op, obj_desc, location)
                    # Extract core identifier from thread name
                    core_id = extract_core_id_from_thread_name(name)
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
    """Get the current core ID from the active thread.

    Returns:
        Core ID like "core0".

    Raises:
        RuntimeError: If called outside a running kernel (no active scheduler).
    """
    scheduler = get_scheduler()
    thread_name = scheduler.get_current_thread_name()
    return extract_core_id_from_thread_name(thread_name)


def block_if_needed(obj: Any, operation: str) -> None:
    """Block current thread if operation cannot proceed, or yield for fair scheduling.

    For greedy scheduler:
    - Only blocks if the operation cannot proceed (can_{operation}() returns False)

    For fair scheduler:
    - Always yields at synchronization points to give other threads a chance
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
            # Still can't proceed after resuming, block again
            scheduler.block_current_thread(obj, operation)
    else:
        # Greedy scheduler: only block if we can't proceed
        if not can_method():
            scheduler.block_current_thread(obj, operation)
        else:
            scheduler.mark_thread_progress()
