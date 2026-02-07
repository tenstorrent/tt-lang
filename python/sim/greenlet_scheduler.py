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

from .block import ThreadType


def _get_ttlang_compile_error() -> Any:
    """Lazy import of TTLangCompileError to avoid circular dependency."""
    import importlib.util
    import sys
    from pathlib import Path

    # Direct import of diagnostics module without going through ttl package
    # This avoids importing the full compiler infrastructure
    diagnostics_path = Path(__file__).parent.parent / "ttl" / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("ttl.diagnostics", diagnostics_path)
    if spec and spec.loader:
        diagnostics = importlib.util.module_from_spec(spec)
        sys.modules["ttl.diagnostics"] = diagnostics
        spec.loader.exec_module(diagnostics)
        return diagnostics.TTLangCompileError
    raise ImportError("Could not load ttl.diagnostics")


class GreenletScheduler:
    """
    Cooperative scheduler using greenlets for thread execution.

    The scheduler maintains a collection of greenlets (threads) and runs them
    in round-robin fashion. When a thread blocks (e.g., on wait/reserve),
    it switches back to the scheduler, which tries other threads.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        # Active greenlets: name -> (greenlet, blocking_obj, operation, thread_type, block_location)
        self._active: Dict[str, Tuple[greenlet, Any, str, ThreadType, str]] = {}
        # Completed greenlets
        self._completed: List[str] = []
        # Main greenlet for the scheduler
        self._main_greenlet: Optional[greenlet] = None
        # Current greenlet being executed
        self._current_name: Optional[str] = None

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
            func()
            # Thread completed successfully
            self._mark_completed(name)

        g = greenlet(wrapped_func)
        # Initially not blocked (will start when scheduled)
        self._active[name] = (g, None, "", thread_type, "")

    def block_current_thread(self, blocking_obj: Any, operation: str) -> None:
        """Block the current thread on an operation.

        This is called by wait()/reserve() operations to yield control back
        to the scheduler.

        Args:
            blocking_obj: Object being waited on (CircularBuffer or CopyTransaction)
            operation: Operation name ("wait" or "reserve")
        """
        if self._current_name is None:
            raise RuntimeError(
                "block_current_thread called outside of scheduler context"
            )

        # Capture location where blocking occurred
        import inspect

        frame = inspect.currentframe()
        location_str = ""
        if frame and frame.f_back:
            # Walk up the call stack to find user code
            caller_frame = frame.f_back
            while caller_frame:
                filename = caller_frame.f_code.co_filename
                # Skip simulator internals
                if "/python/sim/" not in filename and "greenlet" not in filename:
                    lineno = caller_frame.f_lineno
                    location_str = f" at {filename}:{lineno}"
                    break
                caller_frame = caller_frame.f_back

        # Update active entry with blocking info and location
        g, _, _, thread_type, _ = self._active[self._current_name]
        self._active[self._current_name] = (
            g,
            blocking_obj,
            operation,
            thread_type,
            location_str,
        )

        # Switch back to scheduler
        if self._main_greenlet is None:
            raise RuntimeError("Main greenlet not set")
        self._main_greenlet.switch()

    def _mark_completed(self, name: str) -> None:
        """Mark a thread as completed and remove from active set.

        Args:
            name: Thread identifier
        """
        if name in self._active:
            del self._active[name]
        self._completed.append(name)

    def run(self) -> None:
        """Run all threads until completion or deadlock is detected."""
        # Store main greenlet for switching back from threads
        self._main_greenlet = greenlet.getcurrent()

        # Run all threads until completion or deadlock
        while self._active:
            any_progress = False

            # Try to advance each active thread
            for name in list(self._active.keys()):
                g, blocking_obj, blocked_op, thread_type, location = self._active[name]

                # If thread is blocked, check if it can proceed
                if blocking_obj is not None:
                    can_method = getattr(blocking_obj, f"can_{blocked_op}", None)
                    if can_method is None or not can_method():
                        # Still blocked
                        continue

                    # Unblocked! Clear blocking state
                    self._active[name] = (g, None, "", thread_type, "")

                # Set current thread for block_current_thread()
                self._current_name = name

                # Run thread until it blocks or completes
                from .block import _set_current_thread_type, _clear_current_thread_type

                _set_current_thread_type(thread_type)
                try:
                    if g.dead:
                        # Thread already completed (marked by wrapped_func)
                        if name in self._active:
                            del self._active[name]
                        continue

                    # Switch to the greenlet
                    g.switch()
                    any_progress = True

                    # If greenlet is dead, it completed
                    if g.dead and name in self._active:
                        # Should have been marked by wrapped_func, but double-check
                        self._mark_completed(name)
                except Exception as e:
                    # Thread raised an error - preserve traceback for debugging
                    _clear_current_thread_type()
                    self._current_name = None

                    # Format error with thread name and source location using pretty printing
                    import traceback

                    # Extract source location from traceback
                    # Look for the first frame that's in user code (not in python/sim)
                    tb = traceback.extract_tb(e.__traceback__)
                    source_file = None
                    source_line = None
                    source_col = None
                    for frame in tb:
                        # Skip internal greenlet/scheduler/simulator frames
                        if (
                            "greenlet_scheduler.py" not in frame.filename
                            and "greenlet" not in frame.filename
                            and "/python/sim/" not in frame.filename
                        ):
                            source_file = frame.filename
                            source_line = frame.lineno
                            source_col = getattr(frame, "colno", None) or 1
                            break

                    # Use TTLangCompileError for pretty formatting if we have source location
                    if source_file and source_line:
                        try:
                            TTLangCompileError = _get_ttlang_compile_error()
                            compile_error = TTLangCompileError(
                                f"{type(e).__name__}: {e}",
                                source_file=source_file,
                                line=source_line,
                                col=source_col,
                            )
                            print(f"\n❌ Error in {name}:")
                            print(compile_error.format())
                            print("-" * 50)
                            # Re-raise with thread name included for test compatibility
                            error_msg = f"{name}: {type(e).__name__}: {e}"
                            raise RuntimeError(error_msg) from e
                        except ImportError:
                            # Fallback if TTLangCompileError is not available
                            pass

                    # Fallback to basic formatting
                    print(f"\nError in {name}:")
                    if source_file and source_line:
                        print(f"  File: {source_file}:{source_line}")
                    print(f"  {type(e).__name__}: {e}")

                    # Also print full traceback for debugging
                    tb_str = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                    print(f"\nFull traceback:")
                    print(tb_str)

                    # Re-raise with original exception chained
                    error_msg = f"{name}: {type(e).__name__}: {e}"
                    raise RuntimeError(error_msg) from e
                finally:
                    _clear_current_thread_type()

                self._current_name = None

            # Deadlock detection
            if not any_progress and self._active:
                blocked_info: List[str] = []
                for name, (g, blocking_obj, op, _, location) in self._active.items():
                    obj_desc = self._get_obj_description(blocking_obj)
                    blocked_info.append(
                        f"  {name}: blocked on {op}(){obj_desc}{location}"
                    )

                raise RuntimeError(
                    f"Deadlock detected: all generators blocked\n"
                    + "\n".join(blocked_info)
                )

    def _get_obj_description(self, obj: Any) -> str:
        """Get a brief description of an object for debugging output."""
        if obj is None:
            return ""

        from .block import Block
        from .cb import CircularBuffer
        from .pipe import Pipe
        from .ttnnsim import Tensor

        match obj:
            case Block():
                return " on Block"
            case CircularBuffer() if hasattr(obj, "_name"):
                return f" on CircularBuffer({obj._name})"
            case CircularBuffer():
                return " on CircularBuffer"
            case Pipe():
                return f" on Pipe({obj.src}->{obj.dst})"
            case Tensor():
                return " on Tensor"
            case _:
                class_name = (
                    obj.__class__.__name__ if hasattr(obj, "__class__") else str(obj)
                )
                return f" on {class_name}"


# Global scheduler instance for the current execution
_current_scheduler: Optional[GreenletScheduler] = None


def get_scheduler() -> GreenletScheduler:
    """Get the current scheduler instance.

    Returns:
        Current scheduler instance

    Raises:
        RuntimeError: If no scheduler is active
    """
    if _current_scheduler is None:
        raise RuntimeError(
            "No active scheduler. This should only be called from within a kernel."
        )
    return _current_scheduler


def set_scheduler(scheduler: Optional[GreenletScheduler]) -> None:
    """Set the current scheduler instance."""
    global _current_scheduler
    _current_scheduler = scheduler


def block_if_needed(obj: Any, operation: str) -> None:
    """Block current thread if operation cannot proceed.

    Checks if the operation can proceed by calling obj.can_{operation}().
    If it returns False, blocks the current thread via the scheduler.

    Args:
        obj: Object with can_{operation}() method to check
        operation: Operation name (e.g., "wait", "reserve")
    """
    can_method = getattr(obj, f"can_{operation}")
    if not can_method():
        scheduler = get_scheduler()
        scheduler.block_current_thread(obj, operation)
