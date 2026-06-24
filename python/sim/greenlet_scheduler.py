# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Greenlet-based cooperative scheduler for multi-node simulation.

This module provides a cooperative scheduler using greenlets instead of
yield transformations. Each compute or datamovement kernel runs in its own greenlet,
and blocking operations (wait/reserve) switch back to the scheduler.
"""

from dataclasses import dataclass, field
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
from .trace import TRACE, trace


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
    # Cached hash; populated in __post_init__ and returned by __hash__.  Declared
    # as a non-init, non-repr, non-compare, non-hash field so the dataclass
    # machinery ignores it for equality/representation but still recognises
    # the attribute exists on instances.
    _hash: int = field(default=0, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.linear_node < 0:
            raise ValueError(
                f"linear_node must be non-negative; got {self.linear_node!r}"
            )
        if not self.func_name:
            raise ValueError("func_name must be a non-empty string")
        # ``KernelId`` is the key type for the scheduler's ``_active`` /
        # ``_has_made_progress`` dicts and is hashed millions of times per
        # simulation run; the dataclass-generated ``__hash__`` reconstructs
        # and re-hashes the field tuple on every call.  Compute it once here
        # and serve it from ``__hash__``.  ``object.__setattr__`` is needed
        # because the dataclass is frozen.
        object.__setattr__(
            self, "_hash", hash((self.linear_node, self.kind, self.func_name))
        )

    def __hash__(self) -> int:
        return self._hash


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


class _KernelState:
    """Per-kernel scheduler entry.

    Splits the stable bits ((greenlet, kernel_type) -- set once at
    ``add_kernel()`` and never mutated) from the ephemeral bits
    (``blocking_obj``, ``operation`` -- updated on every block/unblock)
    so the hot ``block_current_kernel`` path can mutate slots in place
    rather than allocating a fresh 4-tuple and doing two dict operations
    on ``_active`` per call.

    At >16 M blocks per dry-run that was ~1 GB of tuple garbage and
    ~34 M dict hash/store operations on ``KernelId``.
    """

    __slots__ = ("g", "kernel_type", "blocking_obj", "operation")

    def __init__(self, g: greenlet, kernel_type: KernelType) -> None:
        self.g: greenlet = g
        self.kernel_type: KernelType = kernel_type
        self.blocking_obj: Any = None
        self.operation: str = ""


class GreenletScheduler:
    """
    Cooperative scheduler using greenlets for per-node kernel execution.

    The scheduler maintains a collection of greenlets (one per registered kernel)
    and runs them in round-robin fashion. When a kernel blocks (e.g., on wait/reserve),
    it switches back to the scheduler, which tries other kernels.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        # Per-kernel scheduling state: see ``_KernelState``.  The block location
        # is intentionally NOT stored here: capturing (filename, lineno) on
        # every block adds ~5-6s/run for step_1 worth of stack-walking and
        # string formatting, and the only consumer is the deadlock diagnostic
        # below, which fires at most once per run.  When a deadlock is detected
        # we recover the location lazily from each blocked greenlet's
        # ``gr_frame`` instead.
        # In ``fair`` scheduling mode, ``_active`` is *kept* in
        # least-recently-run-first order: each successful switch moves its
        # kernel to the end of the dict (O(1) ``pop`` + reinsert).  That makes
        # ``list(self._active.keys())`` the fair candidate list with no
        # per-round sort -- replacing what used to be an O(N log N) call per
        # outer loop iteration on the simulator's hottest scheduling path.
        # In ``greedy`` mode we simply leave insertion order alone.
        self._active: Dict[KernelId, _KernelState] = {}
        # Completed greenlets (internal bookkeeping)
        self._completed: List[KernelId] = []
        # Main greenlet for the scheduler
        self._main_greenlet: Optional[greenlet] = None
        # Cached bound method ``self._main_greenlet.switch``; set in ``run()``
        # so the hot ``block_current_kernel`` path avoids one attribute load
        # per block.
        self._main_switch: Optional[Callable[[], Any]] = None
        # Currently executing scheduled kernel
        self._current_kernel_id: Optional[KernelId] = None
        # Cached ``_active`` entry for the currently switched-in kernel.  Set
        # right before each ``g.switch()`` from this side of the cooperative
        # boundary; read by ``block_current_kernel`` from the kernel side to
        # mutate ``blocking_obj`` / ``operation`` without a dict lookup.
        self._current_state: Optional[_KernelState] = None
        # Global timestamp counter; only used for the ``tick`` property.
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
            if TRACE.enabled:
                trace("kernel_start")
            func()
            if TRACE.enabled:
                trace("kernel_end")
            # Kernel completed successfully
            self._mark_completed(kernel_id)

        g = greenlet(wrapped_func)
        # Initially not blocked (will start when scheduled).  ``_KernelState``
        # defaults ``blocking_obj=None`` and ``operation=""``.
        self._active[kernel_id] = _KernelState(g, kernel_id.kind)
        # Kernel hasn't made progress yet
        self._has_made_progress[kernel_id] = False

    def block_current_kernel(self, blocking_obj: Any, operation: str) -> None:
        """Block the current scheduled kernel on an operation.

        This is called by wait()/reserve() operations to yield control back
        to the scheduler.  This is the single hottest path in the simulator
        (16-17 M calls per dry-run); see ``_KernelState`` and ``run()`` for
        the supporting design that keeps the body to two slot writes and one
        greenlet switch.

        Args:
            blocking_obj: Object being waited on (DataflowBuffer or CopyTransaction)
            operation: Operation name ("wait" or "reserve")
        """
        # ``_current_state`` and ``_main_switch`` are set by ``run()`` (and
        # ``_initialization_phase()``) immediately before each ``g.switch()``;
        # both are guaranteed to be live whenever a kernel is executing.  The
        # ``Optional`` type on the declarations is just bootstrap state -- we
        # suppress the resulting pyright warnings here rather than pay an
        # ``if is None`` check per call on the simulator's hottest path.
        # The source location of the blocking call is not captured here --
        # see the comment on ``self._active`` in ``__init__``.
        state = self._current_state
        state.blocking_obj = blocking_obj  # pyright: ignore[reportOptionalMemberAccess]
        state.operation = operation  # pyright: ignore[reportOptionalMemberAccess]

        if TRACE.enabled:
            trace("kernel_block", op=operation, on=blocking_obj._trace_name)
        self._main_switch()  # pyright: ignore[reportOptionalCall]
        if TRACE.enabled:
            trace("kernel_unblock")

    def _mark_completed(self, kernel_id: KernelId) -> None:
        """Mark a kernel as completed and remove from active set.

        Args:
            kernel_id: Kernel identity
        """
        if kernel_id in self._active:
            del self._active[kernel_id]
        self._completed.append(kernel_id)

    def mark_kernel_progress(self) -> None:
        """Mark that the current scheduled kernel has made progress.

        This is called by block_if_needed when a kernel successfully proceeds
        past a blocking check without actually blocking.

        Raises:
            RuntimeError: If no kernel is scheduled.
        """
        # Hot path: ~17M calls per matmul-tutorial dry run.  Hoist
        # ``_current_kernel_id`` into a local (saves two repeated attribute
        # lookups) and write directly to ``_has_made_progress``.  The key
        # is guaranteed to exist because every active kernel is registered
        # via ``add_kernel()`` which initialises the entry; the previous
        # ``not in`` check was defensive and redundant.
        kid = self._current_kernel_id
        if kid is None:
            raise RuntimeError(
                "mark_kernel_progress called but no kernel is currently scheduled. "
                "This indicates a bug in the scheduler."
            )
        self._has_made_progress[kid] = True

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

        Scheduling order is encoded by position in the ``_active`` ordered dict
        rather than a per-kernel timestamp. Kernels that made progress (passed at
        least one block_if_needed check) are moved to the end of ``_active``;
        kernels that blocked on their first check stay at the front, giving them
        priority in the next fair-scheduling round.
        """

        for kernel_id in list(self._active.keys()):
            state = self._active[kernel_id]

            # All kernels should start unblocked in init phase
            if state.blocking_obj is not None:
                label = kernel_display_name(kernel_id)
                raise RuntimeError(
                    f"Kernel {label!r} is already blocked at init phase start. "
                    "This indicates a bug in the scheduler."
                )

            # Set current kernel context.  ``_current_state`` is read from the
            # kernel side by ``block_current_kernel``; setting it here keeps
            # that hot path free of dict lookups.
            self._current_kernel_id = kernel_id
            self._current_state = state
            set_current_kernel_type(state.kernel_type)

            try:
                # Run kernel until it blocks or completes
                state.g.switch()

                # Update timestamp only if kernel made progress
                made_progress = self._has_made_progress.get(kernel_id, False)

                if state.g.dead:
                    self._mark_completed(kernel_id)
                elif made_progress:
                    # Kernel passed one or more block_if_needed checks - bump
                    # the logical clock and promote it to the end of
                    # ``_active`` so that the next round's iteration order is
                    # least-recently-run-first without re-sorting.
                    self._timestamp += 1
                    self._active[kernel_id] = self._active.pop(kernel_id)
                # Kernels that blocked on their first check stay at the front

            except Exception as e:
                # Kernel raised an error during initialization
                clear_current_kernel_type()
                self._current_kernel_id = None
                self._current_state = None

                # Format and raise error with source location
                self._format_and_raise_kernel_error(kernel_display_name(kernel_id), e)

            clear_current_kernel_type()

        self._current_kernel_id = None
        self._current_state = None

    def _seed_fair_order(self) -> None:
        """Establish the initial ``_active`` order for fair scheduling.

        Sorts ``_active`` by ``(linear_node, kind.value, func_name)`` -- the
        tie-break key that used to be applied per round inside
        ``_get_fair_kernel_order``.  Called once at the start of ``run()``
        before the initialization phase; from then on the invariant is
        maintained incrementally by moving each just-run kernel to the end
        of ``_active``.
        """
        ordered = sorted(
            self._active.items(),
            key=lambda kv: (kv[0].linear_node, kv[0].kind.value, kv[0].func_name),
        )
        self._active = {kid: state for kid, state in ordered}

    def run(self) -> None:
        """Run all kernels until completion or deadlock is detected."""
        # Store main greenlet for switching back from kernels.  Cache the
        # bound ``.switch`` method so the hot ``block_current_kernel`` path
        # avoids one attribute load per block.
        self._main_greenlet = greenlet.getcurrent()
        self._main_switch = self._main_greenlet.switch

        # Determine scheduling algorithm
        algorithm = get_scheduler_algorithm()
        fair = algorithm == "fair"

        # Phase 1: Initialization - run all kernels until they first block
        # This ensures all kernels have blocking_obj set so can_{operation}() checks work
        if fair:
            # Seed ``_active`` in deterministic tie-break order so that the
            # initialization phase (and the first main-loop round) iterate
            # in the same order the old per-round sort would have produced.
            self._seed_fair_order()
            self._initialization_phase()

        # Phase 2: Main scheduling loop with fairness
        # Run all kernels until completion or deadlock
        while self._active:
            any_progress = False

            # Both modes simply iterate ``_active`` in its current order; in
            # ``fair`` mode that order is the least-recently-run-first
            # sequence maintained by the move-to-end below.
            kernel_candidates = list(self._active.keys())

            # Try to advance each kernel in the selected order
            for kernel_id in kernel_candidates:
                state = self._active.get(kernel_id)
                if state is None:
                    # Kernel may have completed during this iteration
                    continue

                # If kernel is blocked, check if it can proceed
                blocking_obj = state.blocking_obj
                if blocking_obj is not None:
                    can_method = getattr(blocking_obj, f"can_{state.operation}", None)
                    if can_method is None or not can_method():
                        # Still blocked
                        continue

                    # Unblocked! Clear blocking state in place (no tuple churn).
                    state.blocking_obj = None
                    state.operation = ""

                # Set current kernel for block_current_kernel()
                self._current_kernel_id = kernel_id
                self._current_state = state

                # Run kernel until it blocks or completes
                set_current_kernel_type(state.kernel_type)
                try:
                    if state.g.dead:
                        # Kernel already completed (marked by wrapped_func)
                        if kernel_id in self._active:
                            del self._active[kernel_id]
                        continue

                    # Switch to the greenlet
                    state.g.switch()
                    any_progress = True

                    # Always update timestamp after kernel runs.  The
                    # pre-check above already filtered out kernels that
                    # could not make progress.
                    self._timestamp += 1

                    # If greenlet is dead, it completed.  ``wrapped_func``
                    # usually removes the kernel from ``_active`` already; if
                    # not, finish the cleanup here.
                    if state.g.dead:
                        if kernel_id in self._active:
                            self._mark_completed(kernel_id)
                    elif fair:
                        # Promote to the end of ``_active`` so this kernel
                        # is last in the next round's iteration order.  O(1)
                        # on CPython dicts; replaces the per-round
                        # ``_get_fair_kernel_order`` sort.
                        self._active[kernel_id] = self._active.pop(kernel_id)
                except Exception as e:
                    # Kernel raised an error - preserve traceback for debugging
                    clear_current_kernel_type()
                    self._current_kernel_id = None
                    self._current_state = None

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
                self._current_state = None

            # Deadlock detection
            if not any_progress and self._active:
                # Group kernels by (operation, object, location).  Each blocked
                # greenlet is suspended at its wait()/reserve() call site and
                # still has a live frame chain, so we resolve the user code
                # location here -- on the cold path -- rather than capturing
                # it on every block.
                from collections import defaultdict

                blocked_groups: dict[tuple[str, str, str], list[str]] = defaultdict(
                    list
                )
                # Track raw (filename, lineno) per group for pretty printing
                blocked_raw_locs: dict[
                    tuple[str, str, str], Optional[Tuple[str, int]]
                ] = {}

                for kernel_id, state in self._active.items():
                    obj_desc = self._get_obj_description(state.blocking_obj)
                    raw_loc: Optional[Tuple[str, int]] = None
                    if state.g.gr_frame is not None:
                        try:
                            raw_loc = find_user_code_location(state.g.gr_frame)
                        except RuntimeError:
                            raw_loc = None
                    location = f" at {raw_loc[0]}:{raw_loc[1]}" if raw_loc else ""
                    key = (state.operation, obj_desc, location)
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
        operation: Operation name (must be "wait" or "reserve")
    """
    # Fetch scheduler + algorithm through a single context lookup; both come
    # from the same ``SimulatorContext`` and this function is called ~17M
    # times per matmul-tutorial dry run, so collapsing the two helpers'
    # frames + the redundant ``get_context()`` saves measurable wall time.
    ctx = get_context()
    scheduler = ctx.scheduler
    if scheduler is None:
        raise RuntimeError(
            "No active scheduler. This should only be called from within a kernel."
        )
    # Explicit dispatch instead of ``getattr(obj, f"can_{operation}")``:
    # the f-string + attribute lookup costs ~300 ns each call, vs ~30 ns
    # for an ``==`` test and a single attribute load.  ``operation`` is
    # only ever ``"wait"`` or ``"reserve"`` (enforced by the only three
    # call sites: DataflowBuffer.wait/reserve and CopyTransaction.wait).
    # Bind the bound method (not the result) so the fair path can re-check
    # after yielding without re-running the dispatch.
    if operation == "wait":
        can_method = obj.can_wait
    elif operation == "reserve":
        can_method = obj.can_reserve
    else:
        raise ValueError(
            f"block_if_needed: operation must be 'wait' or 'reserve', got {operation!r}"
        )

    if ctx.config.scheduler_algorithm == "fair":
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
