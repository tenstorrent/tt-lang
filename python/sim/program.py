# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Operation execution framework for multi-node simulation.

This module provides the core execution framework for running compute and data movement
functions across multiple nodes with proper context binding and error handling.
"""

import types
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence

from greenlet import getcurrent

from .dfb import DataflowBuffer
from .typedefs import BindableTemplate, Shape
from .kernel import KernelKind
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
    KernelAnalysis,
)
from .diagnostics import print_diagnostic_error
from .debug_print import ttlang_print
from .trace import TRACE, trace

if TYPE_CHECKING:
    # Imported for typing only; .pipe is imported at call time in run_operation.
    from .pipe import AnyPipeNet


def set_max_dfbs(limit: int) -> None:
    """Set the maximum number of DataflowBuffers per node.

    Args:
        limit: Maximum number of CBs per node (must be non-negative)

    Raises:
        ValueError: If limit is negative

    Example:
        set_max_dfbs(64)  # Allow up to 64 CBs per node
    """
    if limit < 0:
        raise ValueError(f"max_dfbs must be non-negative, got {limit}")
    get_context().config.max_dfbs = limit


def get_max_dfbs() -> int:
    """Get the current maximum number of DataflowBuffers per node.

    Returns:
        Current CB limit per node
    """
    return get_context().config.max_dfbs


def set_max_l1_bytes(limit: int) -> None:
    """Set the maximum L1 memory per node (in bytes).

    The L1 memory used by a node is the sum of capacity_bytes across all of its
    DataflowBuffers. Kernel execution issues a warning if the total CB capacity
    on any node exceeds this limit. Defaults to 1336 KiB (Blackhole/Wormhole
    L1 size minus reserved program space).

    Args:
        limit: Maximum L1 bytes per node (must be positive)

    Raises:
        ValueError: If limit is not positive

    Example:
        set_max_l1_bytes(1_572_864)  # 1.5 MB
    """
    if limit <= 0:
        raise ValueError(f"max_l1_bytes must be positive, got {limit}")
    get_context().config.max_l1_bytes = limit


def get_max_l1_bytes() -> int:
    """Get the current L1 memory limit per node in bytes.

    Returns:
        Current L1 limit in bytes
    """
    return get_context().config.max_l1_bytes


def _total_nodes(grid: Shape) -> int:
    """Number of linear nodes for an arbitrary-rank grid."""
    total = 1
    for dim_size in grid:
        total *= dim_size
    return total


# The roles of the three kernels an operation runs, in the order _order_kernels
# returns them.
_KERNEL_ROLES = (KernelKind.COMPUTE, KernelKind.DATA_MOVEMENT, KernelKind.DATA_MOVEMENT)


def _order_kernels(kernels: List[BindableTemplate]) -> tuple[BindableTemplate, ...]:
    """Validate and order registered kernels as (compute, dm0, dm1).

    Raises:
        ValueError: If the operation did not register exactly one compute and
            two data-movement kernels.
    """
    if len(kernels) != 3:
        raise ValueError(
            f"Operation must define exactly 3 kernels (compute, dm0, dm1), got {len(kernels)}"
        )
    compute_kernels = [
        t for t in kernels if getattr(t, "kernel_type", None) == KernelKind.COMPUTE
    ]
    dm_kernels = [
        t
        for t in kernels
        if getattr(t, "kernel_type", None) == KernelKind.DATA_MOVEMENT
    ]
    if len(compute_kernels) != 1:
        raise ValueError(
            f"Kernel must define exactly 1 compute kernel, got {len(compute_kernels)}"
        )
    if len(dm_kernels) != 2:
        raise ValueError(
            f"Kernel must define exactly 2 datamovement kernels, got {len(dm_kernels)}"
        )
    return (compute_kernels[0], dm_kernels[0], dm_kernels[1])


def _operation_node_context(
    node: int, grid: Shape, ordered: tuple[BindableTemplate, ...]
) -> Dict[str, Any]:
    """Build the bind context for a node whose kernels were produced by running
    the operation body under that node's context.

    The kernels already close over this node's DataflowBuffers and setup
    scalars, so the bind context only needs ``_node`` / ``grid`` / ``print``
    (so ``ttl.node()`` etc. resolve at kernel-execution time) plus the node's
    DataflowBuffers by name (so end-of-run auto-push/pop and validation can
    reach them).  Tensor freevars are named too so trace/locality stats can
    attribute copies to the operation's parameter names.
    """
    node_context: Dict[str, Any] = {}
    for tmpl in ordered:
        # BindableTemplate declares __wrapped__ (the original kernel function);
        # typed Any here so closure/code introspection below stays untyped.
        func: Any = tmpl.__wrapped__
        closure = func.__closure__ or ()
        for name, cell in zip(func.__code__.co_freevars, closure):
            try:
                value: Any = cell.cell_contents
            except ValueError:
                continue
            # DataflowBuffers are named so end-of-run auto-push/pop and
            # validation can reach them; Tensors so trace/locality stats can
            # attribute copies to the operation's parameter names.
            if isinstance(value, (DataflowBuffer, Tensor)):
                setattr(value, "_name", name)
                node_context[name] = value
    node_context["_node"] = node
    node_context["grid"] = grid
    node_context["print"] = ttlang_print
    return node_context


def _dedupe_pipe_nets(nets: Sequence["AnyPipeNet"]) -> List["AnyPipeNet"]:
    """Deduplicate discovered PipeNets by content.

    Re-running the operation body per node creates a fresh PipeNet object each
    time; when the pipe set is node-independent (the common case, matching the
    compiler) they are identical and collapse to one.  Deduping by the tuple of
    pipes (Pipe is a frozen, hashable dataclass) keeps one entry per distinct
    pipe set and preserves encounter order.

    Per-node re-execution is also what makes a node-dependent pipe set
    expressible: a src or dst derived from ``ttl.node()`` yields a different net
    per node, and every distinct one is kept, node 0's first.  The operation's
    graph then holds all of them, so each is validated on its own and the
    active-node set is the union across them -- a node runs when it participates
    in any node's version of the net, not only in its own.  The compiler
    evaluates the body once and sees a single net, so such a body has no
    counterpart there.
    """
    seen: set[Any] = set()
    unique: List["AnyPipeNet"] = []
    for net in nets:
        # net.pipes, not a defaulted getattr: keying on a missing attribute
        # would quietly merge unrelated objects into a single entry.
        key = net.pipes
        if key not in seen:
            seen.add(key)
            unique.append(net)
    return unique


def run_operation(
    body: Callable[..., Any],
    grid: Shape,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """Execute an ``@ttl.operation`` body across the grid, single-phase.

    The operation body (everything inside ``@ttl.operation`` but outside the
    nested compute/datamovement kernels) is re-run once per node with that
    node's context injected, so node-dependent setup -- ``ttl.node()``,
    ``ttl.grid_size()``, and any scalars derived from them -- is computed
    per node.  Each run produces that node's DataflowBuffers (shared among the
    node's kernels) and its three kernels (whose closures capture the per-node
    state).  PipeNets discovered across the per-node runs are aggregated to
    compute the active-node set -- which is why every node's body runs, inactive
    ones included: the set is not known until they have.  Their kernels are then
    skipped, but their setup has already happened.

    The compiler evaluates the body once instead, at compile time, so a body
    that mutates state of the enclosing scope sees those effects repeated here
    and not there.  The specification does not currently say how many times an
    implementation may evaluate the body.
    """
    from .decorators import clear_kernel_registry, get_registered_kernels
    from .pipe import build_pipenets, discover_pipe_nets_from_closures

    total_nodes = _total_nodes(grid)
    ctx = get_context()

    node_plans: Dict[int, tuple[Dict[str, Any], tuple[BindableTemplate, ...]]] = {}
    all_nets: List["AnyPipeNet"] = []
    node_footprints: Dict[int, tuple[int, int]] = {}

    # The body reaches ttl.node() / ttl.grid_size() through the names `_node` and
    # `grid` in its own globals, which the frame walk those functions do looks up.
    # `grid` is put there once by @ttl.operation, on a private copy of the module's
    # globals, so only the node changes here -- and only the node is taken away
    # afterwards, since removing `grid` would leave the body unable to run again.
    body_globals = getattr(body, "__globals__", {})
    try:
        for node in range(total_nodes):
            body_globals["_node"] = node

            clear_kernel_registry()
            ctx.kernel_dfb_count = 0
            ctx.kernel_l1_bytes = 0

            body(*args, **kwargs)

            ordered = _order_kernels(get_registered_kernels())
            node_context = _operation_node_context(node, grid, ordered)
            node_plans[node] = (node_context, ordered)

            # Record every node's footprint for the hardware-limit warnings:
            # the limits are per node, and re-running the body per node means a
            # block_count or shape derived from ttl.node() gives each node its
            # own.
            node_footprints[node] = (ctx.kernel_dfb_count, ctx.kernel_l1_bytes)

            kernel_funcs = [getattr(t, "__wrapped__", None) for t in ordered]
            all_nets.extend(discover_pipe_nets_from_closures(body, *kernel_funcs))
    finally:
        # A node index left behind would answer ttl.node() outside a run.
        body_globals.pop("_node", None)

    pipenets = build_pipenets(_dedupe_pipe_nets(all_nets))
    pipenets.validate()

    _schedule_and_run(
        node_plan_for=lambda n: node_plans[n],
        candidate_nodes=range(total_nodes),
        analysis_templates=tuple(
            template for _, ordered in node_plans.values() for template in ordered
        ),
        pipenets=pipenets,
        grid=grid,
        node_footprints=node_footprints,
    )


def _warn_over_hardware_limits(node_footprints: Dict[int, tuple[int, int]]) -> None:
    """Warn when any node's DataflowBuffer count or L1 footprint is over budget.

    ``node_footprints`` maps a node to the ``(dfb_count, l1_bytes)`` its setup
    produced.

    Both limits are per node, so the worst node decides.  Reporting only one
    node's footprint would miss the node that is actually over budget, since
    re-running the body per node means a ``block_count`` or ``shape`` derived
    from ``ttl.node()`` gives each node a footprint of its own.  Nodes that a
    PipeNet leaves inactive are included: their buffers are built here like any
    other node's, and a limit exceeded anywhere is worth reporting.
    """

    def worst(index: int) -> tuple[int, str]:
        """The largest footprint at ``index``, and where it is if nodes differ.

        The node is named only when the footprints are not all the same, since
        naming one of a set of identical nodes suggests the node is the problem.
        """
        node = max(node_footprints, key=lambda n: node_footprints[n][index])
        values = {footprint[index] for footprint in node_footprints.values()}
        return node_footprints[node][index], (
            f" on node {node}" if len(values) > 1 else ""
        )

    # Count out the simulator frames between here and the user's call, so the
    # warning is reported against the line that ran the operation rather than
    # against a simulator source file the reader cannot act on: this helper,
    # _schedule_and_run, run_operation, and the wrapper @ttl.operation installed.
    stacklevel = 5

    dfb_count, dfb_where = worst(0)
    max_dfbs = get_max_dfbs()
    if dfb_count > max_dfbs:
        warnings.warn(
            f"Kernel defines {dfb_count} dataflow buffers{dfb_where}, "
            f"but the configured limit is {max_dfbs}. "
            f"Reduce the number of ttl.make_dataflow_buffer_like() calls, "
            f"or raise the limit with --max-dfbs. Blackhole supports 64 "
            f"physical DFB indices; Wormhole B0 and Quasar support 32.",
            stacklevel=stacklevel,
        )

    l1_bytes, l1_where = worst(1)
    max_l1 = get_max_l1_bytes()
    if l1_bytes > max_l1:
        warnings.warn(
            f"Total DataflowBuffer capacity per node ({l1_bytes} bytes"
            f"{l1_where}) exceeds the L1 memory limit of {max_l1} bytes. "
            f"Memory is accounted using declared dtypes, so this reflects "
            f"the on-hardware footprint of the kernel.",
            stacklevel=stacklevel,
        )


def _schedule_and_run(
    *,
    node_plan_for: Callable[[int], tuple[Dict[str, Any], Any]],
    candidate_nodes: Any,
    analysis_templates: Any,
    pipenets: Any,
    grid: Shape,
    node_footprints: Dict[int, tuple[int, int]],
) -> None:
    """Analyse kernels, schedule active nodes, run the scheduler, and validate.

    ``node_plan_for(node)`` returns ``(node_context, ordered_templates)`` for a
    node: the context is the bind context (and the source of DataflowBuffers for
    end-of-run cleanup/validation), and ``ordered_templates`` is (compute, dm0,
    dm1).  ``analysis_templates`` is every node's templates: a body that picks
    its kernels by ``ttl.node()`` gives different nodes different code, so one
    node's triple would leave the rest without copy-wait injection points.
    Nodes that do share code cost nothing, because the analysis is keyed by code
    object and skips what it has already visited.  ``node_footprints`` carries
    each node's DataflowBuffer footprint for the hardware-limit warnings; see
    :func:`_warn_over_hardware_limits`.
    """
    _warn_over_hardware_limits(node_footprints)

    scheduler = GreenletScheduler()
    set_scheduler(scheduler)

    # Analyse the kernel functions (and any reachable helpers) of every node.  A
    # shared visited set prevents duplicate analysis -- and duplicate violation
    # reports -- when helpers are called by more than one kernel, or when nodes
    # run the same code, which is the common case.
    _empty = KernelAnalysis(injection_points=(), bare_copy_linenos=frozenset())
    _visited: set[int] = set()
    injection_map: dict[types.CodeType, KernelAnalysis] = {}
    all_violations: List[PatternViolation] = []

    for tmpl in analysis_templates:
        analyses = collect_reachable_analyses(tmpl.__wrapped__, _visited)
        injection_map.update(analyses)
        top = analyses.get(tmpl.__wrapped__.__code__, _empty)
        all_violations.extend(top.violations)

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
            f"Found {n} unsupported pattern{'s' if n > 1 else ''} in kernel "
            "function(s). See errors above for details."
        )

    # Compute the PipeNet active set: linear node indices that participate in
    # any pipe as source or destination.  Inactive nodes skip every kernel, as the
    # specification's gather example describes ("nodes outside the active
    # rectangle skip the operation body") -- but only the kernels: their setup has
    # already run, because the active set is not known until every node's body has
    # been evaluated and its pipes collected (see run_operation).  So an inactive
    # node's dataflow buffers exist and count against the hardware limits.
    #
    # The compiler does not insert this guard: TTLVerifyPipeNetGuards verifies
    # that the user's own scf.if / ttl.if_src / ttl.if_dst narrows the nodes
    # around pipe-coupled operations, which is where the specification puts the
    # obligation. So a program that reads a pipe outside such a guard is refused
    # there and quietly skipped here (tt-lang issue #804).
    active_nodes = (
        pipenets.active_node_set(tuple(grid)) if pipenets is not None else None
    )

    def _is_active(node: int) -> bool:
        return active_nodes is None or node in active_nodes

    try:
        # Track all per-node contexts for validation, each under the node it
        # belongs to: a pipe net can leave nodes out, so a context's position in
        # this list is not its node.
        all_node_contexts: List[tuple[int, Dict[str, Any]]] = []

        for node in candidate_nodes:
            # Skip nodes that are not in any PipeNet's active set.
            if not _is_active(node):
                continue

            node_context, ordered = node_plan_for(node)
            all_node_contexts.append((node, node_context))
            # Add kernels to scheduler (one compute + two DM per node).
            # Identity is (node, kind, __name__); the two DM kernels on a node
            # must have distinct __name__s -- the scheduler rejects duplicates
            # with a user-facing error.
            # The roles come from the ordering rather than being read off the
            # template again: _order_kernels admitted these three by role, so
            # asking a second time can only disagree with itself.
            for kernel_type, tmpl in zip(_KERNEL_ROLES, ordered):
                # Bind template to node context.
                bound_func = tmpl.bind(node_context)

                # Wrap to tag the greenlet with its linear node index so
                # locality analysis in copy.py can read it via getcurrent().
                def _tagged(
                    fn: Callable[[], Any] = bound_func,
                    n: int = node,
                    node_ctx: Dict[str, Any] = node_context,
                ) -> None:
                    getcurrent()._sim_node = n  # type: ignore[attr-defined]
                    fn()
                    # Auto-push/pop any blocks still pending when the kernel
                    # function returns normally (final-iteration cleanup).
                    # This must not run during exception propagation, so it
                    # is placed after fn() rather than in a finally block.
                    for _val in node_ctx.values():
                        if isinstance(_val, DataflowBuffer):
                            _val.auto_push_block()
                            _val.auto_pop_block()

                scheduler.add_kernel(
                    KernelId(node, kernel_type, tmpl.__name__),
                    _tagged,
                )

        # Install injection hooks for all discovered code objects (kernel
        # functions, nested defs, and module-scope helpers).
        install_copy_wait_hooks(injection_map)

        # Nodes that actually run kernels. Inactive nodes are not traced.
        running_nodes = [n for n in candidate_nodes if _is_active(n)]

        # Emit operation_start for each node before the scheduler runs.
        if TRACE.enabled:
            for n in running_nodes:
                trace("operation_start", node=n)

        # Run scheduler; if any kernel raises, the exception propagates
        # immediately and the validation below is intentionally skipped.
        # Reporting a "simulator bug" for unpushed blocks only makes sense when
        # all kernels completed normally (auto-push/pop should have fired).
        scheduler.run()

        # Emit operation_end for each node now that all kernels completed.
        if TRACE.enabled:
            for n in running_nodes:
                trace("operation_end", node=n)

        # Validate all DataflowBuffers have no pending blocks.
        # Only reached on normal exit from the scheduler.
        _validate_dataflow_buffers(all_node_contexts)
    finally:
        set_scheduler(None)


def _validate_dataflow_buffers(
    all_node_contexts: List[tuple[int, Dict[str, Any]]],
) -> None:
    """Validate that all DataflowBuffers have no pending blocks at end of execution.

    Args:
        all_node_contexts: The nodes that ran, each with the context holding its
            DataflowBuffers.  The node is carried rather than counted, so that a
            failure on node 3 of an operation whose active nodes are {1, 3} is
            reported against node 3 and not against the second one that ran.

    Raises:
        RuntimeError: If any DataflowBuffer has pending blocks
    """
    errors: List[str] = []
    for node, node_context in all_node_contexts:
        for key, value in node_context.items():
            match value:
                case DataflowBuffer():
                    try:
                        value.validate_no_pending_blocks()
                    except RuntimeError as e:
                        errors.append(f"node{node}.{key}: {e}")
                case _:
                    pass

    if errors:
        raise RuntimeError(
            "Kernel execution completed with incomplete DataflowBuffer operations:\n"
            + "\n".join(errors)
        )
