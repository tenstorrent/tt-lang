# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Plan and apply logical-kernel splitting for unified operations.

The analysis records statement placement and DFB transaction ownership on the
unmodified AST. Application then clones and prunes the body once per selected
logical kernel. Target-specific processor assignment occurs in ``ttl.atom``
after this module returns the logical split.
"""

from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Union

from ttl.dfb_reset import DFBReset, _BoundDFBReset
from ttl.dfb_reconfiguration import (
    DFBReconfiguration,
    _BoundDFBReconfiguration,
)
from ttl.fabric import FabricManagerClaim
from ttl.kernel import (
    Kernel,
    KernelKind,
    KernelSelector,
    PIPE_SOURCE_KERNEL,
    _DFB_RELEASE_METHODS,
    _format_kernel_capacity_error,
    _format_selector,
    _selector_kind,
    _selector_sort_key,
)

_EXTERNAL_CALL_NAME = "call_extern_func"
_KERNEL_KEYWORD = "kernel"
_DFB_RESET_CALLS = frozenset({"reset_dfbs", "reset_all_dfbs"})
_DFB_RECONFIGURATION_CALL_NAME = "reconfigure_dfbs"


class _Placement(Enum):
    DATA_MOVEMENT = auto()
    CONTROL = auto()
    DEFERRED = auto()


# ----- op registry ----------------------------------------------------------


# DATA_MOVEMENT resolves to the current callback's logical data-movement
# kernel. CONTROL does not constrain placement.
_TTL_OPS: Dict[str, Union[KernelKind, _Placement]] = {
    # Data movement
    "copy": _Placement.DATA_MOVEMENT,
    "element_read": _Placement.DATA_MOVEMENT,
    "element_write": _Placement.DATA_MOVEMENT,
    "read_index": _Placement.DATA_MOVEMENT,
    # Compute
    "fill": KernelKind.COMPUTE,
    "matmul": KernelKind.COMPUTE,
    "reduce_sum": KernelKind.COMPUTE,
    "reduce_max": KernelKind.COMPUTE,
    "broadcast": KernelKind.COMPUTE,
    "transpose": KernelKind.COMPUTE,
    "exp": KernelKind.COMPUTE,
    "mul": KernelKind.COMPUTE,
    "add": KernelKind.COMPUTE,
    "sub": KernelKind.COMPUTE,
    "div": KernelKind.COMPUTE,
    "recip": KernelKind.COMPUTE,
    "neg": KernelKind.COMPUTE,
    "sqrt": KernelKind.COMPUTE,
    "tanh": KernelKind.COMPUTE,
    "log": KernelKind.COMPUTE,
    "abs": KernelKind.COMPUTE,
    "relu": KernelKind.COMPUTE,
    "sign": KernelKind.COMPUTE,
    "sigmoid": KernelKind.COMPUTE,
    "gelu": KernelKind.COMPUTE,
    # Compile-time / scalar producers: duplicated, not anchored.
    "Pipe": _Placement.CONTROL,
    "PipeNet": _Placement.CONTROL,
    "make_dataflow_buffer_like": _Placement.CONTROL,
    "make_tensor_backed_dfb": _Placement.CONTROL,
    "node": _Placement.CONTROL,
    "dfb_descriptor": _Placement.CONTROL,
    "get_dfb_id": _Placement.CONTROL,
    "raw_addr": _Placement.CONTROL,
    "grid_size": _Placement.CONTROL,
    "dims": _Placement.CONTROL,
    "cores": _Placement.CONTROL,
    "tile_index": _Placement.CONTROL,
    "signpost": _Placement.CONTROL,
}

# Every ttl.<ns>.<name>(...) in a namespace shares one placement classification.
_TTL_NAMESPACES: Dict[str, Union[KernelKind, _Placement]] = {
    "math": KernelKind.COMPUTE,
    "block": KernelKind.COMPUTE,
    "DFBAccess": _Placement.CONTROL,
    "DFBEffect": _Placement.CONTROL,
}

# Pipe source and destination callbacks have distinct logical affinities.
_PIPENET_METHODS: Dict[str, KernelSelector] = {
    "if_src": PIPE_SOURCE_KERNEL,
    "if_dst": KernelKind.DATA_MOVEMENT,
}

# Block releases use the transaction owner computed before statement planning.
_BLOCK_METHODS: Dict[str, Union[KernelKind, _Placement]] = {
    "store": KernelKind.COMPUTE,
    "pop": _Placement.DEFERRED,
    "push": _Placement.DEFERRED,
}

# Methods on a DFB name that produce a block.
_DFB_PRODUCING_METHODS: Set[str] = {"wait", "reserve"}
_DFB_DIRECT_METHODS: Dict[str, _Placement] = {"publish": _Placement.DATA_MOVEMENT}


# ----- shared call classification ------------------------------------------


def _materialize_kernels(
    classification: Union[KernelKind, _Placement],
    data_movement_kernels: FrozenSet[KernelSelector],
) -> Optional[FrozenSet[KernelSelector]]:
    if classification is _Placement.DATA_MOVEMENT:
        return data_movement_kernels
    if isinstance(classification, KernelKind):
        return frozenset({classification})
    return None


def _is_external_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == _EXTERNAL_CALL_NAME
    return (
        isinstance(func, ast.Attribute)
        and func.attr == _EXTERNAL_CALL_NAME
        and isinstance(func.value, ast.Name)
        and func.value.id == "ttl"
    )


def _is_dfb_reconfiguration_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == _DFB_RECONFIGURATION_CALL_NAME
    return (
        isinstance(func, ast.Attribute)
        and func.attr == _DFB_RECONFIGURATION_CALL_NAME
        and isinstance(func.value, ast.Name)
        and func.value.id == "ttl"
    )


def _kernel_keyword(call: ast.Call) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == _KERNEL_KEYWORD:
            return keyword.value
    return None


def _keyword_value(call: ast.Call, name: str) -> Optional[ast.expr]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_kernel_kind_union(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)


def _flatten_kernel_kind_union(node: ast.expr) -> List[ast.expr]:
    if not _is_kernel_kind_union(node):
        return [node]
    assert isinstance(node, ast.BinOp)
    return _flatten_kernel_kind_union(node.left) + _flatten_kernel_kind_union(
        node.right
    )


class _DefaultTTLSelectorNamespace:
    KernelKind = KernelKind
    PIPE_SOURCE_KERNEL = PIPE_SOURCE_KERNEL


_DEFAULT_TTL_SELECTOR_NAMESPACE = _DefaultTTLSelectorNamespace()
_MISSING_SELECTOR_VALUE = object()


class _KernelSelectorResolver:
    def __init__(
        self,
        logical_kernels: Mapping[str, Kernel],
        selector_scope: Mapping[str, object],
    ):
        self.logical_kernels = dict(logical_kernels)
        self.selector_scope = dict(selector_scope)
        for name, kernel in self.logical_kernels.items():
            if not isinstance(kernel, Kernel):
                raise TypeError(
                    f"logical kernel {name!r} must be a Kernel, got "
                    f"{type(kernel).__name__}"
                )
            self.selector_scope[name] = kernel

    def resolve_external(
        self,
        call: ast.Call,
        inferred_kernels: FrozenSet[KernelSelector],
    ) -> FrozenSet[KernelSelector]:
        selector = _kernel_keyword(call)
        if selector is None:
            if len(inferred_kernels) == 1:
                self._validate_fabric_manager_effects(call, inferred_kernels)
                return inferred_kernels
            raise _split_error(
                call,
                "call_extern_func requires a kernel selector when its logical "
                "kernel cannot be inferred",
            )

        selected = self._resolve_selection(selector, allow_tuple=True)
        if inferred_kernels and selected != inferred_kernels:
            raise _split_error(
                selector,
                "explicit external-call kernel selection "
                f"({_format_kernels(selected)}) conflicts with inferred "
                f"selection ({_format_kernels(inferred_kernels)})",
            )
        self._validate_fabric_manager_effects(call, selected)
        return selected

    def _validate_fabric_manager_effects(
        self, call: ast.Call, selected: FrozenSet[KernelSelector]
    ) -> None:
        effects = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "fabric_manager_effects"
            ),
            None,
        )
        if not isinstance(effects, (ast.Tuple, ast.List)):
            return
        for effect in effects.elts:
            if (
                not isinstance(effect, ast.Call)
                or not isinstance(effect.func, ast.Attribute)
                or not isinstance(effect.func.value, ast.Name)
            ):
                continue
            claim = self.selector_scope.get(effect.func.value.id)
            if not isinstance(claim, FabricManagerClaim):
                continue
            claim_selection = frozenset({claim.kernel})
            if selected != claim_selection:
                raise _split_error(
                    effect,
                    f"fabric manager claim {claim.identity!r} selects "
                    f"{_format_kernels(claim_selection)}, but the external "
                    f"call selects {_format_kernels(selected)}",
                )

    def resolve_reset(self, call: ast.Call) -> FrozenSet[KernelSelector]:
        if len(call.args) != 1:
            raise _split_error(
                call, f"{ast.unparse(call.func)} requires one DFBReset argument"
            )
        reset_node = call.args[0]
        reset = self._resolve_reference(reset_node)
        if isinstance(reset, _BoundDFBReset):
            participants = reset.participants
        elif isinstance(reset, DFBReset):
            participants = reset.participants
        else:
            type_detail = ""
            if reset is not _MISSING_SELECTOR_VALUE:
                type_detail = f", got {type(reset).__name__}"
            raise _split_error(
                reset_node,
                f"{ast.unparse(call.func)} reset must be a DFBReset captured by "
                f"the enclosing operation{type_detail}",
            )
        for participant in participants:
            if participant._implicit_role is None and not any(
                participant is kernel for kernel in self.logical_kernels.values()
            ):
                raise _split_error(
                    reset_node,
                    "DFBReset participant Kernel must be declared by the "
                    "enclosing operation",
                )
        return frozenset(participants)

    def resolve_dfb_reconfiguration(self, call: ast.Call) -> FrozenSet[KernelSelector]:
        if len(call.args) != 1 or call.keywords:
            raise _split_error(
                call,
                "reconfigure_dfbs requires exactly one positional "
                "DFBReconfiguration argument",
            )
        boundary_node = call.args[0]
        boundary = self._resolve_reference(boundary_node)
        if isinstance(boundary, _BoundDFBReconfiguration):
            participants = boundary.participants
        elif isinstance(boundary, DFBReconfiguration):
            participants = boundary.participants
        else:
            type_detail = ""
            if boundary is not _MISSING_SELECTOR_VALUE:
                type_detail = f", got {type(boundary).__name__}"
            raise _split_error(
                boundary_node,
                "reconfigure_dfbs argument must be a DFBReconfiguration "
                f"captured by the enclosing operation{type_detail}",
            )
        for participant in participants:
            if (
                isinstance(participant, Kernel)
                and participant._implicit_role is None
                and not any(
                    participant is kernel for kernel in self.logical_kernels.values()
                )
            ):
                raise _split_error(
                    boundary_node,
                    "DFBReconfiguration participant Kernel must be declared "
                    "by the enclosing operation",
                )
        return frozenset(participants)

    def resolve_release(self, call: ast.Call) -> Optional[KernelSelector]:
        selector = _kernel_keyword(call)
        if selector is None:
            return None
        if isinstance(selector, ast.Tuple):
            raise _split_error(
                selector,
                f"{ast.unparse(call.func)} accepts one kernel selector, not a tuple",
            )
        if _is_kernel_kind_union(selector):
            raise _split_error(
                selector,
                f"{ast.unparse(call.func)} accepts one kernel selector, not a kind union",
            )
        selected = self._resolve_selector(selector)
        return selected

    def _resolve_selection(
        self,
        node: ast.expr,
        allow_tuple: bool,
    ) -> FrozenSet[KernelSelector]:
        if _is_kernel_kind_union(node):
            selectors = [
                self._resolve_selector(element)
                for element in _flatten_kernel_kind_union(node)
            ]
            if not all(isinstance(selector, KernelKind) for selector in selectors):
                raise _split_error(
                    node,
                    "kernel kind union operands must be KernelKind members; "
                    "use a tuple to include operation-local Kernel handles",
                )
            return self._validate_multiple_selection(node, selectors)
        if not isinstance(node, ast.Tuple):
            return frozenset({self._resolve_selector(node)})
        if not allow_tuple:
            raise _split_error(node, "expected one kernel selector, not a tuple")
        if not node.elts:
            raise _split_error(
                node,
                "call_extern_func kernel selection requires a nonempty tuple",
            )
        selectors = [self._resolve_selector(element) for element in node.elts]
        return self._validate_multiple_selection(node, selectors)

    def _validate_multiple_selection(
        self,
        node: ast.expr,
        selectors: List[KernelSelector],
    ) -> FrozenSet[KernelSelector]:
        selected = frozenset(selectors)
        if len(selected) != len(selectors):
            raise _split_error(
                node,
                "call_extern_func kernel selection contains a duplicate kernel selector",
            )
        return selected

    def _resolve_selector(self, node: ast.expr) -> KernelSelector:
        value = self._resolve_reference(node)
        if isinstance(value, KernelKind):
            return value
        if isinstance(value, Kernel) and (
            value is PIPE_SOURCE_KERNEL
            or any(value is kernel for kernel in self.logical_kernels.values())
        ):
            return value
        if isinstance(node, ast.Attribute):
            owner = self._resolve_reference(node.value)
            if owner is KernelKind and value is _MISSING_SELECTOR_VALUE:
                raise _split_error(
                    node,
                    f"unknown KernelKind member {node.attr!r}",
                )
        type_detail = ""
        if value is not _MISSING_SELECTOR_VALUE:
            type_detail = f", got {type(value).__name__}"
        raise _split_error(
            node,
            "kernel selector must be a KernelKind or Kernel declared as a "
            f"top-level operation resource{type_detail}",
        )

    def _resolve_reference(self, node: ast.expr):
        if isinstance(node, ast.Name):
            if node.id in self.selector_scope:
                return self.selector_scope[node.id]
            if node.id == "KernelKind":
                return KernelKind
            if node.id == "ttl":
                return _DEFAULT_TTL_SELECTOR_NAMESPACE
            return _MISSING_SELECTOR_VALUE
        if not isinstance(node, ast.Attribute):
            return _MISSING_SELECTOR_VALUE
        owner = self._resolve_reference(node.value)
        if owner is _MISSING_SELECTOR_VALUE:
            return _MISSING_SELECTOR_VALUE
        try:
            return inspect.getattr_static(owner, node.attr)
        except AttributeError:
            return _MISSING_SELECTOR_VALUE


def _classify_ttl_call(
    call: ast.Call,
    data_movement_kernels: FrozenSet[KernelSelector],
    inferred_external_kernels: FrozenSet[KernelSelector],
    selector_resolver: _KernelSelectorResolver,
) -> Optional[FrozenSet[KernelSelector]]:
    """Return a registry-driven call's logical-kernel selection."""
    if _is_external_call(call):
        return selector_resolver.resolve_external(call, inferred_external_kernels)
    if _is_dfb_reconfiguration_call(call):
        return selector_resolver.resolve_dfb_reconfiguration(call)

    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        receiver = func.value.id
        name = func.attr
        if receiver == "ttl":
            if name in _DFB_RESET_CALLS:
                return selector_resolver.resolve_reset(call)
            classification = _TTL_OPS.get(name)
            if classification is None:
                raise _split_error(
                    func,
                    f"unknown ttl.{name}(...) call; register its logical "
                    "kernel classification in atom_split._TTL_OPS",
                )
            return _materialize_kernels(classification, data_movement_kernels)
        if name in _PIPENET_METHODS:
            return frozenset({_PIPENET_METHODS[name]})
        return None

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
        outer = func.value
        if isinstance(outer.value, ast.Name) and outer.value.id == "ttl":
            namespace = outer.attr
            classification = _TTL_NAMESPACES.get(namespace)
            if classification is None:
                raise _split_error(
                    func,
                    f"unknown ttl.{namespace}.{func.attr}(...) call; register "
                    f"namespace 'ttl.{namespace}' in atom_split._TTL_NAMESPACES",
                )
            return _materialize_kernels(classification, data_movement_kernels)

    return None


# ----- public API -----------------------------------------------------------


@dataclass(frozen=True)
class StatementSelection:
    statement_id: int
    kernels: FrozenSet[KernelSelector]
    source_line: int


@dataclass(frozen=True)
class DFBTransactionPlan:
    block_name: str
    inferred_kernels: FrozenSet[KernelSelector]
    explicit_kernels: FrozenSet[KernelSelector]
    owner: KernelSelector
    source_line: int


@dataclass(frozen=True)
class SplitPlan:
    kernels: Tuple[KernelSelector, ...]
    statements: Tuple[StatementSelection, ...]
    transactions: Tuple[DFBTransactionPlan, ...]
    kernel_requirements: Tuple[Tuple[KernelKind, int], ...]
    target_capacities: Tuple[Tuple[KernelKind, int], ...]


@dataclass
class SplitResult:
    plan: SplitPlan
    _bodies: Dict[KernelSelector, List[ast.stmt]]

    @property
    def kernels(self) -> Tuple[KernelSelector, ...]:
        return self.plan.kernels

    def body_for(self, kernel: KernelSelector) -> List[ast.stmt]:
        if not isinstance(kernel, (KernelKind, Kernel)):
            raise TypeError(
                "body_for() requires a KernelKind or Kernel, got "
                f"{type(kernel).__name__}"
            )
        return self._bodies.get(kernel, [ast.Pass()])


class _AnalysisState:
    def __init__(self):
        self.anchor_selections: Dict[int, FrozenSet[KernelSelector]] = {}
        self.control_header_selections: Dict[int, FrozenSet[KernelSelector]] = {}
        self.kernel_origins: Dict[KernelSelector, ast.stmt] = {}
        self.transactions: List[DFBTransactionPlan] = []

    def select(self, statement: ast.stmt, kernels: Set[KernelSelector]) -> None:
        self.anchor_selections[id(statement)] = frozenset(kernels)
        for kernel in kernels:
            self.kernel_origins.setdefault(kernel, statement)

    def extend(self, statement: ast.stmt, kernels: Set[KernelSelector]) -> None:
        existing = self.anchor_selections.get(id(statement), frozenset())
        self.anchor_selections[id(statement)] = existing | frozenset(kernels)


def split_function_body(
    fn_def: ast.FunctionDef,
    dfb_param_names: Set[str],
    local_dfb_names: Optional[Set[str]] = None,
    logical_kernels: Optional[Mapping[str, Kernel]] = None,
    selector_scope: Optional[Mapping[str, object]] = None,
    kernel_capacities: Optional[Mapping[KernelKind, int]] = None,
) -> SplitResult:
    """Split a unified operation body into target-independent logical kernels.

    Args:
        fn_def: AST FunctionDef of the user's @ttl.operation function.
        dfb_param_names: parameter names annotated as ttl.DFB / ttl.DFB.Output.
        local_dfb_names: names of DFBs declared inside the body via a DFB
            factory. Treated as DFB receivers for wait/reserve recognition.
        logical_kernels: lifted logical Kernel resources keyed by source name.
        selector_scope: frozen operation scope used for selector references.
        kernel_capacities: target-provided maximum kernel counts by kind.
    """
    dfb_names = set(dfb_param_names) | (local_dfb_names or set())
    selector_resolver = _KernelSelectorResolver(
        logical_kernels or {}, selector_scope or {}
    )
    state = _AnalysisState()
    _AnchorPlanner(
        dfb_names=dfb_names,
        selector_resolver=selector_resolver,
        state=state,
    ).analyze(fn_def.body)

    selected_kernels: Set[KernelSelector] = set()
    for selection in state.anchor_selections.values():
        selected_kernels.update(selection)
    ordered_kernels = tuple(sorted(selected_kernels, key=_selector_sort_key))
    _validate_kernel_capacities(
        fn_def,
        ordered_kernels,
        state.kernel_origins,
        kernel_capacities,
    )

    all_kernels = frozenset(ordered_kernels)
    _ScalarLivenessPlanner(state, all_kernels).analyze(fn_def.body)
    statements = tuple(
        StatementSelection(
            statement_id=id(statement),
            kernels=state.anchor_selections.get(id(statement), all_kernels),
            source_line=getattr(statement, "lineno", 0),
        )
        for statement in _walk_statements(fn_def.body)
    )
    requirements = tuple(
        (
            kind,
            sum(_selector_kind(kernel) == kind for kernel in ordered_kernels),
        )
        for kind in KernelKind
    )
    target_capacities = tuple(
        sorted(
            (kernel_capacities or {}).items(),
            key=lambda item: _selector_sort_key(item[0]),
        )
    )
    plan = SplitPlan(
        kernels=ordered_kernels,
        statements=statements,
        transactions=tuple(state.transactions),
        kernel_requirements=requirements,
        target_capacities=target_capacities,
    )
    bodies = {
        kernel: _apply_split_plan(fn_def.body, kernel, plan)
        for kernel in ordered_kernels
    }
    return SplitResult(
        plan=plan,
        _bodies=bodies,
    )


# ----- logical-kernel analysis ---------------------------------------------


class _AnchorPlanner:
    """Analyze statement placement and DFB transactions without AST mutation.

    Anchors fall into three categories:

    1. Direct anchors: any statement expression containing a registered
       ``ttl.<op>(...)`` call, a ``<pipenet>.if_src/if_dst(...)`` call,
       a DFB direct-method call (``dfb.publish()``), a block-method call
       (``blk.store(...)``), or a ``@`` MatMult. Every executable anchor in
       one expression must have the same complete logical-kernel selection.

    2. Deferred producer anchors: ``name = cb.wait()/.reserve()`` (or
       ``with cb.<m>() as name:``). The producer statement uses the
       block's resolved owner (from the produced block's downstream
       uses).

    3. Block-method anchors with deferred ownership: ``blk.pop()`` /
       ``blk.push()`` resolve to the block's planned logical kernel.

    Data-movement operations use the current callback's logical affinity.
    """

    def __init__(
        self,
        dfb_names: Set[str],
        selector_resolver: _KernelSelectorResolver,
        state: _AnalysisState,
        data_movement_kernels: FrozenSet[KernelSelector] = frozenset(
            {KernelKind.DATA_MOVEMENT}
        ),
        inferred_external_kernels: FrozenSet[KernelSelector] = frozenset(),
    ):
        self.dfb_names = dfb_names
        self._selector_resolver = selector_resolver
        self._state = state
        self._data_movement_kernels = data_movement_kernels
        self._inferred_external_kernels = inferred_external_kernels
        self.block_owners: Dict[str, KernelSelector] = {}
        self._producers: Dict[str, List[ast.stmt]] = {}
        self._callback_kernels: Dict[str, Set[KernelSelector]] = {}

    # --- entry ---

    def analyze(self, body: List[ast.stmt]) -> None:
        _validate_dfb_acquire_keywords(body, self.dfb_names)
        self._discover_producers(body)
        self._discover_callback_kernels(body)
        inferred_users, explicit_releases = _collect_block_ownership(
            body,
            set(self._producers.keys()),
            self.dfb_names,
            data_movement_kernels=self._data_movement_kernels,
            callback_kernels=self._callback_kernels,
            selector_resolver=self._selector_resolver,
            inferred_external_kernels=self._inferred_external_kernels,
        )
        self._resolve_transactions(inferred_users, explicit_releases)
        for stmt in body:
            self._annotate(stmt)

    # --- phase 1a: producer discovery (this scope only) ---

    def _discover_producers(self, stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            self._discover_in_stmt(stmt)

    def _discover_in_stmt(self, stmt: ast.stmt) -> None:
        prod = _bare_wait_assign_target(stmt, self.dfb_names)
        if prod is not None:
            block_name, _ = prod
            self._producers.setdefault(block_name, []).append(stmt)

        # `with cb.wait() as name:` form: register as producer; the
        # with stmt itself is the producer-anchor we'll tag.
        if isinstance(stmt, ast.With):
            for item in stmt.items:
                name = _with_item_block_name(item, self.dfb_names)
                if name:
                    self._producers.setdefault(name, []).append(stmt)

        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            self._discover_producers(stmt.body)
            self._discover_producers(stmt.orelse)
        elif isinstance(stmt, ast.With):
            self._discover_producers(stmt.body)
        # Do NOT descend into FunctionDef / AsyncFunctionDef: they are
        # their own scope and get their own tagger in Phase 3.

    # --- callback affinity discovery (this scope only) ---

    def _discover_callback_kernels(self, stmts: List[ast.stmt]) -> None:
        for node in _walk_skip_nested_fns_iter(stmts):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            kernel = _PIPENET_METHODS.get(func.attr)
            if kernel is None:
                continue
            if node.args and isinstance(node.args[0], ast.Name):
                self._callback_kernels.setdefault(node.args[0].id, set()).add(kernel)

    def _resolve_transactions(
        self,
        inferred_users: Dict[str, Set[KernelSelector]],
        explicit_releases: Dict[str, List[Tuple[ast.Call, KernelSelector]]],
    ) -> None:
        for block_name, producers in self._producers.items():
            inferred = frozenset(inferred_users.get(block_name, set()))
            explicit = frozenset(
                selector for _, selector in explicit_releases.get(block_name, [])
            )
            if len(inferred) > 1:
                raise _split_error(
                    producers[0],
                    f"DFB block {block_name!r} is used by multiple logical "
                    f"kernels ({_format_kernels(inferred)}); each "
                    ".reserve()/.wait() must resolve to exactly one kernel",
                )
            if len(explicit) > 1:
                raise _split_error(
                    producers[0],
                    f"DFB block {block_name!r} has releases assigned to "
                    f"multiple logical kernels ({_format_kernels(explicit)})",
                )
            if inferred and explicit and inferred != explicit:
                raise _split_error(
                    explicit_releases[block_name][0][0],
                    f"explicit {_format_kernels(explicit)} release ownership "
                    f"conflicts with inferred {_format_kernels(inferred)} "
                    f"ownership for DFB block {block_name!r}",
                )
            owners = inferred or explicit
            if not owners:
                raise _split_error(
                    producers[0],
                    f"result of a DFB wait/reserve bound to {block_name!r} has "
                    "no uses; the splitter cannot infer a logical kernel and "
                    "the release has no explicit kernel selector",
                )
            owner = next(iter(owners))
            self.block_owners[block_name] = owner
            self._state.transactions.append(
                DFBTransactionPlan(
                    block_name=block_name,
                    inferred_kernels=inferred,
                    explicit_kernels=explicit,
                    owner=owner,
                    source_line=getattr(producers[0], "lineno", 0),
                )
            )

    # --- statement selection ---

    def _annotate(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            children = list(stmt.body) + list(stmt.orelse)
            for child in children:
                self._annotate(child)
            child_kernels: Set[KernelSelector] = set()
            for child in children:
                child_kernels.update(self._descendant_anchor_kernels(child))
            anchored_kernels = self._stmt_anchor_kernels(stmt)
            if anchored_kernels is not None:
                self._state.control_header_selections[id(stmt)] = frozenset(
                    anchored_kernels
                )
                excluded = child_kernels - anchored_kernels
                if excluded:
                    raise _split_error(
                        stmt,
                        "selected control expression excludes logical kernels "
                        f"used by its body ({_format_kernels(excluded)})",
                    )
                child_kernels.update(anchored_kernels)
            if child_kernels:
                self._state.select(stmt, child_kernels)
            return
        if isinstance(stmt, ast.With):
            kernels = self._with_kernels(stmt)
            if kernels is not None:
                if len(kernels) != 1:
                    raise _split_error(
                        stmt,
                        "DFB acquire statement resolves to multiple logical "
                        f"kernels ({_format_kernels(kernels)}); each "
                        ".reserve()/.wait() must resolve to exactly one kernel",
                    )
                self._state.select(stmt, kernels)
            for child in stmt.body:
                self._annotate(child)
            if kernels is not None:
                self._validate_with_body_selection(stmt, kernels)
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            callback_kernels = frozenset(
                self._callback_kernels.get(stmt.name, self._data_movement_kernels)
            )
            inferred_external_kernels = (
                callback_kernels
                if stmt.name in self._callback_kernels
                else self._inferred_external_kernels
            )
            inner = _AnchorPlanner(
                dfb_names=self.dfb_names,
                selector_resolver=self._selector_resolver,
                state=self._state,
                data_movement_kernels=callback_kernels,
                inferred_external_kernels=inferred_external_kernels,
            )
            inner.analyze(stmt.body)
            return

        prod = _bare_wait_assign_target(stmt, self.dfb_names)
        if prod is not None:
            block_name, dfb_name = prod
            owner = self.block_owners.get(block_name)
            if owner is None:
                raise _split_error(
                    stmt,
                    f"result of {dfb_name}.wait()/.reserve() bound to "
                    f"{block_name!r} has no logical-kernel owner",
                )
            self._state.select(stmt, {owner})
            return

        copy_target = _assigned_copy_target(stmt)
        if copy_target is not None:
            raise _split_error(
                stmt,
                f"assigned transfer handle '{copy_target}' is not supported "
                "yet; write `ttl.copy(...).wait()` as a single statement "
                "instead",
            )

        kernels = self._stmt_anchor_kernels(stmt)
        if kernels is not None:
            self._state.select(stmt, kernels)

    def _descendant_anchor_kernels(self, statement: ast.stmt) -> Set[KernelSelector]:
        kernels: Set[KernelSelector] = set()
        pending = [statement]
        while pending:
            nested_statement = pending.pop()
            kernels.update(
                self._state.anchor_selections.get(
                    id(nested_statement),
                    frozenset(),
                )
            )
            if isinstance(nested_statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            pending.extend(
                child
                for child in ast.iter_child_nodes(nested_statement)
                if isinstance(child, ast.stmt)
            )
        return kernels

    def _with_kernels(self, stmt: ast.With) -> Optional[Set[KernelSelector]]:
        merged: Set[KernelSelector] = set()
        any_dfb = False
        for item in stmt.items:
            name = _with_item_block_name(item, self.dfb_names)
            if name is None:
                continue
            any_dfb = True
            owner = self.block_owners.get(name)
            if owner is not None:
                merged.add(owner)
        return merged if any_dfb else None

    def _validate_with_body_selection(
        self,
        stmt: ast.With,
        acquire_kernels: Set[KernelSelector],
    ) -> None:
        for nested_statement in _walk_statements(stmt.body):
            nested_kernels = self._state.anchor_selections.get(id(nested_statement))
            if nested_kernels is None or nested_kernels.issubset(acquire_kernels):
                continue
            raise _split_error(
                nested_statement,
                "statement selects logical kernels "
                f"({_format_kernels(nested_kernels)}) outside its enclosing "
                "DFB acquire owner "
                f"({_format_kernels(acquire_kernels)})",
            )

    def _block_kernel_set(self, block_name: str) -> Set[KernelSelector]:
        owner = self.block_owners.get(block_name)
        return {owner} if owner is not None else set()

    def _anchor_kernels_in(self, root: ast.AST) -> Set[KernelSelector]:
        selections: List[FrozenSet[KernelSelector]] = []
        for node in _iter_skip_nested_fns(root):
            if isinstance(node, ast.Call):
                selection = self._classify_call(node)
                if selection is not None:
                    selections.append(selection)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                selections.append(frozenset({KernelKind.COMPUTE}))
        if not selections:
            return set()
        expected = selections[0]
        for selection in selections[1:]:
            if selection != expected:
                raise _split_error(
                    root,
                    "calls in one indivisible expression select different "
                    f"logical kernels ({_format_kernels(expected)} versus "
                    f"{_format_kernels(selection)}); split the calls into "
                    "separate statements",
                )
        return set(expected)

    def _stmt_anchor_kernels(self, stmt: ast.stmt) -> Optional[Set[KernelSelector]]:
        """Logical-kernel selection shared by one indivisible expression.

        ``None`` means the statement is control or scalar code and is cloned
        into every selected logical kernel.

        Lambdas are skipped because their data-movement operations inherit the
        enclosing PipeNet callback affinity.

        AugAssign on a known block remains a compute anchor even when the RHS
        has no call or matrix multiplication.

        The result is ``None`` if the stmt has no kernel-pinning anchor.
        """
        if isinstance(stmt, (ast.If, ast.While)):
            anchor_root = stmt.test
        elif isinstance(stmt, ast.For):
            anchor_root = stmt.iter
        else:
            anchor_root = stmt
        kernels = self._anchor_kernels_in(anchor_root)
        if kernels:
            return kernels
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            return self._block_kernel_set(stmt.target.id) or None
        return None

    def _classify_call(self, call: ast.Call) -> Optional[FrozenSet[KernelSelector]]:
        selection = _classify_ttl_call(
            call,
            self._data_movement_kernels,
            self._inferred_external_kernels,
            self._selector_resolver,
        )
        if selection is not None:
            return selection

        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in self.dfb_names:
                method = _DFB_DIRECT_METHODS.get(func.attr)
                if method is not None:
                    return _materialize_kernels(method, self._data_movement_kernels)
            if func.value.id in self._producers:
                method = _BLOCK_METHODS.get(func.attr)
                if isinstance(method, KernelKind):
                    return frozenset({method})
                if method is _Placement.DEFERRED:
                    owner = self.block_owners.get(func.value.id)
                    if owner is not None:
                        return frozenset({owner})
        return None


# ----- scalar liveness -----------------------------------------------------


def _copy_live_map(
    live: Mapping[str, Set[KernelSelector]],
) -> Dict[str, Set[KernelSelector]]:
    return {name: set(kernels) for name, kernels in live.items()}


def _merge_live_maps(
    *live_maps: Mapping[str, Set[KernelSelector]],
) -> Dict[str, Set[KernelSelector]]:
    merged: Dict[str, Set[KernelSelector]] = {}
    for live in live_maps:
        for name, kernels in live.items():
            merged.setdefault(name, set()).update(kernels)
    return merged


def _restrict_live_map(
    live: Mapping[str, Set[KernelSelector]],
    kernels: FrozenSet[KernelSelector],
) -> Dict[str, Set[KernelSelector]]:
    selected_kernels = set(kernels)
    restricted: Dict[str, Set[KernelSelector]] = {}
    for name, live_kernels in live.items():
        selected = set(live_kernels) & selected_kernels
        if selected:
            restricted[name] = selected
    return restricted


def _exclude_live_map(
    live: Mapping[str, Set[KernelSelector]],
    kernels: FrozenSet[KernelSelector],
) -> Dict[str, Set[KernelSelector]]:
    excluded_kernels = set(kernels)
    excluded: Dict[str, Set[KernelSelector]] = {}
    for name, live_kernels in live.items():
        retained = set(live_kernels) - excluded_kernels
        if retained:
            excluded[name] = retained
    return excluded


def _bound_names_in_target(target: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            names.update(_bound_names_in_target(element))
    elif isinstance(target, ast.Starred):
        names.update(_bound_names_in_target(target.value))
    return names


def _direct_bound_names(statement: ast.stmt) -> Set[str]:
    if isinstance(statement, ast.Assign):
        names: Set[str] = set()
        for target in statement.targets:
            names.update(_bound_names_in_target(target))
        return names
    if isinstance(statement, ast.AnnAssign):
        if statement.value is None:
            return set()
        return _bound_names_in_target(statement.target)
    if isinstance(statement, ast.AugAssign):
        return _bound_names_in_target(statement.target)
    return set()


def _loaded_names_in(root: ast.AST) -> Set[str]:
    return {
        node.id
        for node in _iter_skip_nested_fns(root)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


class _LambdaFreeLoadCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: Set[str] = set()
        self._bound_names: List[Set[str]] = []

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        inherited = self._bound_names[-1] if self._bound_names else set()
        self._bound_names.append(inherited)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self._bound_names[-1] = inherited | _nested_scope_bindings(node)
        self.visit(node.body)
        self._bound_names.pop()

    def visit_Name(self, node):
        if (
            self._bound_names
            and isinstance(node.ctx, ast.Load)
            and node.id not in self._bound_names[-1]
        ):
            self.names.add(node.id)


def _lambda_free_loaded_names(root: ast.AST) -> Set[str]:
    collector = _LambdaFreeLoadCollector()
    for node in ast.iter_child_nodes(root):
        collector.visit(node)
    return collector.names


def _direct_loaded_names(statement: ast.stmt) -> Set[str]:
    names = _loaded_names_in(statement)
    names.update(_lambda_free_loaded_names(statement))
    if isinstance(statement, ast.AugAssign):
        names.update(_bound_names_in_target(statement.target))
    return names


def _control_header_loaded_names(statement: ast.stmt) -> Set[str]:
    if isinstance(statement, (ast.If, ast.While)):
        return _loaded_names_in(statement.test)
    if isinstance(statement, ast.For):
        return _loaded_names_in(statement.iter)
    if isinstance(statement, ast.With):
        names: Set[str] = set()
        for item in statement.items:
            names.update(_loaded_names_in(item.context_expr))
        return names
    return set()


def _function_header_loaded_names(
    statement: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Set[str]:
    expressions: List[ast.expr] = list(statement.decorator_list)
    expressions.extend(statement.args.defaults)
    expressions.extend(
        default for default in statement.args.kw_defaults if default is not None
    )
    if statement.returns is not None:
        expressions.append(statement.returns)
    for argument in (
        list(statement.args.posonlyargs)
        + list(statement.args.args)
        + list(statement.args.kwonlyargs)
    ):
        if argument.annotation is not None:
            expressions.append(argument.annotation)
    if statement.args.vararg is not None:
        annotation = statement.args.vararg.annotation
        if annotation is not None:
            expressions.append(annotation)
    if statement.args.kwarg is not None:
        annotation = statement.args.kwarg.annotation
        if annotation is not None:
            expressions.append(annotation)
    names: Set[str] = set()
    for expression in expressions:
        names.update(_loaded_names_in(expression))
    return names


def _assigned_names_in(statements: List[ast.stmt]) -> Set[str]:
    names: Set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names.update(_direct_bound_names(statement))
        if isinstance(statement, ast.For):
            names.update(_bound_names_in_target(statement.target))
            names.update(_assigned_names_in(statement.body))
            names.update(_assigned_names_in(statement.orelse))
        elif isinstance(statement, (ast.If, ast.While)):
            names.update(_assigned_names_in(statement.body))
            names.update(_assigned_names_in(statement.orelse))
        elif isinstance(statement, ast.With):
            for item in statement.items:
                if item.optional_vars is not None:
                    names.update(_bound_names_in_target(item.optional_vars))
            names.update(_assigned_names_in(statement.body))
    return names


class _ScalarLivenessPlanner:
    """Preserve scalar definitions in every kernel that observes them."""

    def __init__(
        self,
        state: _AnalysisState,
        all_kernels: FrozenSet[KernelSelector],
    ):
        self._state = state
        self._all_kernels = all_kernels

    def analyze(self, body: List[ast.stmt]) -> None:
        self._analyze_body(body, {}, self._all_kernels)

    def _selection(
        self, statement: ast.stmt, parent_kernels: FrozenSet[KernelSelector]
    ) -> FrozenSet[KernelSelector]:
        return self._state.anchor_selections.get(id(statement), parent_kernels)

    def _analyze_body(
        self,
        body: List[ast.stmt],
        live_out: Mapping[str, Set[KernelSelector]],
        parent_kernels: FrozenSet[KernelSelector],
    ) -> Dict[str, Set[KernelSelector]]:
        live = _copy_live_map(live_out)
        for statement in reversed(body):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner_live = self._analyze_body(statement.body, {}, self._all_kernels)
                local_names = _nested_scope_bindings(statement)
                live = _merge_live_maps(
                    live,
                    {
                        name: kernels
                        for name, kernels in inner_live.items()
                        if name not in local_names
                    },
                )
                for name in _function_header_loaded_names(statement):
                    live.setdefault(name, set()).update(parent_kernels)
                continue
            if isinstance(statement, (ast.For, ast.While, ast.If)):
                live = self._analyze_control(statement, live, parent_kernels)
                continue
            if isinstance(statement, ast.With):
                live = self._analyze_with(statement, live, parent_kernels)
                continue
            live = self._analyze_statement(statement, live, parent_kernels)
        return live

    @staticmethod
    def _required_output_kernels(
        assigned_names: Set[str],
        live: Mapping[str, Set[KernelSelector]],
    ) -> Set[KernelSelector]:
        required: Set[KernelSelector] = set()
        for name in assigned_names:
            required.update(live.get(name, set()))
        return required

    def _analyze_control(
        self,
        statement: Union[ast.For, ast.While, ast.If],
        live: Dict[str, Set[KernelSelector]],
        parent_kernels: FrozenSet[KernelSelector],
    ) -> Dict[str, Set[KernelSelector]]:
        assigned_names = _assigned_names_in(statement.body + statement.orelse)
        required = self._required_output_kernels(assigned_names, live)
        header_kernels = self._state.control_header_selections.get(id(statement))
        if header_kernels is not None:
            excluded = required - set(header_kernels)
            if excluded:
                raise _split_error(
                    statement,
                    "selected control expression excludes logical kernels "
                    "that consume values defined by its body "
                    f"({_format_kernels(excluded)})",
                )
        if required:
            self._state.extend(statement, required)
        statement_kernels = self._selection(statement, parent_kernels)
        outside_live = _exclude_live_map(live, statement_kernels)
        branch_live_out = _restrict_live_map(live, statement_kernels)
        else_live = self._analyze_body(
            statement.orelse, branch_live_out, statement_kernels
        )
        if isinstance(statement, ast.If):
            body_live = self._analyze_body(
                statement.body, branch_live_out, statement_kernels
            )
            result = _merge_live_maps(outside_live, body_live, else_live)
        else:
            loop_exit_live = _merge_live_maps(branch_live_out, else_live)
            if isinstance(statement, ast.While):
                for name in _control_header_loaded_names(statement):
                    loop_exit_live.setdefault(name, set()).update(statement_kernels)
            loop_entry_live = _copy_live_map(loop_exit_live)
            while True:
                body_live = self._analyze_body(
                    statement.body,
                    loop_entry_live,
                    statement_kernels,
                )
                if isinstance(statement, ast.For):
                    for name in _bound_names_in_target(statement.target):
                        body_live.get(name, set()).difference_update(statement_kernels)
                next_loop_entry_live = _merge_live_maps(loop_exit_live, body_live)
                if next_loop_entry_live == loop_entry_live:
                    break
                loop_entry_live = next_loop_entry_live
            result = _merge_live_maps(outside_live, loop_entry_live)
        if not isinstance(statement, ast.While):
            for name in _control_header_loaded_names(statement):
                result.setdefault(name, set()).update(statement_kernels)
        return result

    def _analyze_with(
        self,
        statement: ast.With,
        live: Dict[str, Set[KernelSelector]],
        parent_kernels: FrozenSet[KernelSelector],
    ) -> Dict[str, Set[KernelSelector]]:
        optional_names: Set[str] = set()
        for item in statement.items:
            if item.optional_vars is not None:
                optional_names.update(_bound_names_in_target(item.optional_vars))
        assigned_names = _assigned_names_in(statement.body) | optional_names
        required = self._required_output_kernels(assigned_names, live)
        statement_kernels = self._selection(statement, parent_kernels)
        missing = required - set(statement_kernels)
        if missing and id(statement) in self._state.anchor_selections:
            raise _split_error(
                statement,
                "values defined inside a selected DFB acquisition are used "
                "by excluded logical kernels "
                f"({_format_kernels(missing)})",
            )
        body_live = self._analyze_body(
            statement.body,
            _restrict_live_map(live, statement_kernels),
            statement_kernels,
        )
        for item in reversed(statement.items):
            if item.optional_vars is not None:
                for name in _bound_names_in_target(item.optional_vars):
                    body_live.get(name, set()).difference_update(statement_kernels)
            for name in _loaded_names_in(item.context_expr):
                body_live.setdefault(name, set()).update(statement_kernels)
        result = _merge_live_maps(_exclude_live_map(live, statement_kernels), body_live)
        return result

    def _analyze_statement(
        self,
        statement: ast.stmt,
        live: Dict[str, Set[KernelSelector]],
        parent_kernels: FrozenSet[KernelSelector],
    ) -> Dict[str, Set[KernelSelector]]:
        statement_kernels = self._selection(statement, parent_kernels)
        anchored = id(statement) in self._state.anchor_selections
        for name in _direct_bound_names(statement):
            required = live.get(name, set())
            missing = required - set(statement_kernels)
            if missing and anchored:
                raise _split_error(
                    statement,
                    f"value {name!r} is produced for "
                    f"({_format_kernels(statement_kernels)}) but consumed by "
                    f"excluded logical kernels ({_format_kernels(missing)})",
                )
            required.difference_update(statement_kernels)
        for name in _direct_loaded_names(statement):
            live.setdefault(name, set()).update(statement_kernels)
        return live


# ----- split-plan application ---------------------------------------------


class _KernelKeywordStripper(ast.NodeTransformer):
    def __init__(self, block_names: Set[str]):
        self.block_names = block_names

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        is_release = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.block_names
            and node.func.attr in _DFB_RELEASE_METHODS
        )
        if _is_external_call(node) or is_release:
            node.keywords = [
                keyword for keyword in node.keywords if keyword.arg != _KERNEL_KEYWORD
            ]
        return node


def _walk_statements(statements: List[ast.stmt]):
    for statement in statements:
        yield statement
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                yield from _walk_statements([child])
            elif isinstance(child, ast.ExceptHandler):
                yield from _walk_statements(child.body)


def _validate_kernel_capacities(
    fn_def: ast.FunctionDef,
    kernels: Tuple[KernelSelector, ...],
    kernel_origins: Mapping[KernelSelector, ast.stmt],
    kernel_capacities: Optional[Mapping[KernelKind, int]],
) -> None:
    if kernel_capacities is None:
        return
    for kind in KernelKind:
        capacity = kernel_capacities.get(kind)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError(
                f"kernel capacity for {kind.value} must be a nonnegative integer"
            )
        selected = tuple(kernel for kernel in kernels if _selector_kind(kernel) == kind)
        required = len(selected)
        if required > capacity:
            diagnostic_node = max(
                (kernel_origins[kernel] for kernel in selected),
                key=lambda statement: getattr(statement, "lineno", 0),
                default=fn_def,
            )
            raise _split_error(
                diagnostic_node,
                _format_kernel_capacity_error(kind, selected, capacity),
            )


def _apply_split_plan(
    body: List[ast.stmt],
    kernel: KernelSelector,
    plan: SplitPlan,
) -> List[ast.stmt]:
    memo: Dict[int, object] = {}
    copy.deepcopy(body, memo)
    selections = {
        statement.statement_id: statement.kernels for statement in plan.statements
    }
    cloned = _prune_statement_list(body, kernel, selections, memo, insert_pass=True)
    stripper = _KernelKeywordStripper(
        {transaction.block_name for transaction in plan.transactions}
    )
    return [
        ast.fix_missing_locations(stripper.visit(statement)) for statement in cloned
    ]


def _prune_statement_list(
    originals: List[ast.stmt],
    kernel: KernelSelector,
    selections: Dict[int, FrozenSet[KernelSelector]],
    memo: Dict[int, object],
    insert_pass: bool,
) -> List[ast.stmt]:
    result = []
    for original in originals:
        if kernel not in selections[id(original)]:
            continue
        clone = memo[id(original)]
        assert isinstance(clone, ast.stmt)
        if isinstance(original, (ast.For, ast.While, ast.If)):
            clone.body = _prune_statement_list(
                original.body, kernel, selections, memo, insert_pass=True
            )
            clone.orelse = _prune_statement_list(
                original.orelse, kernel, selections, memo, insert_pass=False
            )
        elif isinstance(original, ast.With):
            clone.body = _prune_statement_list(
                original.body, kernel, selections, memo, insert_pass=True
            )
        elif isinstance(original, (ast.FunctionDef, ast.AsyncFunctionDef)):
            clone.body = _prune_statement_list(
                original.body, kernel, selections, memo, insert_pass=True
            )
        result.append(clone)
    if not result and insert_pass:
        return [ast.Pass()]
    return result


# ----- helpers --------------------------------------------------------------


def _assigned_copy_target(stmt: ast.stmt) -> Optional[str]:
    """Return the target of ``name = ttl.copy(...)``, otherwise None."""
    if not isinstance(stmt, ast.Assign):
        return None
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    value = stmt.value
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "ttl":
        return None
    if func.attr != "copy":
        return None
    return stmt.targets[0].id


def _validate_dfb_acquire_keywords(
    statements: List[ast.stmt], dfb_names: Set[str]
) -> None:
    """Reject placement selectors on DFB acquisition operations."""
    for statement in statements:
        for node in _iter_skip_nested_fns(statement):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _DFB_PRODUCING_METHODS:
                continue
            if not isinstance(func.value, ast.Name) or func.value.id not in dfb_names:
                continue
            if _kernel_keyword(node) is not None:
                raise _split_error(
                    node,
                    f"kernel= is not supported on DFB {func.attr}(); "
                    "select release ownership on push() or pop()",
                )


def _bare_wait_assign_target(
    stmt: ast.stmt, dfb_names: Set[str]
) -> Optional[Tuple[str, str]]:
    """If ``stmt`` is ``name = <dfb>.wait()`` or ``name = <dfb>.reserve()``,
    return ``(block_name, dfb_name)``. Otherwise None.
    """
    if not isinstance(stmt, ast.Assign):
        return None
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    if not isinstance(stmt.value, ast.Call):
        return None
    func = stmt.value.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in _DFB_PRODUCING_METHODS:
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if func.value.id not in dfb_names:
        return None
    return stmt.targets[0].id, func.value.id


def _with_item_block_name(item: ast.withitem, dfb_names: Set[str]) -> Optional[str]:
    """If ``item`` is ``<dfb>.wait()/reserve() as name``, return ``name``."""
    ctx = item.context_expr
    if not isinstance(ctx, ast.Call):
        return None
    func = ctx.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in _DFB_PRODUCING_METHODS:
        return None
    if not isinstance(func.value, ast.Name):
        return None
    if func.value.id not in dfb_names:
        return None
    if not isinstance(item.optional_vars, ast.Name):
        return None
    return item.optional_vars.id


def _expr_root_name(node) -> Optional[str]:
    """Leftmost Name id of an expression, or None."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
            continue
        if isinstance(node, ast.Subscript):
            node = node.value
            continue
        if isinstance(node, ast.Call):
            node = node.func
            continue
        return None


def _format_kernels(kernels) -> str:
    return ", ".join(
        _format_selector(kernel) for kernel in sorted(kernels, key=_selector_sort_key)
    )


def _split_error(node, msg: str) -> ValueError:
    line = getattr(node, "lineno", "?")
    return ValueError(f"@ttl.operation split: {msg} (line {line})")


# ----- block use collection (shadow-aware) ---------------------------------


def _iter_skip_nested_fns(root):
    """Yield every AST node in ``root`` without crossing into nested
    FunctionDef / AsyncFunctionDef / Lambda scopes.
    """
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _walk_skip_nested_fns_iter(stmts: List[ast.stmt]):
    """Same as ``_iter_skip_nested_fns`` but over a stmt list."""
    for stmt in stmts:
        yield from _iter_skip_nested_fns(stmt)


def _nested_scope_bindings(fn_node) -> Set[str]:
    """Names in a nested scope that do not resolve in its enclosing scope."""
    locals_: Set[str] = set()
    args = fn_node.args
    for arg_list in (args.args, args.posonlyargs, args.kwonlyargs):
        for a in arg_list:
            locals_.add(a.arg)
    if args.vararg:
        locals_.add(args.vararg.arg)
    if args.kwarg:
        locals_.add(args.kwarg.arg)

    if isinstance(fn_node, ast.Lambda):
        return locals_

    global_names: Set[str] = set()
    nonlocal_names: Set[str] = set()
    for stmt in fn_node.body:
        for sub in _iter_skip_nested_fns(stmt):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                locals_.add(sub.name)
            elif isinstance(sub, ast.Assign):
                for target in sub.targets:
                    locals_.update(_bound_names_in_target(target))
            elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                locals_.update(_bound_names_in_target(sub.target))
            elif isinstance(sub, ast.For):
                locals_.update(_bound_names_in_target(sub.target))
            elif isinstance(sub, ast.With):
                for item in sub.items:
                    if item.optional_vars is not None:
                        locals_.update(_bound_names_in_target(item.optional_vars))
            elif isinstance(sub, ast.Global):
                global_names.update(sub.names)
            elif isinstance(sub, ast.Nonlocal):
                nonlocal_names.update(sub.names)
    return (locals_ - nonlocal_names) | global_names


def _collect_block_ownership(
    stmts: List[ast.stmt],
    block_names: Set[str],
    dfb_names: Set[str],
    data_movement_kernels: FrozenSet[KernelSelector],
    callback_kernels: Mapping[str, Set[KernelSelector]],
    selector_resolver: _KernelSelectorResolver,
    inferred_external_kernels: FrozenSet[KernelSelector],
) -> Tuple[
    Dict[str, Set[KernelSelector]],
    Dict[str, List[Tuple[ast.Call, KernelSelector]]],
]:
    """Collect inferred block uses and explicit release ownership separately."""
    inferred_users: Dict[str, Set[KernelSelector]] = {
        name: set() for name in block_names
    }
    explicit_releases: Dict[str, List[Tuple[ast.Call, KernelSelector]]] = {
        name: [] for name in block_names
    }

    def record_call_args(
        node: ast.Call,
        kernels: FrozenSet[KernelSelector],
        visible: Set[str],
    ) -> None:
        for arg in node.args:
            root = _expr_root_name(arg)
            if root in visible:
                inferred_users[root].update(kernels)
        for kw in node.keywords:
            root = _expr_root_name(kw.value)
            if root in visible:
                inferred_users[root].update(kernels)

    def visit(
        node,
        visible: Set[str],
        current_data_movement_kernels: FrozenSet[KernelSelector],
        current_external_kernels: FrozenSet[KernelSelector],
    ):
        if isinstance(node, ast.Call):
            func = node.func
            sub_data_movement_kernels = current_data_movement_kernels
            sub_external_kernels = current_external_kernels
            block_method_selection: Optional[FrozenSet[KernelSelector]] = None
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                receiver = func.value.id
                method = func.attr
                if receiver in visible and method in _DFB_RELEASE_METHODS:
                    explicit = selector_resolver.resolve_release(node)
                    if explicit is not None:
                        explicit_releases[receiver].append((node, explicit))
                    for arg in node.args:
                        visit(
                            arg,
                            visible,
                            sub_data_movement_kernels,
                            sub_external_kernels,
                        )
                    for keyword in node.keywords:
                        visit(
                            keyword.value,
                            visible,
                            sub_data_movement_kernels,
                            sub_external_kernels,
                        )
                    return
                if receiver in visible and method == "store":
                    inferred_users[receiver].add(KernelKind.COMPUTE)
                    block_method_selection = frozenset({KernelKind.COMPUTE})
                if method in _PIPENET_METHODS:
                    callback_kernel = _PIPENET_METHODS[method]
                    sub_data_movement_kernels = frozenset({callback_kernel})
                    sub_external_kernels = sub_data_movement_kernels

            selection = _classify_ttl_call(
                node,
                current_data_movement_kernels,
                current_external_kernels,
                selector_resolver,
            )
            selection = selection or block_method_selection
            if selection is not None and not _is_external_call(node):
                record_call_args(node, selection, visible)
            for arg in node.args:
                visit(
                    arg,
                    visible,
                    sub_data_movement_kernels,
                    sub_external_kernels,
                )
            for kw in node.keywords:
                visit(
                    kw.value,
                    visible,
                    sub_data_movement_kernels,
                    sub_external_kernels,
                )
            if isinstance(func, ast.Attribute):
                visit(
                    func.value,
                    visible,
                    sub_data_movement_kernels,
                    sub_external_kernels,
                )
            return
        if isinstance(node, ast.BinOp):
            for side in (node.left, node.right):
                root = _expr_root_name(side)
                if root in visible:
                    inferred_users[root].add(KernelKind.COMPUTE)
        elif isinstance(node, ast.AugAssign):
            target = _expr_root_name(node.target)
            if target in visible:
                inferred_users[target].add(KernelKind.COMPUTE)
            rhs = _expr_root_name(node.value)
            if rhs in visible:
                inferred_users[rhs].add(KernelKind.COMPUTE)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    root = _expr_root_name(tgt.value)
                    if root in visible:
                        inferred_users[root].add(KernelKind.COMPUTE)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = visible - _nested_scope_bindings(node)
            selected_callback_kernels = callback_kernels.get(node.name)
            if selected_callback_kernels is None:
                nested_data_movement_kernels = current_data_movement_kernels
                nested_external_kernels = current_external_kernels
            else:
                nested_data_movement_kernels = frozenset(selected_callback_kernels)
                nested_external_kernels = nested_data_movement_kernels
            for child in node.body:
                visit(
                    child,
                    inner,
                    nested_data_movement_kernels,
                    nested_external_kernels,
                )
            return
        if isinstance(node, ast.Lambda):
            inner = visible - _nested_scope_bindings(node)
            visit(
                node.body,
                inner,
                current_data_movement_kernels,
                current_external_kernels,
            )
            return

        for child in ast.iter_child_nodes(node):
            visit(
                child,
                visible,
                current_data_movement_kernels,
                current_external_kernels,
            )

    for stmt in stmts:
        visit(
            stmt,
            block_names,
            data_movement_kernels,
            inferred_external_kernels,
        )
    return inferred_users, explicit_releases
