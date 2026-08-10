# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Duplicate-and-prune splitter for unified-body @ttl.operation kernels.

Walks the function body of a unified @ttl.operation kernel and produces
three stripped function ASTs: one each for the TRISC (compute), NCRISC
(default DM, receivers + non-pipe DM), and BRISC (pipe senders) threads.

Algorithm:

    trisc_body, ncrisc_body, brisc_body = three deepcopies of body

    Pass A (anchor tagging):
        Walk every Call/MatMult node and look it up in the op
        registry. Anchor stmts get an ``_ttl_threads`` attribute
        set to the frozenset of threads they belong on. Stmts that
        produce a block via ``cb.wait()`` / ``cb.reserve()`` are
        *deferred anchors*: their thread is the union of threads
        observed on the produced block's downstream anchor uses.

        DM ops are registered with the sentinel ``"dm"``; the tagger
        materializes that to a concrete thread (``"ncrisc"`` outside
        an if_src callback, ``"brisc"`` inside one). Pipe dispatch
        methods bind directly: ``if_src`` -> brisc, ``if_dst`` -> ncrisc.

    Pass B (per-side prune):
        For each side, walk the duplicated body. If a stmt has
        ``_ttl_threads`` and this side isn't in it, drop the stmt.
        Non-anchor stmts (control flow, scalar arithmetic, ttl.node,
        make_dataflow_buffer_like, make_tensor_backed_dfb, ...) stay on
        every side; MLIR DCE removes dead code per-thread later.

    Post-check:
        A DFB producer (``blk = cb.wait()`` / ``cb.reserve()``) must have
        users on exactly one thread. Splitting an acquire across threads
        would issue the synchronization operation more than once against
        the same DFB.

Unknown ``ttl.<name>(...)`` calls are a hard error. To add an op,
register it in ``_TTL_OPS`` with its thread.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ----- threads --------------------------------------------------------------


THREADS: Tuple[str, ...] = ("trisc", "ncrisc", "brisc")
DM_THREADS: Tuple[str, ...] = ("ncrisc", "brisc")


# ----- op registry ----------------------------------------------------------


# ttl.<name>(...) -> "trisc" | "dm" | "control".
#
# ``"dm"`` is a sentinel: at tagging time it is materialized to either
# "ncrisc" or "brisc" depending on whether the call sits inside an
# ``if_src`` callback (-> brisc) or anywhere else (-> ncrisc).
_TTL_OPS: Dict[str, str] = {
    # Data movement
    "copy": "dm",
    "element_read": "dm",
    "element_write": "dm",
    # Compute
    "fill": "trisc",
    "matmul": "trisc",
    "reduce_sum": "trisc",
    "reduce_max": "trisc",
    "broadcast": "trisc",
    "transpose": "trisc",
    "exp": "trisc",
    "mul": "trisc",
    "add": "trisc",
    "sub": "trisc",
    "div": "trisc",
    "recip": "trisc",
    "neg": "trisc",
    "sqrt": "trisc",
    "tanh": "trisc",
    "log": "trisc",
    "abs": "trisc",
    "relu": "trisc",
    "sign": "trisc",
    "sigmoid": "trisc",
    "gelu": "trisc",
    # Compile-time / scalar producers: duplicated, not anchored.
    "Pipe": "control",
    "PipeNet": "control",
    "make_dataflow_buffer_like": "control",
    "make_tensor_backed_dfb": "control",
    "node": "control",
    "raw_addr": "control",
    "grid_size": "control",
    "dims": "control",
    "cores": "control",
    "tile_index": "control",
    "signpost": "control",
}

# ttl.<ns>.<name>(...) -> thread, applies to every name in the namespace.
_TTL_NAMESPACES: Dict[str, str] = {
    "math": "trisc",
    "block": "trisc",
}

# <pipenet>.<method>(...) -> thread. The receiver is any Name (PipeNet
# object). ``if_src`` dispatches the sender side on BRISC; ``if_dst``
# the receiver side on NCRISC (sender = BRISC, receiver = NCRISC).
_PIPENET_METHODS: Dict[str, str] = {
    "if_src": "brisc",
    "if_dst": "ncrisc",
}

# <block>.<method>(...) where <block> is a known block. "trisc" pins
# the call; "deferred" pins it to the block's inferred thread.
_BLOCK_METHODS: Dict[str, str] = {
    "store": "trisc",
    "pop": "deferred",
    "push": "deferred",
}

# Methods on a DFB name that produce a block.
_DFB_PRODUCING_METHODS: Set[str] = {"wait", "reserve"}
_DFB_DIRECT_METHODS: Dict[str, str] = {"publish": "dm"}


# ----- shared call -> thread classification ---------------------------------


def _materialize_thread(op: Optional[str], dm_thread: str) -> Optional[str]:
    """Resolve a registry value to a concrete thread tag.

    ``"dm"`` becomes the scope's DM thread (ncrisc by default, brisc inside an
    if_src callback). Anything that is not a concrete thread (``"control"`` and
    compile-time producers) returns None, since it does not pin a thread.
    """
    if op == "dm":
        return dm_thread
    if op in THREADS:
        return op
    return None


def _classify_ttl_call(func: ast.expr, dm_thread: str) -> Optional[str]:
    """Thread a registry-driven call pins to, or None when the call is not a
    registry anchor (control op, bare-name call, block / DFB method, or a
    chained call).

    Handles ``ttl.<op>(...)``, ``ttl.<ns>.<name>(...)`` and
    ``<recv>.if_src/if_dst(...)``, raising on an unknown ``ttl.<...>`` form so a
    new op must be registered rather than silently mis-split. This is the
    single source of the call->thread decision; both anchor tagging and
    block-use collection route through it.
    """
    # ttl.<name>(...) or <recv>.if_src/if_dst(...)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        recv = func.value.id
        name = func.attr
        if recv == "ttl":
            op = _TTL_OPS.get(name)
            if op is None:
                raise _split_error(
                    func,
                    f"unknown ttl.{name}(...) call; register it in "
                    f"atom_split._TTL_OPS with its thread "
                    f"('trisc', 'dm', or 'control')",
                )
            return _materialize_thread(op, dm_thread)
        if name in _PIPENET_METHODS:
            return _PIPENET_METHODS[name]
        return None

    # ttl.<ns>.<name>(...)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
        outer = func.value
        if isinstance(outer.value, ast.Name) and outer.value.id == "ttl":
            ns = outer.attr
            ns_thread = _TTL_NAMESPACES.get(ns)
            if ns_thread is None:
                raise _split_error(
                    func,
                    f"unknown ttl.{ns}.{func.attr}(...) call; register "
                    f"namespace 'ttl.{ns}' in atom_split._TTL_NAMESPACES",
                )
            return _materialize_thread(ns_thread, dm_thread)

    return None


# ----- public API -----------------------------------------------------------


@dataclass
class SplitResult:
    trisc_body: List[ast.stmt]
    ncrisc_body: List[ast.stmt]
    brisc_body: List[ast.stmt]

    def body_for(self, thread: str) -> List[ast.stmt]:
        return getattr(self, f"{thread}_body")


def split_function_body(
    fn_def: ast.FunctionDef,
    dfb_param_names: Set[str],
    local_dfb_names: Optional[Set[str]] = None,
) -> SplitResult:
    """Split a unified @ttl.operation body into trisc / ncrisc / brisc bodies.

    Args:
        fn_def: AST FunctionDef of the user's @ttl.operation function.
        dfb_param_names: parameter names annotated as ttl.DFB / ttl.DFB.Output.
        local_dfb_names: names of DFBs declared inside the body via a DFB
            factory. Treated as DFB receivers for wait/reserve recognition.
    """
    dfb_names = set(dfb_param_names) | (local_dfb_names or set())

    # Pass A: tag anchor stmts in-place on the original body.
    _AnchorTagger(dfb_names=dfb_names).tag(fn_def.body)

    # Pass B: deep-copy the body once per side; prune anchors that
    # don't apply on this side. Tag attributes survive deep-copy.
    bodies = {
        thread: _prune_body([copy.deepcopy(s) for s in fn_def.body], thread)
        for thread in THREADS
    }

    return SplitResult(
        trisc_body=bodies["trisc"],
        ncrisc_body=bodies["ncrisc"],
        brisc_body=bodies["brisc"],
    )


# ----- Pass A: anchor tagging ----------------------------------------------


class _AnchorTagger:
    """Walks the body and annotates anchor stmts with ``_ttl_threads``.

    Anchors fall into three categories:

    1. Direct anchors: any stmt whose AST subtree contains a registered
       ``ttl.<op>(...)`` call, a ``<pipenet>.if_src/if_dst(...)`` call,
       a DFB direct-method call (``dfb.publish()``), a block-method call
       (``blk.store(...)``), or a ``@`` MatMult.
       Thread = union of anchor threads found.

    2. Deferred producer anchors: ``name = cb.wait()/.reserve()`` (or
       ``with cb.<m>() as name:``). The producer stmt's thread is the
       block's inferred thread (from the produced block's downstream
       uses).

    3. Block-method anchors with deferred thread: ``blk.pop()`` /
       ``blk.push()`` resolve to the block's inferred thread.

    ``dm_thread`` is the concrete thread used for DM-classified ops in
    the current scope: ``"ncrisc"`` by default and ``"brisc"`` for the
    body of an ``if_src`` callback. The outer scope's Phase 1 detects
    which top-level FunctionDef names are passed as ``if_src(<name>)``;
    those FunctionDefs get an inner tagger with ``dm_thread="brisc"``.
    """

    def __init__(self, dfb_names: Set[str], dm_thread: str = "ncrisc"):
        self.dfb_names = dfb_names
        self._dm_thread = dm_thread
        # block_name -> frozenset of threads the block is used on.
        self.block_threads: Dict[str, FrozenSet[str]] = {}
        # block_name -> list of producer stmts (Assign / With) in this scope.
        self._producers: Dict[str, List[ast.stmt]] = {}
        # FunctionDef names in this scope used as the callback argument
        # to a ``<recv>.if_src(<Name>)`` call. Their inner bodies are
        # tagged with dm_thread="brisc".
        self._brisc_callbacks: Set[str] = set()

    # --- entry ---

    def tag(self, body: List[ast.stmt]) -> None:
        self._discover_producers(body)
        self._discover_brisc_callbacks(body)
        block_users = _collect_block_users(
            body,
            set(self._producers.keys()),
            self.dfb_names,
            dm_thread=self._dm_thread,
            brisc_callbacks=self._brisc_callbacks,
        )
        for name, threads in block_users.items():
            if threads:
                self.block_threads[name] = frozenset(threads)
        for stmt in body:
            self._annotate(stmt)
        self._check_producer_threads()

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

    # --- phase 1b: if_src callback discovery (this scope only) ---

    def _discover_brisc_callbacks(self, stmts: List[ast.stmt]) -> None:
        for node in _walk_skip_nested_fns_iter(stmts):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "if_src":
                continue
            if node.args and isinstance(node.args[0], ast.Name):
                self._brisc_callbacks.add(node.args[0].id)

    # --- phase 2: annotate every stmt with anchor threads ---

    def _annotate(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.For, ast.While, ast.If)):
            for child in list(stmt.body) + list(stmt.orelse):
                self._annotate(child)
            return
        if isinstance(stmt, ast.With):
            threads = self._with_threads(stmt)
            if threads is not None:
                if not threads:
                    names = [
                        _with_item_block_name(item, self.dfb_names)
                        for item in stmt.items
                    ]
                    names = [n for n in names if n]
                    raise _split_error(
                        stmt,
                        f"result of `with <dfb>.wait()/.reserve() as ...` "
                        f"({', '.join(names)}) has no uses; the splitter "
                        f"cannot pick a thread for the wait/reserve. "
                        f"Add a use inside the with-body or remove it.",
                    )
                if len(threads) != 1:
                    raise _split_error(
                        stmt,
                        "DFB acquire statement resolves to multiple threads "
                        f"({_format_threads(threads)}); each .reserve()/.wait() "
                        "must resolve to exactly one thread",
                    )
                _tag(stmt, threads)
            for child in stmt.body:
                self._annotate(child)
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner_dm = (
                "brisc" if stmt.name in self._brisc_callbacks else self._dm_thread
            )
            inner = _AnchorTagger(dfb_names=self.dfb_names, dm_thread=inner_dm)
            inner.tag(stmt.body)
            return

        # Producer Assign: thread = block's inferred thread.
        prod = _bare_wait_assign_target(stmt, self.dfb_names)
        if prod is not None:
            block_name, dfb_name = prod
            threads = self._block_thread_set(block_name)
            if not threads:
                raise _split_error(
                    stmt,
                    f"result of {dfb_name}.wait()/.reserve() bound to "
                    f"'{block_name}' has no uses; the splitter cannot pick "
                    f"a thread for the wait/reserve. Add a use of "
                    f"'{block_name}' (e.g. a no-op store into another CB) "
                    f"or remove the producer.",
                )
            _tag(stmt, threads)
            return

        copy_target = _assigned_copy_target(stmt)
        if copy_target is not None:
            raise _split_error(
                stmt,
                f"assigned transfer handle '{copy_target}' is not supported "
                "yet; write `ttl.copy(...).wait()` as a single statement "
                "instead",
            )

        threads = self._stmt_anchor_threads(stmt)
        if threads is not None:
            if len(threads) != 1:
                raise _split_error(
                    stmt,
                    "executable statement is pinned to multiple threads "
                    f"({_format_threads(threads)}); split the compute and "
                    "data movement work into separate statements",
                )
            _tag(stmt, threads)

    def _with_threads(self, stmt: ast.With) -> Optional[Set[str]]:
        """Thread set for a ``with cb.<wait|reserve>() as blk:`` form,
        unioned across all DFB-sync withitems. ``None`` if the With has
        no DFB-sync items (it's then a plain control-flow With).
        """
        merged: Set[str] = set()
        any_dfb = False
        for item in stmt.items:
            name = _with_item_block_name(item, self.dfb_names)
            if name is None:
                continue
            any_dfb = True
            merged |= self._block_thread_set(name)
        return merged if any_dfb else None

    def _block_thread_set(self, block_name: str) -> Set[str]:
        return set(self.block_threads.get(block_name, frozenset()))

    def _anchor_threads_in(self, root: ast.AST) -> Set[str]:
        """Return all concrete thread anchors in ``root``."""
        threads: Set[str] = set()
        for node in _iter_skip_nested_fns(root):
            if isinstance(node, ast.Call):
                thread = self._classify_call(node)
                if thread in THREADS:
                    threads.add(thread)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                threads.add("trisc")
        return threads

    def _stmt_anchor_threads(self, stmt: ast.stmt) -> Optional[Set[str]]:
        """Union of anchor threads found in the stmt's subtree. ``None``
        if the stmt has no thread-pinning anchor (control / scalar /
        registered compile-time op); those stay duplicated on all sides.

        Lambdas inside the stmt are deliberately skipped: their body's
        DM ops are dispatched by the enclosing ``<recv>.if_src/if_dst``
        call, whose own classification already pins the stmt to the
        right thread. Including the lambda body would double-count an
        ``if_src(lambda: ttl.copy(...))`` as both brisc (from if_src)
        and ncrisc (from ttl.copy), forcing duplication.

        AugAssign on a known block is still a compute anchor even when
        the RHS has no Call / MatMult (e.g. ``blk += scalar``).
        """
        threads = self._anchor_threads_in(stmt)
        if threads:
            return threads
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            return self._block_thread_set(stmt.target.id) or None
        return None

    def _classify_call(self, call: ast.Call) -> Optional[str]:
        """Return the thread of a Call ("trisc"/"ncrisc"/"brisc"), or None if
        it isn't an anchor at all (e.g. ``range(...)``, ``int(...)``,
        ``.wait()`` on a call result). Raises on a ``ttl.<unknown>(...)`` form.

        Delegates the registry decision (ttl ops, namespaces, pipe dispatch)
        to ``_classify_ttl_call``; resolves ``<block>.store/pop/push`` here,
        since deferred block methods need this scope's inferred block threads.
        """
        thread = _classify_ttl_call(call.func, self._dm_thread)
        if thread is not None:
            return thread

        # <block>.<method>(...) where <block> is a producer in this scope.
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in self.dfb_names:
                method = _DFB_DIRECT_METHODS.get(func.attr)
                if method is not None:
                    return _materialize_thread(method, self._dm_thread)
            if func.value.id in self._producers:
                method = _BLOCK_METHODS.get(func.attr)
                if method == "trisc":
                    return "trisc"
                if method == "deferred":
                    bt = self.block_threads.get(func.value.id)
                    if bt and len(bt) == 1:
                        return next(iter(bt))
        return None

    # --- post-check: every DFB acquire belongs to exactly one thread ---

    def _check_producer_threads(self) -> None:
        for name, stmts in self._producers.items():
            threads = self.block_threads.get(name, frozenset())
            if len(threads) > 1:
                raise _split_error(
                    stmts[0],
                    f"DFB block '{name}' is used on multiple threads "
                    f"({_format_threads(threads)}). Its .reserve()/.wait() "
                    f"acquire must resolve to exactly one thread; use a "
                    f"separate acquired block for each thread.",
                )


# ----- Pass B: prune --------------------------------------------------------


def _prune_body(body: List[ast.stmt], side: str) -> List[ast.stmt]:
    """Walk a deep-copied body and drop anchor stmts not for ``side``.
    Empty control-flow bodies are filled with ``pass`` (AST well-formedness).
    """
    result: List[ast.stmt] = []
    for stmt in body:
        kept = _prune_stmt(stmt, side)
        if kept is not None:
            result.append(kept)
    return result if result else [ast.Pass()]


def _prune_stmt(stmt: ast.stmt, side: str) -> Optional[ast.stmt]:
    threads = getattr(stmt, "_ttl_threads", None)
    if threads is not None and side not in threads:
        return None
    if isinstance(stmt, ast.For):
        stmt.body = _prune_body(stmt.body, side)
        stmt.orelse = _prune_orelse(stmt.orelse, side)
    elif isinstance(stmt, ast.While):
        stmt.body = _prune_body(stmt.body, side)
        stmt.orelse = _prune_orelse(stmt.orelse, side)
    elif isinstance(stmt, ast.If):
        stmt.body = _prune_body(stmt.body, side)
        stmt.orelse = _prune_orelse(stmt.orelse, side)
    elif isinstance(stmt, ast.With):
        stmt.body = _prune_body(stmt.body, side)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        stmt.body = _prune_body(stmt.body, side)
    return stmt


def _prune_orelse(body: List[ast.stmt], side: str) -> List[ast.stmt]:
    """Prune an orelse body without inserting a placeholder ``pass``.

    Python treats an empty orelse as 'no else clause'.
    """
    result: List[ast.stmt] = []
    for stmt in body:
        kept = _prune_stmt(stmt, side)
        if kept is not None:
            result.append(kept)
    return result


# ----- helpers --------------------------------------------------------------


def _tag(stmt: ast.stmt, threads: Set[str]) -> None:
    """Annotate a stmt with its anchor thread set. ``frozenset()`` means
    'drop on every side'.
    """
    stmt._ttl_threads = frozenset(threads)


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


def _format_threads(threads) -> str:
    """Stable, user-facing rendering of a concrete thread collection."""
    return ", ".join(thread.upper() for thread in THREADS if thread in threads)


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


def _local_bindings(fn_node) -> Set[str]:
    """Names locally bound inside a FunctionDef/AsyncFunctionDef/Lambda."""
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

    for stmt in fn_node.body:
        for sub in _iter_skip_nested_fns(stmt):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                locals_.add(sub.name)
            elif isinstance(sub, ast.Assign):
                for t in sub.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            locals_.add(n.id)
            elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                if isinstance(sub.target, ast.Name):
                    locals_.add(sub.target.id)
            elif isinstance(sub, ast.For):
                if isinstance(sub.target, ast.Name):
                    locals_.add(sub.target.id)
                elif isinstance(sub.target, ast.Tuple):
                    for elt in sub.target.elts:
                        if isinstance(elt, ast.Name):
                            locals_.add(elt.id)
            elif isinstance(sub, ast.With):
                for item in sub.items:
                    if isinstance(item.optional_vars, ast.Name):
                        locals_.add(item.optional_vars.id)
    return locals_


def _collect_block_users(
    stmts: List[ast.stmt],
    block_names: Set[str],
    dfb_names: Set[str],
    dm_thread: str = "ncrisc",
    brisc_callbacks: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """For each block name, the set of threads ("trisc"/"ncrisc"/"brisc")
    on which it has an anchor-relevant use. Walks nested functions and
    lambdas with shadow awareness: a re-bound name inside a callback
    hides the outer name.

    ``dm_thread`` is the concrete thread for DM-classified ops in the
    current scope (default "ncrisc"). When the walker descends into a
    callback recognized as an if_src dispatch target (a FunctionDef whose
    name is in ``brisc_callbacks``, or the lambda/named-arg of an
    ``<recv>.if_src(...)`` call), it switches to "brisc". Descent into
    ``<recv>.if_dst(...)`` arguments forces "ncrisc".

    An anchor-relevant use:
      - argument to a ttl.* / pipenet method call -> that call's thread
      - receiver of ``.store(...)`` -> trisc
      - receiver of ``.pop()``/``.push()`` -> does not pin (sync helper)
      - operand of MatMult ``@`` or any other BinOp -> trisc
      - target of an AugAssign -> trisc
    """
    brisc_callbacks = brisc_callbacks or set()
    users: Dict[str, Set[str]] = {n: set() for n in block_names}

    def record_call_args(node: ast.Call, thread: str, visible: Set[str]) -> None:
        for arg in node.args:
            root = _expr_root_name(arg)
            if root in visible:
                users[root].add(thread)
        for kw in node.keywords:
            root = _expr_root_name(kw.value)
            if root in visible:
                users[root].add(thread)

    def visit(node, visible, dm):
        if isinstance(node, ast.Call):
            func = node.func
            thread = _classify_ttl_call(func, dm)
            sub_dm = dm
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                recv = func.value.id
                method = func.attr
                if thread is None and recv in visible and method == "store":
                    users[recv].add("trisc")
                    thread = "trisc"
                if method in _PIPENET_METHODS:
                    sub_dm = _PIPENET_METHODS[method]
            if thread is not None:
                record_call_args(node, thread, visible)
            for arg in node.args:
                visit(arg, visible, sub_dm)
            for kw in node.keywords:
                visit(kw.value, visible, sub_dm)
            # Recurse into the callee so a chained call such as
            # `ttl.copy(blk, pipe).wait()` still records `blk`'s use in the
            # inner call (the inner call is func.value, not an arg/keyword).
            if isinstance(func, ast.Attribute):
                visit(func.value, visible, sub_dm)
            return
        if isinstance(node, ast.BinOp):
            for side in (node.left, node.right):
                root = _expr_root_name(side)
                if root in visible:
                    users[root].add("trisc")
        elif isinstance(node, ast.AugAssign):
            target = _expr_root_name(node.target)
            if target in visible:
                users[target].add("trisc")
            rhs = _expr_root_name(node.value)
            if rhs in visible:
                users[rhs].add("trisc")
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    root = _expr_root_name(tgt.value)
                    if root in visible:
                        users[root].add("trisc")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = visible - _local_bindings(node)
            sub_dm = "brisc" if node.name in brisc_callbacks else dm
            for child in node.body:
                visit(child, inner, sub_dm)
            return
        if isinstance(node, ast.Lambda):
            inner = visible - _local_bindings(node)
            visit(node.body, inner, dm)
            return

        for child in ast.iter_child_nodes(node):
            visit(child, visible, dm)

    for stmt in stmts:
        visit(stmt, block_names, dm_thread)
    return users
