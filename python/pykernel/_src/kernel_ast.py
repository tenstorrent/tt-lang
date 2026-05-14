# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file was copied from tt-mlir/tools/pykernel/_src/kernel_ast.py
# and cleaned up to remove unused code (TTKernelCompiler) and fix i32->i64.

import ast
import inspect

from ttl.dialects import arith, emitc, func, memref, scf
from ttl.ir import *

from .base_ast import PyKernelAstBase
from .kernel_types import ClassRegistry
from .utils import _cast, _get_type_str


def _extract_target_names(target):
    """Names bound by a single assignment target, supporting nested tuples
    and starred unpacking. Subscript/Attribute targets bind storage, not
    variables, and are skipped."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _extract_target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _extract_target_names(target.value)


class _ScopedCollector(ast.NodeVisitor):
    """Base for collectors that walk a statement body without descending
    into nested function or lambda definitions."""

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_For(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_With(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node):
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)


class _ReadVariableCollector(_ScopedCollector):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _collect_read_variable_names(node):
    collector = _ReadVariableCollector()
    collector.visit(node)
    return collector.names


class _SelfRebindingLoopVarCollector(_ScopedCollector):
    """Names that are assigned in a loop body to an expression that reads
    them (`acc = acc + x`, `a, b = a + 1, b * 2`). Order of first
    appearance is preserved."""

    def __init__(self):
        self.names = []
        self._seen = set()

    def _add_if_self_rebinding(self, target, value):
        read_variable_names = _collect_read_variable_names(value)
        for name in _extract_target_names(target):
            if name not in read_variable_names or name in self._seen:
                continue
            self.names.append(name)
            self._seen.add(name)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            return
        self._add_if_self_rebinding(node.targets[0], node.value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._add_if_self_rebinding(node.target, node.value)


class _AssignedVariableCollector(_ScopedCollector):
    """Names assigned anywhere inside the visited body. Order of first
    appearance is preserved."""

    def __init__(self):
        self.names = []
        self._seen = set()

    def _add_target(self, target):
        for name in _extract_target_names(target):
            if name in self._seen:
                continue
            self.names.append(name)
            self._seen.add(name)

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            self._add_target(node.targets[0])

    def visit_AnnAssign(self, node):
        self._add_target(node.target)


def _get_single_result(value):
    if isinstance(value, OpView):
        if len(value.results) != 1:
            raise ValueError(
                f"Expected operation with exactly one result, got "
                f"{len(value.results)} from {value.operation.name}"
            )
        return value.result
    if isinstance(value, Operation):
        if len(value.results) != 1:
            raise ValueError(
                f"Expected operation with exactly one result, got "
                f"{len(value.results)} from {value.name}"
            )
        return value.results[0]
    return value


def _get_value_type(value):
    if hasattr(value, "type"):
        return value.type
    value = _get_single_result(value)
    if hasattr(value, "type"):
        return value.type
    return None


class TTCompilerBase(PyKernelAstBase):
    def __init__(self, name, kernel_type=None, *args, **kwargs):
        assert kernel_type in [
            None,
            "datamovement",
            "noc",
            "compute",
        ], "Invalid kernel type"
        self.supported_nodes = [
            # Variables
            ast.Name,
            ast.Load,
            ast.Store,
            # control-flow
            ast.If,
            ast.For,
            # Literals
            ast.Constant,
            # Expressions
            ast.Attribute,
            ast.Expr,
            ast.IfExp,
            ast.Call,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.Invert,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            # Subscripting
            ast.Subscript,
            ast.Attribute,
            ast.List,
            ast.Tuple,
            # Statements
            ast.Pass,
            ast.Assign,
            ast.AugAssign,
            ast.AnnAssign,
            # Function-and-class-definitions
            ast.Module,
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
        ]

        self.name = name
        try:
            from ttl.dialects._ods_common import get_default_loc_context

            default_context = get_default_loc_context()
        except ValueError:
            default_context = None
        self.ctx = default_context if default_context is not None else Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.symbol_tables = []
        self.module_symbol_table = None
        self.kernel_type = kernel_type

        self.args = args
        self.ct_args = {}
        self.rt_args = None

        for arg in args:
            if hasattr(arg, "value") and hasattr(arg, "key"):
                # This is a CompileTimeValue
                self.ct_args[arg.key] = arg.value

        # Get rid of appended metadata sent into compiler
        self.verbose = kwargs.get("_verbose", False)
        self.source_code = kwargs.get("_source_code", "")

    # Control Flow
    def _on_scope_exit(self):
        """Hook for subclasses to act before exiting a scoped body."""
        pass

    def visit_If(self, node):
        # NOTE: else-if blocks are not supported in SCF dialect
        if_cond = self.visit(node.test)
        cond_type = None

        if hasattr(if_cond, "result"):
            if_cond = if_cond.result

        if hasattr(if_cond, "type") and isinstance(if_cond.type, memref.MemRefType):
            if_cond = memref.LoadOp(
                if_cond, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
            cond_type = if_cond.type
        elif hasattr(if_cond, "type") and isinstance(if_cond.type, IntegerType):
            cond_type = if_cond.type
        elif isinstance(if_cond, arith.ConstantOp):
            cond_type = if_cond.type

        # Create C-Style comparison if cond_type is not None
        if cond_type is None or not isinstance(cond_type, IntegerType):
            raise ValueError("Cannot Compare Non-Integer Values")

        if cond_type.width != 1:
            # Turn into comparison to make sure value is not 0
            if_cond = arith.cmpi(
                arith.CmpIPredicate.ne, if_cond, arith.ConstantOp(cond_type, 0)
            )
        carried_var_names = self._get_if_carried_var_names(node)
        carried_initial_values = [
            _get_single_result(self._var_exists(var_name)[var_name])
            for var_name in carried_var_names
        ]
        carried_types = [_get_value_type(value) for value in carried_initial_values]

        if_exp = scf.IfOp(
            cond=if_cond,
            results_=carried_types,
            has_else=bool(node.orelse) or bool(carried_var_names),
        )

        self._on_scope_exit()
        with InsertionPoint(if_exp.then_block), Location.unknown():
            self._visit_if_region(node.body, carried_var_names, carried_initial_values)

        if node.orelse or carried_var_names:
            with InsertionPoint(if_exp.else_block), Location.unknown():
                self._visit_if_region(
                    node.orelse, carried_var_names, carried_initial_values
                )

        for var_name, result in zip(carried_var_names, if_exp.results):
            self._set_var(var_name, result)

    def _visit_if_region(self, stmts, carried_var_names, carried_initial_values):
        self.symbol_tables.append({})
        for stmt in stmts:
            self.visit(stmt)
        self._on_scope_exit()

        yield_values = []
        for var_name, initial_value in zip(carried_var_names, carried_initial_values):
            final_value = self.symbol_tables[-1].get(var_name, initial_value)
            if _get_value_type(final_value) != _get_value_type(initial_value):
                raise ValueError(
                    f"If-carried variable '{var_name}' changes type from "
                    f"{_get_value_type(initial_value)} to "
                    f"{_get_value_type(final_value)}"
                )
            yield_values.append(_get_single_result(final_value))
        scf.YieldOp(yield_values)
        self.symbol_tables.pop()

    def visit_For(self, node):
        assert node.iter.func.id == "range", "Only range() supported in for loops"

        if len(node.iter.args) == 1:
            lower_bound = arith.ConstantOp(IndexType.get(self.ctx), 0)
            upper_bound = self.visit(node.iter.args[0])
            step = arith.ConstantOp(IndexType.get(self.ctx), 1)
        elif len(node.iter.args) == 2:
            lower_bound = self.visit(node.iter.args[0])
            upper_bound = self.visit(node.iter.args[1])
            step = arith.ConstantOp(IndexType.get(self.ctx), 1)
        elif len(node.iter.args) == 3:
            lower_bound = self.visit(node.iter.args[0])
            upper_bound = self.visit(node.iter.args[1])
            step = self.visit(node.iter.args[2])

        if isinstance(lower_bound.type, memref.MemRefType):
            lower_bound = memref.LoadOp(
                lower_bound, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        if isinstance(upper_bound.type, memref.MemRefType):
            upper_bound = memref.LoadOp(
                upper_bound, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        if isinstance(step.type, memref.MemRefType):
            step = memref.LoadOp(
                step, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result

        # Cast all to index type for scf.for
        if not isinstance(lower_bound.type, IndexType):
            lower_bound = arith.IndexCastOp(IndexType.get(self.ctx), lower_bound).result
        if not isinstance(upper_bound.type, IndexType):
            upper_bound = arith.IndexCastOp(IndexType.get(self.ctx), upper_bound).result
        if not isinstance(step.type, IndexType):
            step = arith.IndexCastOp(IndexType.get(self.ctx), step).result

        if self.verbose:
            comment = self._get_source_comment_block(node)
            emitc.verbatim(comment, [])

        carried_var_names = self._get_loop_carried_var_names(node)
        carried_initial_values = [
            _get_single_result(self._var_exists(var_name)[var_name])
            for var_name in carried_var_names
        ]

        self._on_scope_exit()
        for_op = scf.ForOp(lower_bound, upper_bound, step, carried_initial_values)
        with InsertionPoint(for_op.body), Location.unknown():
            self.symbol_tables.append({})

            # Add the iterator into the symbol table.
            self._set_var(node.target.id, for_op.induction_variable)
            for var_name, iter_arg in zip(carried_var_names, for_op.inner_iter_args):
                self._set_var(var_name, iter_arg)

            for stmt in node.body:
                self.visit(stmt)
            self._on_scope_exit()
            yield_values = []
            for var_name, initial_value in zip(
                carried_var_names, carried_initial_values
            ):
                final_value = self.symbol_tables[-1][var_name]
                if _get_value_type(final_value) != _get_value_type(initial_value):
                    raise ValueError(
                        f"Loop-carried variable '{var_name}' changes type from "
                        f"{_get_value_type(initial_value)} to "
                        f"{_get_value_type(final_value)}"
                    )
                yield_values.append(_get_single_result(final_value))
            scf.YieldOp(yield_values)
            self.symbol_tables.pop()

        for var_name, result in zip(carried_var_names, for_op.results):
            self._set_var(var_name, result)

    def _get_loop_carried_var_names(self, node):
        collector = _SelfRebindingLoopVarCollector()
        for stmt in node.body:
            collector.visit(stmt)

        # A name is carried only if it already exists outside the loop;
        # otherwise it is loop-local and rebinding it does not need an
        # iter_arg. Type is not constrained: scf.for accepts any iter_arg
        # type, so tensor and scalar recurrences are both materialized.
        carried_var_names = []
        for var_name in collector.names:
            if var_name == node.target.id:
                continue
            if not self._var_exists(var_name):
                continue
            carried_var_names.append(var_name)
        return carried_var_names

    def _get_if_carried_var_names(self, node):
        collector = _AssignedVariableCollector()
        for stmt in node.body:
            collector.visit(stmt)
        for stmt in node.orelse:
            collector.visit(stmt)

        # Only names that exist outside the if are carried; fresh names
        # bound inside a branch stay branch-local.
        carried_var_names = []
        for var_name in collector.names:
            if not self._var_exists(var_name):
                continue
            carried_var_names.append(var_name)
        return carried_var_names

    # Statements
    def visit_Name(self, node):
        var_name = node.id

        # NOTE: some kernelops require passing return type as arg
        if var_name == "int":
            return IntegerType.get_signless(64, self.ctx)

        existing_var_table = self._var_exists(var_name)
        if existing_var_table:
            return existing_var_table[var_name]

        return None

    def visit_Assign(self, node):
        # Loosely support slice + tuple assignment for rt_args
        assert len(node.targets) == 1, "Only single assignments supported"
        if isinstance(node.targets[0], ast.Tuple):
            # Make sure that these are being assigned from rt_args
            if self.rt_args is None:
                raise NotImplementedError(
                    "Tuple Assignment except for rt_args not supported."
                )

            if (
                isinstance(node.value, ast.Subscript)
                and node.value.value.id == self.rt_args.arg
            ):
                _tuple = node.targets[0]
                _vars = [self.visit(elt) for elt in _tuple.elts]
                values = self.visit(node.value)
                if not isinstance(values, list):
                    raise ValueError(
                        f"Not enough values to unpack from rt_args slice (expected {len(_vars)}, got 1)"
                    )
                if len(values) != len(_vars):
                    raise ValueError(
                        f"Not enough values to unpack from rt_args slice (expected {len(_vars)}, got {len(values)})"
                    )
                # Since we are unpacking a tuple, types can't be assigned here.
                for i in range(len(_vars)):
                    self._set_var(_tuple.elts[i].id, values[i])

                # Exit out of function now
                return

        var = self.visit(node.targets[0])
        value = self.visit(node.value)

        # Handle Subscript Assignment here
        if isinstance(node.targets[0], ast.Subscript):
            # Var will contain a memref.LoadOp here, we can access the memref and write to it
            memref.StoreOp(value, var.memref, var.indices)
            return

        var_name = node.targets[0].id

        if hasattr(var, "type") and isinstance(var.type, MemRefType):
            memref.StoreOp(value, var, [arith.ConstantOp(IndexType.get(self.ctx), 0)])
        else:
            self._set_var(var_name, value)

    def visit_AnnAssign(self, node):
        # NOTE: TTKernel types can not be used with memrefs
        var = self.visit(node.target)
        value = self.visit(node.value)
        var_name = node.target.id

        # Check the annotation for array creation
        if isinstance(node.annotation, ast.List):
            # Syntax is [dtype, *shape]
            if not len(node.annotation.elts) >= 2 or not isinstance(
                node.annotation.elts[0], ast.Name
            ):
                raise ValueError(
                    "Array Initialization must follow [dtype, *shape] syntax."
                )
            # Use i64 for array element type
            var_type = IntegerType.get_signless(64, self.ctx)

            if all(isinstance(elt, ast.Constant) for elt in node.annotation.elts[1:]):
                # Strictly constant, easiest case.
                memref_type = MemRefType.get(
                    [elt.value for elt in node.annotation.elts[1:]], var_type
                )
                self._set_var(var_name, memref.alloca(memref_type, [], []))
                return
            else:
                raise NotImplementedError(
                    "Not possible to use dynamic dimensions in EmitC."
                )

        if hasattr(value, "type") and isinstance(value.type, MemRefType):
            raise ValueError(
                "Not allowed to AnnAssign to another AnnAssign'ed variable. Temporary fix is to just add 0 to the variable."
            )

        if not var:
            var_type = value.type
            memref_type = MemRefType.get([1], var_type)
            var = memref.alloca(memref_type, [], [])
            self._set_var(var_name, var)
        else:
            assert isinstance(var, MemRefType), "Can not AnnAssign to non-memref types"

        memref.StoreOp(value, var, [arith.ConstantOp(IndexType.get(self.ctx), 0)])

    def visit_AugAssign(self, node):
        target = self.visit(node.target)

        # Target must already be defined in the scope of the symbol table
        if not target:
            raise ValueError(
                "AugAssign can only Assign to values that have been defined"
            )

        value = self.visit(node.value)

        if not isinstance(target.type, memref.MemRefType):
            raise ValueError("Can not AugAssign to non-memref types")

        _target = memref.LoadOp(
            target, arith.ConstantOp(IndexType.get(self.ctx), 0)
        ).result

        # Determine the operation based on the type of AugAssign
        match node.op:
            case ast.Add():
                result = arith.AddIOp(_target, value)
            case ast.Sub():
                result = arith.SubIOp(_target, value)
            case ast.Mult():
                result = arith.MulIOp(_target, value)
            case _:
                raise NotImplementedError(
                    f"AugAssign operation {type(node.op).__name__} not supported"
                )

        # Store the result back to the target
        memref.StoreOp(result, target, [arith.ConstantOp(IndexType.get(self.ctx), 0)])

    # Function calls
    def visit_Call(self, node):
        def _load_func_arg(func_arg):
            if not func_arg:
                raise ValueError(
                    f"Function argument not found for {ast.unparse(node.func)}"
                )
            if hasattr(func_arg, "type") and isinstance(
                func_arg.type, memref.MemRefType
            ):
                func_arg = memref.LoadOp(
                    func_arg, arith.ConstantOp(IndexType.get(self.ctx), 0)
                )
            return func_arg

        if not isinstance(node.func, ast.Attribute):
            # print is special case to handle string formatting
            if node.func.id == "print":
                return self.visit_Print(node.args)

            # if not an Attribute, it's just a kernel api call.
            assert (
                node.func.id in self._fn_map
            ), f"Function {node.func.id} not supported"
            func = self._fn_map[node.func.id]
            args_as_attr = [False] * len(node.args)
            if type(func) is tuple:
                func, args_as_attr = func
            func_args = []
            assert len(node.args) == len(args_as_attr)
            for arg, as_attr in zip(node.args, args_as_attr):
                arg._ttkernel_as_attr = as_attr
                func_arg = _load_func_arg(self.visit(arg))
                func_args.append(func_arg)
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = _load_func_arg(self.visit(kw.value))
            return func(*func_args, **kwargs)  # type checking will occur downstream
        else:
            func_args = []
            for arg in node.args:
                func_arg = _load_func_arg(self.visit(arg))
                func_args.append(func_arg)
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = _load_func_arg(self.visit(kw.value))
            return self.visit(
                node.func, func_args=func_args, kwargs=kwargs
            )  # visit_Attribute

    def visit_Print(self, node):
        # Import ttkernel here to avoid circular import at module level
        from ttl.dialects import ttkernel

        fmt = ""
        argv = []
        for arg in node:
            # handles printing vars, eg: print(x)
            if isinstance(arg, ast.Name):
                fmt += "{} "
                argv.append(self.visit(arg))
            # handles printing constants, eg: print("hello world")
            elif isinstance(arg, ast.Constant):
                fmt += str(arg.value) + " "
            # handles printing format strings, eg: print("hello {}".format(x))
            elif isinstance(arg, ast.Call):
                fmt += arg.func.value.value + " "
                for arg in arg.args:
                    argv.append(self.visit(arg))
            else:
                raise NotImplementedError(
                    f"Print argument {type(arg).__name__} not supported"
                )

        fmt = fmt.strip() + "\\n"
        ttkernel.dprint(fmt, argv)

    # Expressions
    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit_BoolOp(self, node):
        values = [self.visit(arg) for arg in node.values]

        # Make sure that each of the values are booleans
        for i in range(len(values)):
            value = values[i]
            value_type = None
            if hasattr(value, "type") and isinstance(value.type, memref.MemRefType):
                value = memref.LoadOp(
                    value, arith.ConstantOp(IndexType.get(self.ctx), 0)
                ).result
                value_type = value.type
            elif hasattr(value, "type") and isinstance(value.type, IntegerType):
                value_type = value.type
            elif isinstance(value, arith.ConstantOp):
                value_type = value.type

            if value_type is None:
                raise ValueError(
                    "BoolOp values must be MemRef, ConstantOp, or IntegerType"
                )

            if not isinstance(value_type, IntegerType):
                raise ValueError(
                    "BoolOp values must be MemRef or ConstantOp of IntegerType"
                )

            if value_type.width != 1:
                # Set the value to 1 if not equal to 0, otherwise 0. This is the C-style way
                values[i] = arith.cmpi(
                    arith.CmpIPredicate.ne, value, arith.ConstantOp(value_type, 0)
                )

        # Chain all of the comparisons together
        def _match_bool_op(lhs, rhs):
            # We will know and assume LHS and RHS are booleans
            match (node.op):
                case ast.And():
                    return arith.andi(lhs, rhs)
                case ast.Or():
                    return arith.ori(lhs, rhs)
                case _:
                    raise NotImplementedError(f"BoolOp {node.op} not supported")

        # Atleast 2 Ops must exist in BoolOp
        chained_op = _match_bool_op(values[0], values[1])

        # Chain all of the remaining values
        for i in range(2, len(values)):
            chained_op = _match_bool_op(chained_op, values[i])

        return chained_op

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not lhs or not rhs:
            raise ValueError("Binary operands not found")

        # load variable if needed
        if isinstance(lhs, OpView):
            lhs = lhs.result

        if isinstance(rhs, OpView):
            rhs = rhs.result

        if hasattr(lhs, "type") and isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(
                lhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        if hasattr(rhs, "type") and isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(
                rhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result

        # Matmul: operands have different shapes (A[M,K] @ B[K,N]), so dispatch
        # before the elementwise type-matching cast.
        if isinstance(node.op, ast.MatMult):
            mlir_type = _get_type_str(lhs.type)
            fn = self._fn_map.get(
                f"{mlir_type}.__matmul__",
                lambda *a, **k: (_ for _ in ()).throw(
                    NotImplementedError("MatMult not implemented")
                ),
            )
            return fn(lhs, rhs)

        if lhs.type != rhs.type:
            rhs = _cast(rhs, lhs.type)
        assert lhs.type == rhs.type, f"{lhs.type} != {rhs.type}"
        mlir_type = _get_type_str(lhs.type)

        def qualified_or(attr, otherwise, *args, **kwargs):
            qualified_object_syntax = f"{mlir_type}.{attr}"
            fn = self._fn_map.get(qualified_object_syntax, otherwise)
            return fn(*args, **kwargs)

        def unimplemented(*args, **kwargs):
            raise NotImplementedError(f"{node.op} not implemented")

        match (node.op):
            case ast.Add():
                return qualified_or("__add__", arith.addi, lhs, rhs)
            case ast.Sub():
                return qualified_or("__sub__", arith.subi, lhs, rhs)
            case ast.Mult():
                return qualified_or("__mul__", arith.muli, lhs, rhs)
            case ast.Div():
                return qualified_or("__truediv__", unimplemented, lhs, rhs)
            case ast.FloorDiv():
                return qualified_or("__floordiv__", arith.divsi, lhs, rhs)
            case ast.Mod():
                return qualified_or("__mod__", arith.remsi, lhs, rhs)
            case ast.Pow():
                return qualified_or("__pow__", unimplemented, lhs, rhs)
            case ast.LShift():
                return qualified_or("__lshift__", arith.shli, lhs, rhs)
            case ast.RShift():
                return qualified_or("__rshift__", arith.shrsi, lhs, rhs)
            case ast.BitOr():
                return qualified_or("__or__", arith.ori, lhs, rhs)
            case ast.BitAnd():
                return qualified_or("__and__", arith.andi, lhs, rhs)
            case ast.BitXor():
                return qualified_or("__xor__", arith.xori, lhs, rhs)
            case _:
                raise NotImplementedError(
                    f"Binary operator {type(node.op).__name__} not implemented"
                )

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if not operand:
            raise ValueError("Unary operand not found")

        if isinstance(operand.type, memref.MemRefType):
            operand = memref.LoadOp(
                operand, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result

        match (node.op):
            # need to expose emitc for these unary operators
            case ast.USub():
                return emitc.UnaryMinusOp(operand.type, operand)
            case ast.UAdd():
                return emitc.UnaryPlusOp(operand.type, operand)
            case ast.Not():
                # Must return a 1-bit Signless Integer (bool)
                return emitc.logical_not(IntegerType.get_signless(1, self.ctx), operand)
            case ast.Invert():
                return emitc.bitwise_not(operand.type, operand)
            case _:
                raise NotImplementedError(
                    f"Unary operator {type(node.op).__name__} not implemented"
                )

    def visit_Compare(self, node):
        assert len(node.ops) == 1, "Only single operators supported"
        assert len(node.comparators) == 1, "Only single comparators supported"
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        if not lhs or not rhs:
            raise ValueError("Compare operands not found")

        if isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(
                lhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        if isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(
                rhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result

        if lhs.type != rhs.type:
            rhs = _cast(rhs, lhs.type)
        assert lhs.type == rhs.type, f"{lhs.type} != {rhs.type}"

        match (node.ops[0]):
            case ast.Eq():
                return arith.cmpi(arith.CmpIPredicate.eq, lhs, rhs)
            case ast.NotEq():
                return arith.cmpi(arith.CmpIPredicate.ne, lhs, rhs)
            case ast.Gt():
                return arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs)
            case ast.GtE():
                return arith.cmpi(arith.CmpIPredicate.sge, lhs, rhs)
            case ast.Lt():
                return arith.cmpi(arith.CmpIPredicate.slt, lhs, rhs)
            case ast.LtE():
                return arith.cmpi(arith.CmpIPredicate.sle, lhs, rhs)
            case _:
                raise NotImplementedError(
                    f"Compare operator {type(node.ops).__name__} not implemented"
                )

    def visit_Attribute(self, node, func_args=None, kwargs=None):
        if func_args is None:
            func_args = []
        if kwargs is None:
            kwargs = {}
        # Resolve the receiver: a named variable, a chained call result
        # (e.g., ttl.copy(...).wait()), or any other expression.
        mlir_value = self.visit(node.value)
        if mlir_value is None:
            receiver_src = ast.unparse(node.value)
            raise ValueError(
                f"cannot call .{node.attr}() on '{receiver_src}': "
                "expression does not produce a value"
            )

        # type name should be !ttkernel.* if it has attributes
        mlir_type = _get_type_str(mlir_value.type)
        qualified_object_syntax = f"{mlir_type}.{node.attr}"
        fn = self._fn_map.get(qualified_object_syntax, None)
        if fn is not None:
            return fn(mlir_value, *func_args, **kwargs)
        elif not mlir_type.startswith("!ttkernel."):
            receiver_name = (
                node.value.id if isinstance(node.value, ast.Name) else "<expr>"
            )
            raise ValueError(
                f"{receiver_name} is not a ttkernel type, thus can not have attributes."
            )
        # ignore the '!' at the start of the type name
        type_name = mlir_type[1:]

        if ClassRegistry.exists(type_name):
            # Instantiate class and call its emit_mlir method.
            func_args = [mlir_value] + func_args
            attr_class = ClassRegistry.get(type_name)()
            attr_class.emit_mlir(node.attr, func_args)
        else:
            receiver_name = (
                node.value.id if isinstance(node.value, ast.Name) else "<expr>"
            )
            raise ValueError(
                f"{receiver_name} has no attributes. Did you define a PyKernelAttributesBase subclass?"
            )
        return

    def visit_List(self, node):
        # Snoop List for nested loops and get size
        def snoop_list(node):
            result_arr = []
            result_shape = []
            sz = 0

            if any(isinstance(elt, ast.List) for elt in node.elts) and not all(
                isinstance(elt, ast.List) for elt in node.elts
            ):
                # The shape is not consistent, we will raise an error here:
                raise NotImplementedError("All nested arrays must be of same size.")

            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    tbl = self._var_exists(elt.id)
                    elt = tbl[elt.id]
                    if hasattr(elt, "type") and isinstance(elt.type, MemRefType):
                        if elt.type.rank > 1 or elt.type.shape[0] != 1:
                            raise NotImplementedError(
                                "Creating Arrays with Pre-Defined Nested Arrays Not Supported."
                            )
                    sz += 1
                    result_arr.append(
                        memref.LoadOp(
                            elt, arith.ConstantOp(IndexType.get(self.ctx), 0)
                        ).result
                    )
                elif isinstance(elt, ast.List):
                    size, arr = snoop_list(elt)
                    if not result_shape:
                        result_shape = size
                    elif size != result_shape:
                        raise NotImplementedError(
                            "All nested arrays must be of same size."
                        )
                    sz += 1
                    result_arr.append(arr)
                elif isinstance(elt, ast.Constant):
                    elt = self.visit(elt)
                    sz += 1
                    result_arr.append(elt.result)
                else:
                    elt = self.visit(elt)
                    if (
                        not hasattr(elt, "type")
                        or not isinstance(elt.type, IntegerType)
                        or elt.type.width != 64
                    ):
                        raise ValueError("Array element must be an i64 integer type")
                    result_arr.append(elt)
                    sz += 1
            # Collect the size and result_shape
            return ([sz] + result_shape), result_arr

        # Need to deal with nested loops, determine the shape from filled array
        shape, array = snoop_list(node)

        # empty sz catch case
        if shape == [0]:
            raise NotImplementedError(
                "Array object must be filled, otherwise use AnnAssign."
            )

        # Create the memref with i64 element type
        var_type = IntegerType.get_signless(64, self.ctx)
        memref_type = MemRefType.get(shape, var_type)
        var = memref.alloca(memref_type, [], [])

        # Populate the table
        def populate_list(arr, _idx=[]):
            nonlocal var
            for i, elt in enumerate(arr):
                idx = _idx + [arith.ConstantOp(IndexType.get(self.ctx), i)]
                if isinstance(elt, list):
                    populate_list(elt, idx)
                else:
                    memref.StoreOp(elt, var, idx)

        populate_list(array)

        return var

    def visit_Tuple(self, node):
        return tuple(map(self.visit, node.elts))

    # Literals
    def visit_Constant(self, node):
        as_attr = getattr(node, "_ttkernel_as_attr", False)
        op_constructor = IntegerAttr.get if as_attr else arith.ConstantOp
        if callable(as_attr):
            return as_attr(node)
        elif isinstance(node.value, bool):
            return op_constructor(IntegerType.get_signless(1, self.ctx), node.value)
        elif isinstance(node.value, int):
            return op_constructor(IntegerType.get_signless(64, self.ctx), node.value)
        else:
            raise NotImplementedError(
                f"constant type {type(node.value).__name__} not implemented"
            )
