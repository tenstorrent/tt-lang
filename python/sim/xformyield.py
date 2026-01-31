# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AST transformers for TTL code.

This module provides AST transformers that modify Python AST nodes
to enable cooperative multitasking and other program transformations.
"""

import ast
from typing import TYPE_CHECKING, Literal, Tuple, Union

if TYPE_CHECKING:
    from .cb import CircularBuffer
    from .copy import CopyTransaction

# Central registry of known generator-like functions exposed via ttl
KNOWN_TTL_GENERATORS: set[str] = {
    "if_src",  # PipeNet.if_src method
    "if_dst",  # PipeNet.if_dst method
}

# Type for values yielded by generator transformations
# The AST transformation converts:
# - cb.wait() to yield (cb, 'wait')
# - cb.reserve() to yield (cb, 'reserve')
# - tx.wait() to yield (tx, 'wait')
YieldedValue = Union[
    Tuple["CircularBuffer", Literal["wait", "reserve"]],
    Tuple["CopyTransaction", Literal["wait"]],
]

__all__ = [
    "YieldInserter",
    "YieldingFunctionMarker",
    "YieldFromInserter",
    "YieldedValue",
    "transform_wait_reserve_to_yield_ast",
    # Legacy alias for backward compatibility with tests
    "WaitReserveToYieldTransformer",
]


class YieldInserter(ast.NodeTransformer):
    """
    Stage 1: Insert yields before wait() and reserve() calls.

    Transforms:
        block = cb.wait()

    Into:
        yield (cb, 'wait')
        block = cb.wait()
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
        """Visit assignment statements and check for wait/reserve calls."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Pattern match on assignment value to detect wait/reserve calls
        match node.value:
            case ast.Call() as call if self._is_wait_or_reserve_call(call):
                result: list[ast.AST] = list(self._insert_yield_before(node))
                return result
            case _:
                return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        """Visit expression statements and check for wait/reserve calls."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Pattern match on expression value to detect wait/reserve calls
        match node.value:
            case ast.Call() as call if self._is_wait_or_reserve_call(call):
                result: list[ast.AST] = list(self._insert_yield_before(node))
                return result
            case _:
                return node

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.AST]:
        """Visit with statements and check for wait/reserve calls in context expressions.

        Transforms:
            with cb.wait() as blk:
                ...

        Into:
            yield (cb, 'wait')
            with cb.wait() as blk:
                ...
        """
        # First, recursively visit child nodes (body and nested withs)
        self.generic_visit(node)

        # Check each context manager (withitem) for wait/reserve calls
        yields_to_insert: list[ast.stmt] = []
        for item in node.items:
            match item.context_expr:
                case ast.Call() as call if self._is_wait_or_reserve_call(call):
                    # Extract operation info
                    match call.func:
                        case ast.Attribute(value=obj, attr=operation):
                            # Create: yield (cb, 'wait') or yield (cb, 'reserve')
                            yield_value = ast.Tuple(
                                elts=[obj, ast.Constant(value=operation)],
                                ctx=ast.Load(),
                            )
                            yield_stmt = ast.Expr(value=ast.Yield(value=yield_value))
                            # Copy location from the call node so yield has same line number
                            ast.copy_location(yield_stmt, call)
                            yields_to_insert.append(yield_stmt)
                        case _:
                            # Non-attribute call, skip
                            pass
                case _:
                    # Non-call expression, skip
                    pass

        # If we found any wait/reserve calls, insert yields before the with statement
        if yields_to_insert:
            return yields_to_insert + [node]  # type: ignore[return-value]
        return node

    def _is_wait_or_reserve_call(self, call_node: ast.Call) -> bool:
        """Check if a call node is a wait() or reserve() method call."""
        match call_node.func:
            case ast.Attribute(attr=attr) if attr in ("wait", "reserve"):
                return True
            case _:
                return False

    def _insert_yield_before(self, stmt: ast.stmt) -> list[ast.stmt]:
        """
        Insert yield with operation info before a wait/reserve call statement.

        Generates:
        1. yield (cb, 'operation')  # Info for scheduler
        2. original statement
        """
        # Extract call node and get operation info
        call: ast.Call
        match stmt:
            case ast.Assign(value=ast.Call() as call_node):
                call = call_node
            case ast.Expr(value=ast.Call() as call_node):
                call = call_node
            case _:
                raise ValueError(f"Unexpected statement type: {type(stmt)}")

        match call.func:
            case ast.Attribute(value=obj, attr=operation):
                pass
            case _:
                raise ValueError(f"Expected attribute access, got {type(call.func)}")

        # Create: yield (cb, 'wait') or yield (cb, 'reserve')
        yield_value = ast.Tuple(
            elts=[obj, ast.Constant(value=operation)], ctx=ast.Load()
        )
        yield_stmt = ast.Expr(value=ast.Yield(value=yield_value))
        # Copy location from the call node so yield has same line number
        ast.copy_location(yield_stmt, call)

        # Return yield followed by original statement
        return [yield_stmt, stmt]


class YieldingFunctionMarker(ast.NodeVisitor):
    """
    Stage 2: Identify all functions that contain yields (directly or transitively).

    Traverses the AST to mark:
    1. Functions that directly contain yield statements
    2. Functions that call other yielding functions
    3. Functions that call function parameters (potential generator callbacks)
    """

    def __init__(self) -> None:
        super().__init__()
        self.functions_with_yields: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and check if they contain yields."""
        # Visit children to process nested functions first
        self.generic_visit(node)

        # Check if this function contains yields or calls yielding functions
        if self._function_has_yields(node):
            self.functions_with_yields.add(node.name)

    def _function_has_yields(self, func_node: ast.FunctionDef) -> bool:
        """Check if a function contains yields directly or calls yielding functions."""
        # Get this function's parameters for checking callback calls
        func_params = {arg.arg for arg in func_node.args.args}

        for node in ast.walk(func_node):
            match node:
                # Direct yield statements
                case ast.Yield() | ast.YieldFrom():
                    return True

                # Direct calls to known yielding functions
                case ast.Call(func=ast.Name(id=func_id)) if (
                    func_id in self.functions_with_yields
                ):
                    return True

                # Calls to function parameters (potential generator callbacks)
                # TODO: This conservatively assumes all function parameters might be generators,
                # which could lead to unnecessary 'yield from' insertions for regular functions.
                # However, we can't determine at AST time whether a parameter will be a generator,
                # and 'yield from' on a non-generator just returns its value, so this is safe.
                case ast.Call(func=ast.Name(id=param_id)) if param_id in func_params:
                    return True

                # Calls to known generator functions from ttl module
                case ast.Call(func=ast.Attribute(attr=attr_name)) if (
                    attr_name in KNOWN_TTL_GENERATORS
                ):
                    return True

                case _:
                    continue

        return False


class YieldFromInserter(ast.NodeTransformer):
    """
    Stage 3: Insert 'yield from' for calls to functions that contain yields.

    Transforms:
        func(args)

    Into:
        yield from func(args)

    When func is known to be a generator function.
    """

    def __init__(self, functions_with_yields: set[str]) -> None:
        super().__init__()
        self.functions_with_yields = functions_with_yields
        self.current_function_params: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function parameters while visiting."""
        # Save outer scope state
        outer_params = self.current_function_params

        # Set up state for this function
        self.current_function_params = {arg.arg for arg in node.args.args}

        # Visit children
        self.generic_visit(node)

        # Restore outer scope state
        self.current_function_params = outer_params

        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        """Visit expression statements and check for calls to yielding functions."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Pattern match to detect call to yielding function
        match node.value:
            case ast.Call() as call if self._should_yield_from_call(call):
                yield_from_expr = ast.Expr(value=ast.YieldFrom(value=node.value))
                ast.copy_location(yield_from_expr, node)
                return yield_from_expr
            case _:
                return node

    def _should_yield_from_call(self, call_node: ast.Call) -> bool:
        """Check if a call should be transformed to 'yield from'."""
        match call_node.func:
            # Handle simple name calls: func(...)
            case ast.Name(id=func_name) if func_name in self.functions_with_yields:
                return True
            # Yield from function parameters (potential generator callbacks)
            case ast.Name(id=param_name) if param_name in self.current_function_params:
                return True
            # Handle attribute access calls: pipe_net.if_src(...), obj.method(...)
            case ast.Attribute(attr=attr_name) if attr_name in KNOWN_TTL_GENERATORS:
                return True
            case _:
                return False


def transform_wait_reserve_to_yield_ast(source: str) -> ast.Module:
    """
    Transform wait() and reserve() calls to yield statements, returning AST.

    Performs a three-stage transformation:
    1. Insert yields before wait()/reserve() calls
    2. Mark all functions that contain yields (directly or transitively)
    3. Insert 'yield from' for calls to yielding functions

    The returned AST preserves original line numbers for accurate error reporting.

    Args:
        source: Python source code as string

    Returns:
        Transformed AST module with original line numbers preserved

    Example:
        >>> code = '''
        ... def func():
        ...     block = cb.wait()
        ...     return block
        ... '''
        >>> tree = transform_wait_reserve_to_yield_ast(code)
        >>> # Compile and execute the AST directly to preserve line numbers
    """
    tree = ast.parse(source)

    # Stage 1: Insert yields before wait()/reserve() calls
    inserter = YieldInserter()
    tree = inserter.visit(tree)
    ast.fix_missing_locations(tree)

    # Stage 2: Mark all functions that contain yields (directly or transitively)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    # Stage 3: Insert 'yield from' for calls to yielding functions
    yield_from_inserter = YieldFromInserter(marker.functions_with_yields)
    tree = yield_from_inserter.visit(tree)
    ast.fix_missing_locations(tree)

    return tree


# Legacy alias for backward compatibility with existing tests
WaitReserveToYieldTransformer = YieldInserter
