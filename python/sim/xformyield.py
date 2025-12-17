# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AST transformers for TTL code.

This module provides AST transformers that modify Python AST nodes
to enable cooperative multitasking and other program transformations.
"""

import ast


class WaitReserveToYieldTransformer(ast.NodeTransformer):
    """
    Transforms wait() and reserve() calls to add cooperative yielding.

    Inserts a yield before each wait()/reserve() call, passing the object
    and operation name to the scheduler for deadlock detection.

    Examples:
        block = cb.wait()

    Becomes:
        yield (cb, 'wait')
        block = cb.wait()

    The NodeTransformer base class automatically recurses into all nested
    control structures (for, while, if, with, try, etc.) without explicit handling.
    """

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
        """Visit assignment statements and check for wait/reserve calls."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Check if this is an assignment from wait/reserve call
        if isinstance(node.value, ast.Call) and self._is_wait_or_reserve_call(
            node.value
        ):
            result: list[ast.AST] = list(self._insert_yield_before(node))
            return result
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        """Visit expression statements and check for wait/reserve calls."""
        # First, recursively visit child nodes
        self.generic_visit(node)

        # Check if this is a standalone wait/reserve call
        if isinstance(node.value, ast.Call) and self._is_wait_or_reserve_call(
            node.value
        ):
            result: list[ast.AST] = list(self._insert_yield_before(node))
            return result
        return node

    def _is_wait_or_reserve_call(self, call_node: ast.Call) -> bool:
        """Check if a call node is a wait() or reserve() method call."""
        return isinstance(call_node.func, ast.Attribute) and call_node.func.attr in (
            "wait",
            "reserve",
        )

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
            case ast.Assign(value=call_node) if isinstance(call_node, ast.Call):
                call = call_node
            case ast.Expr(value=call_node) if isinstance(call_node, ast.Call):
                call = call_node
            case _:
                raise ValueError(f"Unexpected statement type: {type(stmt)}")

        if not isinstance(call.func, ast.Attribute):
            raise ValueError(f"Expected attribute access, got {type(call.func)}")

        obj = call.func.value
        operation = call.func.attr

        # Create: yield (cb, 'wait') or yield (cb, 'reserve')
        yield_value = ast.Tuple(
            elts=[obj, ast.Constant(value=operation)], ctx=ast.Load()
        )
        yield_stmt = ast.Expr(value=ast.Yield(value=yield_value))

        # Return yield followed by original statement
        return [yield_stmt, stmt]


def transform_wait_reserve_to_yield(source: str) -> str:
    """
    Transform wait() and reserve() calls to yield operation info to the scheduler.

    Inserts `yield (cb, 'operation')` before each wait()/reserve() call,
    allowing the scheduler to check if the operation can proceed and handle
    all control logic externally.

    The NodeTransformer automatically recurses into all nested control structures
    (for, while, if, with, try, etc.) without explicit handling.

    Args:
        source: Python source code as string

    Returns:
        Transformed source code with yields before blocking operations

    Example:
        >>> code = '''
        ... def func():
        ...     block = cb.wait()
        ...     return block
        ... '''
        >>> transformed = transform_wait_reserve_to_yield(code)
        >>> print(transformed)
        def func():
            yield (cb, 'wait')
            block = cb.wait()
            return block
    """
    tree = ast.parse(source)
    transformer = WaitReserveToYieldTransformer()
    transformed_tree = transformer.visit(tree)
    ast.fix_missing_locations(transformed_tree)
    return ast.unparse(transformed_tree)
