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

    Inserts an unconditional yield before each wait()/reserve() call,
    allowing the scheduler to check if the operation can proceed.

    Examples:
        block = cb.wait()

    Becomes:
        yield
        block = cb.wait()

    The scheduler (in program.py) handles all control logic:
        1. Generator yields before blocking operation
        2. Scheduler checks if operation can proceed
        3. If yes, continues the generator
        4. If no, tries other generators
        5. If all blocked, raises deadlock error
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visit function definitions and transform their bodies."""
        # First, recursively visit children
        self.generic_visit(node)

        # Transform the body statements
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            new_stmts = self._transform_statement(stmt)
            new_body.extend(new_stmts)

        node.body = new_body
        return node

    def _transform_statement(self, stmt: ast.stmt) -> list[ast.stmt]:
        """
        Transform a statement, replacing wait/reserve calls with deadlock-aware versions.

        Returns:
            List of statements (possibly expanded with checks and yields)
        """
        match stmt:
            case ast.Assign(
                targets=[_], value=ast.Call() as call
            ) if self._is_wait_or_reserve_call(call):
                return self._insert_yield_before(stmt)
            case ast.Expr(value=ast.Call() as call) if self._is_wait_or_reserve_call(
                call
            ):
                return self._insert_yield_before(stmt)
            case _:
                # For all other statements, return as-is
                return [stmt]

    def _is_wait_or_reserve_call(self, call_node: ast.Call) -> bool:
        """Check if a call node is a wait() or reserve() method call."""
        match call_node.func:
            case ast.Attribute(attr="wait" | "reserve"):
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
        obj: ast.expr
        operation: str
        match stmt:
            case ast.Assign(
                value=ast.Call(func=ast.Attribute(value=obj_node, attr=op_name))
            ):
                obj = obj_node
                operation = op_name
            case ast.Expr(
                value=ast.Call(func=ast.Attribute(value=obj_node, attr=op_name))
            ):
                obj = obj_node
                operation = op_name
            case _:
                raise ValueError(
                    f"Expected statement with wait/reserve call, got {type(stmt)}"
                )

        # Create: yield (cb, 'wait')  or  yield (cb, 'reserve')
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
    # Parse source to AST
    tree = ast.parse(source)

    # Apply transformation
    transformer = WaitReserveToYieldTransformer()
    transformed_tree = transformer.visit(tree)

    # Fix missing locations
    ast.fix_missing_locations(transformed_tree)

    # Unparse back to Python
    return ast.unparse(transformed_tree)
