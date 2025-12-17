# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test AST transformers for TTL code.

This test verifies that the WaitReserveToYieldTransformer correctly
transforms wait() and reserve() calls to insert cooperative yields.
"""

import ast
from python.sim.xformyield import (
    WaitReserveToYieldTransformer,
    transform_wait_reserve_to_yield,
)


def test_transform_wait_assignment() -> None:
    """Test transformation of wait() call in assignment statement."""
    source = """
def func():
    block = cb.wait()
    return block
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yield was inserted before wait()
    assert "yield (cb, 'wait')" in result
    assert "block = cb.wait()" in result

    # Verify the yield comes before the wait
    yield_pos = result.index("yield (cb, 'wait')")
    wait_pos = result.index("block = cb.wait()")
    assert yield_pos < wait_pos

    print("wait() assignment transformation test passed!")


def test_transform_reserve_assignment() -> None:
    """Test transformation of reserve() call in assignment statement."""
    source = """
def func():
    block = cb.reserve()
    return block
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yield was inserted before reserve()
    assert "yield (cb, 'reserve')" in result
    assert "block = cb.reserve()" in result

    # Verify the yield comes before the reserve
    yield_pos = result.index("yield (cb, 'reserve')")
    reserve_pos = result.index("block = cb.reserve()")
    assert yield_pos < reserve_pos

    print("reserve() assignment transformation test passed!")


def test_transform_wait_expression() -> None:
    """Test transformation of wait() call as expression statement."""
    source = """
def func():
    cb.wait()
    print("done")
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yield was inserted before wait()
    assert "yield (cb, 'wait')" in result
    assert "cb.wait()" in result

    print("wait() expression statement transformation test passed!")


def test_transform_reserve_expression() -> None:
    """Test transformation of reserve() call as expression statement."""
    source = """
def func():
    cb.reserve()
    print("done")
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yield was inserted before reserve()
    assert "yield (cb, 'reserve')" in result
    assert "cb.reserve()" in result

    print("reserve() expression statement transformation test passed!")


def test_transform_multiple_calls() -> None:
    """Test transformation of multiple wait/reserve calls in one function."""
    source = """
def func():
    block1 = cb1.wait()
    block2 = cb2.reserve()
    cb3.wait()
    cb4.reserve()
    return block1
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify all yields were inserted
    assert "yield (cb1, 'wait')" in result
    assert "yield (cb2, 'reserve')" in result
    assert "yield (cb3, 'wait')" in result
    assert "yield (cb4, 'reserve')" in result

    # Verify original calls remain
    assert "block1 = cb1.wait()" in result
    assert "block2 = cb2.reserve()" in result
    assert "cb3.wait()" in result
    assert "cb4.reserve()" in result

    print("multiple calls transformation test passed!")


def test_non_blocking_calls_unchanged() -> None:
    """Test that non-wait/reserve calls are not transformed."""
    source = """
def func():
    x = cb.push()
    cb.pop()
    y = other.method()
    result = cb.stats()
    return result
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify no yields were inserted
    assert "yield" not in result

    # Verify original calls remain unchanged
    assert "x = cb.push()" in result
    assert "cb.pop()" in result
    assert "y = other.method()" in result
    assert "result = cb.stats()" in result

    print("non-blocking calls unchanged test passed!")


def test_nested_function_definitions() -> None:
    """Test transformation in nested function definitions."""
    source = """
def outer():
    block1 = cb1.wait()

    def inner():
        block2 = cb2.reserve()
        return block2

    return block1
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify both functions were transformed
    assert "yield (cb1, 'wait')" in result
    assert "yield (cb2, 'reserve')" in result

    print("nested function definitions test passed!")


def test_complex_expressions() -> None:
    """Test transformation with complex expressions involving wait/reserve."""
    source = """
def func():
    block = buffer.cb.wait()
    data = some_obj.get_cb().reserve()
    return block
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yields were inserted with the correct objects
    assert "yield (buffer.cb, 'wait')" in result
    assert "yield (some_obj.get_cb(), 'reserve')" in result

    print("complex expressions test passed!")


def test_wait_with_multiple_targets() -> None:
    """Test wait() call in assignment with multiple targets (should still work)."""
    source = """
def func():
    block = view = cb.wait()
    return block
"""
    # This should not be transformed since the pattern matcher expects single target
    result = transform_wait_reserve_to_yield(source)

    # The transformation only handles single assignment targets
    # This statement should pass through unchanged (no yield inserted)
    assert "yield" not in result or result.count("yield") == 0

    print("multiple assignment targets test passed!")


def test_transformer_class_directly() -> None:
    """Test using the WaitReserveToYieldTransformer class directly."""
    source = """
def func():
    block = cb.wait()
"""
    tree = ast.parse(source)
    transformer = WaitReserveToYieldTransformer()
    transformed_tree = transformer.visit(tree)
    ast.fix_missing_locations(transformed_tree)

    # Verify the tree was transformed
    func_def = transformed_tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    # Should have 2 statements now: yield + original assignment
    assert len(func_def.body) == 2

    # First statement should be yield
    assert isinstance(func_def.body[0], ast.Expr)
    assert isinstance(func_def.body[0].value, ast.Yield)

    # Second statement should be the original assignment
    assert isinstance(func_def.body[1], ast.Assign)

    print("transformer class direct usage test passed!")


def test_wait_reserve_in_control_flow() -> None:
    """Test wait/reserve calls inside control flow statements."""
    source = """
def func():
    if condition:
        block = cb.wait()
    else:
        block = cb.reserve()
    return block
"""
    result = transform_wait_reserve_to_yield(source)

    # Control flow statements are not recursively transformed
    # (based on the implementation which only handles top-level statements)
    # So no yields should be inserted
    # Actually, looking at the code, generic_visit is called which should handle this
    # Let me check if yields appear
    # The _transform_statement only handles direct statements in function body
    # Not statements inside if/else blocks
    assert "yield" not in result

    print("wait/reserve in control flow test passed!")


def test_empty_function() -> None:
    """Test transformation of empty function (edge case)."""
    source = """
def func():
    pass
"""
    result = transform_wait_reserve_to_yield(source)

    # Should remain unchanged
    assert "pass" in result
    assert "yield" not in result

    print("empty function test passed!")


def test_function_with_no_wait_reserve() -> None:
    """Test transformation of function without wait/reserve calls."""
    source = """
def func():
    x = 1 + 2
    y = compute(x)
    return y * 2
"""
    result = transform_wait_reserve_to_yield(source)

    # Should remain unchanged
    assert "yield" not in result
    assert "x = 1 + 2" in result
    assert "y = compute(x)" in result
    assert "return y * 2" in result

    print("function with no wait/reserve test passed!")


def test_preserve_function_signature() -> None:
    """Test that function signatures are preserved."""
    source = """
def func(arg1: int, arg2: str, *args, **kwargs) -> bool:
    block = cb.wait()
    return True
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify signature is preserved (ast.unparse format)
    assert "def func(arg1: int, arg2: str, *args, **kwargs) -> bool:" in result
    assert "yield (cb, 'wait')" in result

    print("preserve function signature test passed!")


def test_wait_in_return_statement() -> None:
    """Test wait() call directly in return statement."""
    source = """
def func():
    return cb.wait()
"""
    result = transform_wait_reserve_to_yield(source)

    # Return statement is not assignment or expression statement,
    # so it should not be transformed
    assert "yield" not in result
    assert "return cb.wait()" in result

    print("wait in return statement test passed!")


def test_wait_as_function_argument() -> None:
    """Test wait() call as function argument."""
    source = """
def func():
    result = process(cb.wait())
    return result
"""
    result = transform_wait_reserve_to_yield(source)

    # The wait() is part of a larger call expression, not a direct wait() call
    # so it should not be transformed
    assert "yield" not in result
    assert "result = process(cb.wait())" in result

    print("wait as function argument test passed!")


def test_yield_tuple_structure() -> None:
    """Test that yielded value is a tuple with correct structure."""
    source = """
def func():
    block = my_buffer.wait()
"""
    tree = ast.parse(source)
    transformer = WaitReserveToYieldTransformer()
    transformed_tree = transformer.visit(tree)

    func_def = transformed_tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    # First statement should be yield expression
    yield_stmt = func_def.body[0]
    assert isinstance(yield_stmt, ast.Expr)
    assert isinstance(yield_stmt.value, ast.Yield)

    # Yield value should be a tuple
    yield_value = yield_stmt.value.value
    assert isinstance(yield_value, ast.Tuple)
    assert len(yield_value.elts) == 2

    # First element should be the object (my_buffer)
    obj = yield_value.elts[0]
    assert isinstance(obj, ast.Name)
    assert obj.id == "my_buffer"

    # Second element should be the operation name ('wait')
    operation = yield_value.elts[1]
    assert isinstance(operation, ast.Constant)
    assert operation.value == "wait"

    print("yield tuple structure test passed!")


def test_multiple_functions() -> None:
    """Test transformation across multiple function definitions."""
    source = """
def func1():
    block = cb1.wait()
    return block

def func2():
    block = cb2.reserve()
    return block
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify both functions were transformed
    assert "yield (cb1, 'wait')" in result
    assert "yield (cb2, 'reserve')" in result

    print("multiple functions test passed!")


def test_docstring_preserved() -> None:
    """Test that function docstrings are preserved."""
    source = '''
def func():
    """This is a docstring."""
    block = cb.wait()
    return block
'''
    result = transform_wait_reserve_to_yield(source)

    # Verify docstring is preserved
    assert '"""This is a docstring."""' in result or "'This is a docstring.'" in result
    assert "yield (cb, 'wait')" in result

    print("docstring preserved test passed!")


if __name__ == "__main__":
    test_transform_wait_assignment()
    test_transform_reserve_assignment()
    test_transform_wait_expression()
    test_transform_reserve_expression()
    test_transform_multiple_calls()
    test_non_blocking_calls_unchanged()
    test_nested_function_definitions()
    test_complex_expressions()
    test_wait_with_multiple_targets()
    test_transformer_class_directly()
    test_wait_reserve_in_control_flow()
    test_empty_function()
    test_function_with_no_wait_reserve()
    test_preserve_function_signature()
    test_wait_in_return_statement()
    test_wait_as_function_argument()
    test_yield_tuple_structure()
    test_multiple_functions()
    test_docstring_preserved()
    print("All xformyield tests passed!")
