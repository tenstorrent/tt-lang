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
    YieldFromInserter,
    YieldingFunctionMarker,
    YieldInserter,
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
    """Test wait() call in assignment with multiple targets (should be transformed)."""
    source = """
def func():
    block = view = cb.wait()
    return block
"""
    # The new implementation transforms all assignments, including chained ones
    result = transform_wait_reserve_to_yield(source)

    # Should insert yield before the chained assignment
    assert "yield (cb, 'wait')" in result
    assert "block = view = cb.wait()" in result

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
    """Test wait/reserve calls inside control flow statements are transformed."""
    source = """
def func():
    if condition:
        block = cb.wait()
    else:
        block = cb.reserve()
    return block
"""
    result = transform_wait_reserve_to_yield(source)

    # The new implementation recursively transforms control flow statements
    # So yields should be inserted inside if/else blocks
    assert "yield (cb, 'wait')" in result
    assert "yield (cb, 'reserve')" in result

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


# ============================================================================
# Tests for the three-stage transformation approach
# ============================================================================


def test_stage1_yield_inserter_only() -> None:
    """Test that stage 1 only inserts yields, no yield from."""
    source = """
def inner():
    block = cb.wait()

def outer():
    inner()
"""
    tree = ast.parse(source)
    inserter = YieldInserter()
    tree = inserter.visit(tree)
    result = ast.unparse(tree)

    # Should have yield before wait()
    assert "yield (cb, 'wait')" in result
    # Should NOT have yield from yet (that's stage 3)
    assert "yield from" not in result

    print("Stage 1 yield inserter test passed!")


def test_stage2_marker_no_yields() -> None:
    """Test that marker correctly identifies no yields when there are none."""
    source = """
def func1():
    x = 1 + 2
    return x

def func2():
    return func1()
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    # No functions should be marked
    assert len(marker.functions_with_yields) == 0

    print("Stage 2 marker (no yields) test passed!")


def test_stage2_marker_direct_yields() -> None:
    """Test that marker identifies functions with direct yields."""
    source = """
def func():
    yield (cb, 'wait')
    block = cb.wait()
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    # func should be marked as it has a direct yield
    assert "func" in marker.functions_with_yields

    print("Stage 2 marker (direct yields) test passed!")


def test_stage2_marker_transitive_yields() -> None:
    """Test that marker identifies functions that call yielding functions."""
    source = """
def inner():
    yield (cb, 'wait')
    block = cb.wait()

def middle():
    inner()

def outer():
    middle()
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    # All three should be marked
    assert "inner" in marker.functions_with_yields
    assert "middle" in marker.functions_with_yields
    assert "outer" in marker.functions_with_yields

    print("Stage 2 marker (transitive yields) test passed!")


def test_stage2_marker_function_params() -> None:
    """Test that marker identifies functions that call function parameters."""
    source = """
def callback_wrapper(func):
    func()
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    # callback_wrapper should be marked because it calls a parameter
    assert "callback_wrapper" in marker.functions_with_yields

    print("Stage 2 marker (function params) test passed!")


def test_stage2_marker_ttl_pipe_functions() -> None:
    """Test that marker identifies functions calling ttl.if_pipe_src/dst."""
    source = """
def compute():
    ttl.if_pipe_src(pipe, callback)
    ttl.if_pipe_dst(pipe, callback)
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    # compute should be marked due to ttl.if_pipe_* calls
    assert "compute" in marker.functions_with_yields

    print("Stage 2 marker (ttl pipe functions) test passed!")


def test_stage3_yield_from_nested_function() -> None:
    """Test that stage 3 inserts yield from for nested function calls."""
    source = """
def inner():
    yield (cb, 'wait')
    block = cb.wait()

def outer():
    inner()
"""
    tree = ast.parse(source)

    # Run through all stages
    inserter = YieldInserter()
    tree = inserter.visit(tree)
    ast.fix_missing_locations(tree)

    marker = YieldingFunctionMarker()
    marker.visit(tree)

    yield_from_inserter = YieldFromInserter(marker.functions_with_yields)
    tree = yield_from_inserter.visit(tree)
    result = ast.unparse(tree)

    # Should have yield from inner()
    assert "yield from inner()" in result

    print("Stage 3 yield from (nested function) test passed!")


def test_stage3_yield_from_function_param() -> None:
    """Test that stage 3 inserts yield from for function parameter calls."""
    source = """
def wrapper(func):
    func()
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    yield_from_inserter = YieldFromInserter(marker.functions_with_yields)
    tree = yield_from_inserter.visit(tree)
    result = ast.unparse(tree)

    # Should have yield from func()
    assert "yield from func()" in result

    print("Stage 3 yield from (function param) test passed!")


def test_stage3_yield_from_ttl_functions() -> None:
    """Test that stage 3 inserts yield from for ttl.if_pipe_* calls."""
    source = """
def compute():
    ttl.if_pipe_src(pipe, dm0)
    ttl.if_pipe_dst(pipe, dm1)
"""
    tree = ast.parse(source)
    marker = YieldingFunctionMarker()
    marker.visit(tree)

    yield_from_inserter = YieldFromInserter(marker.functions_with_yields)
    tree = yield_from_inserter.visit(tree)
    result = ast.unparse(tree)

    # Should have yield from for both calls
    assert "yield from ttl.if_pipe_src(pipe, dm0)" in result
    assert "yield from ttl.if_pipe_dst(pipe, dm1)" in result

    print("Stage 3 yield from (ttl functions) test passed!")


def test_three_level_nesting() -> None:
    """Test three levels of nested yielding functions."""
    source = """
def level1():
    block = cb.wait()

def level2():
    level1()

def level3():
    level2()
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have yield in level1
    assert "yield (cb, 'wait')" in result
    # Should have yield from in level2 and level3
    assert "yield from level1()" in result
    assert "yield from level2()" in result

    print("Three level nesting test passed!")


def test_four_level_nesting() -> None:
    """Test four levels of nested yielding functions."""
    source = """
def level1():
    x = cb.reserve()

def level2():
    level1()

def level3():
    level2()

def level4():
    level3()
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have yield in level1
    assert "yield (cb, 'reserve')" in result
    # Should have yield from in levels 2-4
    assert "yield from level1()" in result
    assert "yield from level2()" in result
    assert "yield from level3()" in result

    print("Four level nesting test passed!")


def test_mixed_yielding_and_non_yielding() -> None:
    """Test mix of yielding and non-yielding functions."""
    source = """
def yielding():
    block = cb.wait()

def non_yielding():
    x = 1 + 2
    return x

def caller():
    yielding()
    non_yielding()
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have yield from for yielding()
    assert "yield from yielding()" in result
    # Should NOT have yield from for non_yielding()
    assert "yield from non_yielding()" not in result
    # non_yielding() should still be called normally
    assert "non_yielding()" in result

    print("Mixed yielding/non-yielding test passed!")


def test_multiple_nested_functions_same_level() -> None:
    """Test multiple nested functions at the same level."""
    source = """
def pipe_src():
    block = cb.wait()

def pipe_dst():
    block = cb.reserve()

def coordinator():
    pipe_src()
    pipe_dst()
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have yields in both inner functions
    assert result.count("yield (cb,") == 2
    # Should have yield from for both calls
    assert "yield from pipe_src()" in result
    assert "yield from pipe_dst()" in result

    print("Multiple nested functions test passed!")


def test_recursive_function_marking() -> None:
    """Test that recursive calls are handled correctly."""
    source = """
def recursive(n):
    if n > 0:
        block = cb.wait()
        recursive(n - 1)
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have yield before wait
    assert "yield (cb, 'wait')" in result
    # Should have yield from for recursive call
    assert "yield from recursive(n - 1)" in result

    print("Recursive function marking test passed!")


def test_no_yields_no_transformation() -> None:
    """Test that code with no yields remains unchanged (except formatting)."""
    source = """
def func1():
    x = 1 + 2
    return x

def func2():
    y = func1()
    return y * 2
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have no yields or yield froms
    assert "yield" not in result
    # Functions should still exist and be callable normally
    assert "def func1():" in result
    assert "def func2():" in result
    assert "func1()" in result

    print("No yields no transformation test passed!")


def test_yield_from_not_inserted_for_non_yielding() -> None:
    """Test that yield from is not inserted for non-yielding function calls."""
    source = """
def helper():
    return 42

def main():
    x = cb.wait()
    result = helper()
"""
    result = transform_wait_reserve_to_yield(source)

    # Should have yield for wait
    assert "yield (cb, 'wait')" in result
    # Should NOT have yield from for helper()
    assert "yield from helper()" not in result
    # helper() should be called normally (might be in assignment)
    assert "helper()" in result

    print("Yield from not inserted for non-yielding test passed!")


def test_transform_copy_wait() -> None:
    """Test transformation of copy().wait() calls in assignment.

    Verifies that CopyTransaction.wait() calls get yields inserted,
    confirming that copy operations participate in cooperative scheduling.
    """
    source = """
def dm_func():
    # Create copy transaction
    tx = ttl.copy(src_tensor, dst_block)
    # Wait for copy to complete
    tx.wait()
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yield was inserted before copy().wait()
    assert "yield (tx, 'wait')" in result
    assert "tx.wait()" in result

    # Verify the yield comes before the wait
    yield_pos = result.index("yield (tx, 'wait')")
    wait_pos = result.index("tx.wait()")
    assert yield_pos < wait_pos

    print("copy().wait() transformation test passed!")


def test_transform_multiple_copy_waits() -> None:
    """Test transformation of multiple copy().wait() calls.

    Verifies yields are inserted for all copy transactions,
    enabling proper scheduling of multiple concurrent copy operations.
    """
    source = """
def dm_func():
    a_tx = ttl.copy(a_src, a_block)
    b_tx = ttl.copy(b_src, b_block)
    a_tx.wait()
    b_tx.wait()
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yields were inserted for both copy operations
    assert "yield (a_tx, 'wait')" in result
    assert "yield (b_tx, 'wait')" in result
    assert "a_tx.wait()" in result
    assert "b_tx.wait()" in result

    # Verify yields come before their respective waits
    a_yield_pos = result.index("yield (a_tx, 'wait')")
    a_wait_pos = result.index("a_tx.wait()")
    assert a_yield_pos < a_wait_pos

    b_yield_pos = result.index("yield (b_tx, 'wait')")
    b_wait_pos = result.index("b_tx.wait()")
    assert b_yield_pos < b_wait_pos

    print("multiple copy().wait() transformation test passed!")


def test_transform_mixed_cb_and_copy_waits() -> None:
    """Test transformation mixing CircularBuffer and CopyTransaction operations.

    Verifies that both CB.wait()/reserve() and copy().wait() get yields,
    confirming both types participate in cooperative scheduling.
    """
    source = """
def mixed_func():
    block = cb.reserve()
    tx = ttl.copy(tensor, block)
    tx.wait()
    out_block = cb.wait()
    out_tx = ttl.copy(out_block, tensor_out)
    out_tx.wait()
    cb.pop()
"""
    result = transform_wait_reserve_to_yield(source)

    # Verify yields for CB operations
    assert "yield (cb, 'reserve')" in result
    assert "yield (cb, 'wait')" in result

    # Verify yields for copy operations
    assert "yield (tx, 'wait')" in result
    assert "yield (out_tx, 'wait')" in result

    # Verify all wait/reserve calls are present
    assert "block = cb.reserve()" in result
    assert "out_block = cb.wait()" in result
    assert "tx.wait()" in result
    assert "out_tx.wait()" in result

    print("mixed CB and copy wait transformation test passed!")


if __name__ == "__main__":
    # Run original tests
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

    # New three-stage transformation tests
    test_stage1_yield_inserter_only()
    test_stage2_marker_no_yields()
    test_stage2_marker_direct_yields()
    test_stage2_marker_transitive_yields()
    test_stage2_marker_function_params()
    test_stage2_marker_ttl_pipe_functions()
    test_stage3_yield_from_nested_function()
    test_stage3_yield_from_function_param()
    test_stage3_yield_from_ttl_functions()
    test_three_level_nesting()
    test_four_level_nesting()
    test_mixed_yielding_and_non_yielding()
    test_multiple_nested_functions_same_level()
    test_recursive_function_marking()
    test_no_yields_no_transformation()
    test_yield_from_not_inserted_for_non_yielding()

    # Copy operation transformation tests
    test_transform_copy_wait()
    test_transform_multiple_copy_waits()
    test_transform_mixed_cb_and_copy_waits()

    print("All xformyield tests passed!")
