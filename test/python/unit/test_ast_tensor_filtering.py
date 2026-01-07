# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for AST-based tensor filtering in ttl_ast.py.

Tests that TTLGenericCompiler._collect_used_tensor_names() correctly identifies
which captured tensors are actually referenced in thread functions by validating
against known expected results from realistic kernel patterns.
"""

import ast
import pytest
from ttlang._src.ttl_ast import TTLGenericCompiler


@pytest.fixture(scope="module")
def compiler():
    """Create a minimal TTLGenericCompiler instance for testing."""
    return TTLGenericCompiler(name="test", kernel_type="noc")


def apply_tensor_filtering(compiler, code: str, available_tensors: set) -> set:
    """
    Helper method to apply AST-based tensor filtering using the
    TTLGenericCompiler._collect_used_tensor_names method (being tested).

    Args:
        compiler: TTLGenericCompiler instance (from fixture)
        code: Python code string containing a function definition
        available_tensors: Set of tensor names that could be captured

    Returns:
        Set of tensor names actually referenced in the function body
    """
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)

    # Call the actual method we're testing
    result = compiler._collect_used_tensor_names(func_node, available_tensors)

    return result


def test_dm_read_uses_lhs_and_rhs(compiler):
    """dm_read kernel reads lhs and rhs tensors, doesn't use out."""
    code = """
def dm_read():
    lhs_cb.reserve()
    tx_lhs = copy(lhs[0, 0], lhs_cb)
    tx_lhs.wait()
    lhs_cb.push()

    rhs_cb.reserve()
    tx_rhs = copy(rhs[0, 0], rhs_cb)
    tx_rhs.wait()
    rhs_cb.push()
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs", "rhs"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_dm_write_uses_only_out():
    """dm_write kernel writes to out tensor only, doesn't read lhs/rhs."""
    code = """
def dm_write():
    out_cb.wait()
    tx = copy(out_cb, out[0, 0])
    tx.wait()
    out_cb.pop()
"""
    available = {"lhs", "rhs", "out"}
    expected = {"out"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_compute_kernel_uses_no_tensors():
    """Compute kernel operates on CBs only, doesn't access tensors."""
    code = """
def add_compute():
    l = lhs_cb.wait()
    r = rhs_cb.wait()
    o = out_cb.reserve()
    result = l + r
    o.store(result)
    lhs_cb.pop()
    rhs_cb.pop()
    out_cb.push()
"""
    available = {"lhs", "rhs", "out"}
    expected = set()

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_with_statement_pattern():
    """Tensors used inside with statements are detected."""
    code = """
def dm_read():
    with lhs_cb.reserve():
        copy(lhs[0, 0], lhs_cb)
    with rhs_cb.reserve():
        copy(rhs[0, 0], rhs_cb)
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs", "rhs"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_nested_loops_and_conditionals():
    """Tensors used in nested control flow are detected."""
    code = """
def dm_read():
    for i in range(2):
        if i == 0:
            copy(lhs[i, 0], lhs_cb)
        else:
            copy(rhs[i, 0], rhs_cb)
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs", "rhs"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_tensor_in_subscript_expression():
    """Tensor used in subscript (e.g., lhs[0, 0]) is detected."""
    code = """
def thread_fn():
    tile = lhs[0, 0]
    copy(tile, cb)
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_tensor_attribute_access():
    """Tensor attribute access (e.g., lhs.shape) is detected."""
    code = """
def thread_fn():
    s = lhs.shape
    copy(rhs[0, 0], cb)
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs", "rhs"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_all_tensors_used():
    """All captured tensors are used in the function."""
    code = """
def thread_fn():
    copy(lhs[0, 0], lhs_cb)
    copy(rhs[0, 0], rhs_cb)
    copy(out_cb, out[0, 0])
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs", "rhs", "out"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_empty_function_uses_no_tensors():
    """Empty function (or pass statement) uses no tensors."""
    code = """
def thread_fn():
    pass
"""
    available = {"lhs", "rhs", "out"}
    expected = set()

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_similar_names_not_confused():
    """Variables with similar names to tensors don't cause false positives."""
    code = """
def thread_fn():
    lhs_local = 5
    rhs_index = 2
    copy(lhs[0, 0], cb)
    x = lhs_local + rhs_index
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs"}  # Only 'lhs' tensor, not 'lhs_local' or 'rhs_index'

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_tensor_used_only_in_loop():
    """Tensor used only inside loop body is detected."""
    code = """
def thread_fn():
    for i in range(4):
        copy(lhs[i, 0], cb)
"""
    available = {"lhs", "rhs", "out"}
    expected = {"lhs"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_tensor_used_only_in_else_branch(compiler):
    """Tensor used only in else branch is detected."""
    code = """
def thread_fn():
    if condition:
        use_cb()
    else:
        copy(out[0, 0], out_cb)
"""
    available = {"lhs", "rhs", "out"}
    expected = {"out"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_sparse_subset_from_many_tensors(compiler):
    """Non-sequential subset of tensors used from a large capture list."""
    code = """
def complex_kernel():
    # Has access to 10 tensors (input0-input7, weight, output)
    # But only uses input2, input5, weight, and output (indices 2, 5, 8, 9)
    copy(input2[0, 0], cb0)
    copy(input5[1, 0], cb1)
    w = weight[0, 0]
    result = compute(w)
    copy(cb_out, output[0, 0])
"""
    available = {
        "input0",
        "input1",
        "input2",
        "input3",
        "input4",
        "input5",
        "input6",
        "input7",
        "weight",
        "output",
    }
    # Only indices 2, 5, 8, 9 from the sorted list are used
    expected = {"input2", "input5", "weight", "output"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_first_and_last_from_many_tensors(compiler):
    """Uses only first and last tensors from a large set."""
    code = """
def edge_case_kernel():
    # Has access to tensor0-tensor9
    # Uses only tensor0 and tensor9
    copy(tensor0[0, 0], cb_in)
    process_data()
    copy(cb_out, tensor9[0, 0])
"""
    available = {f"tensor{i}" for i in range(10)}
    expected = {"tensor0", "tensor9"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_middle_tensors_only(compiler):
    """Uses only middle tensors, skipping first and last."""
    code = """
def middle_kernel():
    # Has access to a, b, c, d, e, f, g
    # Uses only c, d, e (middle three)
    copy(c[0, 0], cb0)
    copy(d[0, 0], cb1)
    copy(e[0, 0], cb2)
"""
    available = {"a", "b", "c", "d", "e", "f", "g"}
    expected = {"c", "d", "e"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_first_and_second_from_three(compiler):
    """First thread uses x and y, validates correct indices."""
    code = """
def thread_fn():
    copy(x[0, 0], cb0)
    copy(y[0, 0], cb1)
"""
    available = {"x", "y", "z"}
    expected = {"x", "y"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


def test_first_and_third_from_three(compiler):
    """Second thread uses x and z, skipping y in the middle."""
    code = """
def thread_fn():
    copy(x[0, 0], cb0)
    copy(z[0, 0], cb1)
"""
    available = {"x", "y", "z"}
    expected = {"x", "z"}

    result = apply_tensor_filtering(compiler, code, available)
    assert result == expected


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
