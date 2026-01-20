# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for DST register allocation with multi-consumer block arguments.

Tests various patterns where block arguments or operation results are consumed
by multiple operations, including combinations of unary and binary operations.
These tests validate correct copy_tile insertion to prevent register clobbering.

Test patterns:
1. SiLU (x * sigmoid(x)): block arg with unary + binary consumers
2. Unary+Binary (abs(x) + x+y + x*y): block arg with 1 unary + 2 binary
3. Three consumers (sigmoid(a), exp(a), a+b): block arg with 2 unary + 1 binary
4. Square pattern (x * x): same value used as both operands
5. Unary chain branch (abs→exp, abs→add): operation result with mixed consumers
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttnn
from test_helpers import assert_allclose, to_dram

from ttl import ttl

pytestmark = pytest.mark.requires_ttnn


@ttl.kernel(grid=(1, 1))
def silu_kernel(x, out):
    """SiLU: x * sigmoid(x) - tests multi-consumer DST allocation."""
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv:
            with out_cb.reserve() as o:
                sig = ttl.math.sigmoid(xv)
                result = xv * sig
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def unary_binary_kernel(x, y, out):
    """Tests block arg with one unary and two binary consumers."""
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    y_cb = ttl.make_circular_buffer_like(y, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv, y_cb.wait() as yv:
            with out_cb.reserve() as o:
                # x is used by: abs (unary), add (binary), mul (binary)
                abs_x = ttl.math.abs(xv)
                add_result = xv + yv
                mul_result = xv * yv
                # Combine all results
                result = abs_x + add_result + mul_result
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as x_blk, y_cb.reserve() as y_blk:
            tx = ttl.copy(x[0, 0], x_blk)
            ty = ttl.copy(y[0, 0], y_blk)
            tx.wait()
            ty.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_silu(device):
    """Test SiLU activation: x * sigmoid(x)."""
    x_torch = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] + [1.0] * 26] * 32, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    expected = x_torch * torch.sigmoid(x_torch.float()).to(torch.bfloat16)

    x_t = to_dram(x_torch, device)
    out_t = to_dram(out_torch, device)

    silu_kernel(x_t, out_t)

    result = ttnn.to_torch(out_t)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_unary_binary_consumers(device):
    """Test block arg used by one unary op and two binary ops."""
    x_torch = torch.tensor(
        [[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0] + [0.5] * 26] * 32, dtype=torch.bfloat16
    )
    y_torch = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [2.0] * 26] * 32, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    # Expected: abs(x) + (x + y) + (x * y)
    expected = (
        torch.abs(x_torch.float())
        + (x_torch.float() + y_torch.float())
        + (x_torch.float() * y_torch.float())
    ).to(torch.bfloat16)

    x_t = to_dram(x_torch, device)
    y_t = to_dram(y_torch, device)
    out_t = to_dram(out_torch, device)

    unary_binary_kernel(x_t, y_t, out_t)

    result = ttnn.to_torch(out_t)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@ttl.kernel(grid=(1, 1))
def three_consumers_kernel(a, b, out_sig, out_exp, out_add):
    """Block arg with 2 unary + 1 binary consumers: sigmoid(a), exp(a), a+b."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    sig_cb = ttl.make_circular_buffer_like(out_sig, shape=(1, 1), buffer_factor=2)
    exp_cb = ttl.make_circular_buffer_like(out_exp, shape=(1, 1), buffer_factor=2)
    add_cb = ttl.make_circular_buffer_like(out_add, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, b_cb.wait() as bv:
            with (
                sig_cb.reserve() as o_sig,
                exp_cb.reserve() as o_exp,
                add_cb.reserve() as o_add,
            ):
                # a used by: sigmoid (unary), exp (unary), add (binary)
                sig = ttl.math.sigmoid(av)
                exp = ttl.math.exp(av)
                add = av + bv
                o_sig.store(sig)
                o_exp.store(exp)
                o_add.store(add)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
            ta = ttl.copy(a[0, 0], a_blk)
            tb = ttl.copy(b[0, 0], b_blk)
            ta.wait()
            tb.wait()

    @ttl.datamovement()
    def dm_write():
        with (
            sig_cb.wait() as sig_blk,
            exp_cb.wait() as exp_blk,
            add_cb.wait() as add_blk,
        ):
            ts = ttl.copy(sig_blk, out_sig[0, 0])
            te = ttl.copy(exp_blk, out_exp[0, 0])
            ta = ttl.copy(add_blk, out_add[0, 0])
            ts.wait()
            te.wait()
            ta.wait()


def test_three_consumers(device):
    """Test block arg with 2 unary + 1 binary consumers."""
    a_torch = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0, -2.0, 0.5] + [0.1] * 26] * 32, dtype=torch.bfloat16
    )
    b_torch = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [1.5] * 26] * 32, dtype=torch.bfloat16
    )

    expected_sig = torch.sigmoid(a_torch.float()).to(torch.bfloat16)
    expected_exp = torch.exp(a_torch.float()).to(torch.bfloat16)
    expected_add = (a_torch.float() + b_torch.float()).to(torch.bfloat16)

    a_t = to_dram(a_torch, device)
    b_t = to_dram(b_torch, device)
    out_sig_t = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    out_exp_t = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    out_add_t = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    three_consumers_kernel(a_t, b_t, out_sig_t, out_exp_t, out_add_t)

    result_sig = ttnn.to_torch(out_sig_t)
    result_exp = ttnn.to_torch(out_exp_t)
    result_add = ttnn.to_torch(out_add_t)

    assert_allclose(result_sig.float(), expected_sig.float(), rtol=1e-2, atol=1e-2)
    assert_allclose(result_exp.float(), expected_exp.float(), rtol=1e-1, atol=1e-1)
    assert_allclose(result_add.float(), expected_add.float(), rtol=1e-2, atol=1e-2)


@ttl.kernel(grid=(1, 1))
def square_kernel(x, out):
    """Square pattern: x * x - same value used as both operands."""
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv:
            with out_cb.reserve() as o:
                result = xv * xv  # Same value as both operands
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_square_pattern(device):
    """Test square pattern: x * x."""
    x_torch = torch.tensor(
        [[1.0, 2.0, 3.0, -1.0, -2.0, 0.5] + [1.5] * 26] * 32, dtype=torch.bfloat16
    )

    expected = (x_torch.float() * x_torch.float()).to(torch.bfloat16)

    x_t = to_dram(x_torch, device)
    out_t = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    square_kernel(x_t, out_t)

    result = ttnn.to_torch(out_t)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@ttl.kernel(grid=(1, 1))
def unary_chain_branch_kernel(a, b, out_exp, out_add):
    """Unary chain that branches: abs(a) feeds both exp and add operations."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    exp_cb = ttl.make_circular_buffer_like(out_exp, shape=(1, 1), buffer_factor=2)
    add_cb = ttl.make_circular_buffer_like(out_add, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, b_cb.wait() as bv:
            with exp_cb.reserve() as o_exp, add_cb.reserve() as o_add:
                # Unary chain that branches: abs result used by both exp and add
                abs_val = ttl.math.abs(av)
                exp_val = ttl.math.exp(abs_val)  # unary consumer of abs
                add_val = abs_val + bv  # binary consumer of abs
                o_exp.store(exp_val)
                o_add.store(add_val)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
            ta = ttl.copy(a[0, 0], a_blk)
            tb = ttl.copy(b[0, 0], b_blk)
            ta.wait()
            tb.wait()

    @ttl.datamovement()
    def dm_write():
        with exp_cb.wait() as exp_blk, add_cb.wait() as add_blk:
            te = ttl.copy(exp_blk, out_exp[0, 0])
            ta = ttl.copy(add_blk, out_add[0, 0])
            te.wait()
            ta.wait()


def test_unary_chain_branch(device):
    """Test unary chain that branches to both unary and binary consumers."""
    a_torch = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5] + [0.2] * 26] * 32, dtype=torch.bfloat16
    )
    b_torch = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [2.0] * 26] * 32, dtype=torch.bfloat16
    )

    # abs(a) is used by both exp and add
    abs_a = torch.abs(a_torch.float())
    expected_exp = torch.exp(abs_a).to(torch.bfloat16)
    expected_add = (abs_a + b_torch.float()).to(torch.bfloat16)

    a_t = to_dram(a_torch, device)
    b_t = to_dram(b_torch, device)
    out_exp_t = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    out_add_t = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    unary_chain_branch_kernel(a_t, b_t, out_exp_t, out_add_t)

    result_exp = ttnn.to_torch(out_exp_t)
    result_add = ttnn.to_torch(out_add_t)

    assert_allclose(result_exp.float(), expected_exp.float(), rtol=5e-2, atol=5e-2)
    assert_allclose(result_add.float(), expected_add.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
