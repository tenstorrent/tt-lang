# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SSA conditional variable reassignment.

Exercises patterns where variables are assigned or reassigned inside
conditional (if/else) blocks and must remain visible after the block exits.
The compiler propagates SSA values through scf.if regions by yielding
modified values from each branch and rebinding the results in the outer
scope (fix for ISSUE #380).
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_l1


# =============================================================================
# Pattern 1: If-only reassignment in datamovement
# =============================================================================
# Variable b starts at one value and is conditionally reassigned to another.
# After the if block, the reassigned value must be visible.
#
# The tracer pre-scans the branch AST for assigned names, yields modified
# values from the scf.if, and rebinds results in the outer scope.


@ttl.operation(grid=(1, 1))
def dm_cond_reassign_f32_kernel(inp, out):
    """Conditionally reassign a scalar and write the result.

    inp[0,0] = initial b, inp[0,1] = cond_val, inp[0,2] = new b,
    inp[0,3] = threshold.  If cond_val < threshold, b becomes inp[0,2].
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                b = ttl.raw_element_read(rblk, 0, 0)
                cond_val = ttl.raw_element_read(rblk, 0, 1)
                threshold = ttl.raw_element_read(rblk, 0, 3)
                if cond_val < threshold:
                    b = ttl.raw_element_read(rblk, 0, 2)
                ttl.raw_element_write(wblk, 0, 0, b)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def test_dm_cond_reassign_f32(device):
    """f32 variable reassigned inside if-only block is visible after."""
    inp_torch = torch.zeros(32, 32, dtype=torch.float32)
    inp_torch[0, 0] = 5.0  # initial b
    inp_torch[0, 1] = 3.0  # cond_val
    inp_torch[0, 2] = 7.0  # new b when condition is true
    inp_torch[0, 3] = 4.0  # threshold (3.0 < 4.0 is true)

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    dm_cond_reassign_f32_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(7.0, abs=1e-5)


# =============================================================================
# Pattern 2: If-else with variable pre-initialised then reassigned in branches
# =============================================================================
# Variable is initialised before the if/else and reassigned inside both
# branches, then used after the block.  The pre-initialisation provides a
# known type so scf.if can declare proper result types.


@ttl.operation(grid=(1, 1))
def dm_if_else_new_var_f32_kernel(inp, out):
    """Pre-init a variable, reassign in both if/else branches, use after.

    inp[0,0] = a (also used as dummy init for b),
    inp[0,1] = b_true, inp[0,2] = threshold,
    inp[0,3] = b_false.  If a < threshold, b = b_true, else b = b_false.
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                a = ttl.raw_element_read(rblk, 0, 0)
                threshold = ttl.raw_element_read(rblk, 0, 2)
                b = ttl.raw_element_read(rblk, 0, 0)
                if a < threshold:
                    b = ttl.raw_element_read(rblk, 0, 1)
                else:
                    b = ttl.raw_element_read(rblk, 0, 3)
                ttl.raw_element_write(wblk, 0, 0, b)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def test_dm_if_else_new_var_f32(device):
    """f32 variable first defined inside both if/else branches is accessible after."""
    inp_torch = torch.zeros(32, 32, dtype=torch.float32)
    inp_torch[0, 0] = 2.0  # a (condition value)
    inp_torch[0, 1] = 9.0  # b_true
    inp_torch[0, 2] = 5.0  # threshold (2.0 < 5.0 is true)
    inp_torch[0, 3] = 1.0  # b_false

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    dm_if_else_new_var_f32_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(9.0, abs=1e-5)


# =============================================================================
# Pattern 3: If-else reassignment of existing variable
# =============================================================================
# Variable exists before the if/else and is reassigned to different values
# in each branch.  This is a phi-node in SSA form: the post-if value must
# be the result of whichever branch executed.


@ttl.operation(grid=(1, 1))
def dm_if_else_reassign_f32_kernel(inp, out):
    """Reassign existing variable in both if/else branches.

    inp[0,0] = initial b, inp[0,1] = a, inp[0,2] = b_true,
    inp[0,3] = b_false, inp[0,4] = threshold.
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                b = ttl.raw_element_read(rblk, 0, 0)
                a = ttl.raw_element_read(rblk, 0, 1)
                threshold = ttl.raw_element_read(rblk, 0, 4)
                if a < threshold:
                    b = ttl.raw_element_read(rblk, 0, 2)
                else:
                    b = ttl.raw_element_read(rblk, 0, 3)
                ttl.raw_element_write(wblk, 0, 0, b)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def test_dm_if_else_reassign_true_branch_f32(device):
    """f32 existing variable reassigned in both branches -- true branch taken."""
    inp_torch = torch.zeros(32, 32, dtype=torch.float32)
    inp_torch[0, 0] = 5.0  # initial b
    inp_torch[0, 1] = 2.0  # a (condition value)
    inp_torch[0, 2] = 7.0  # b_true
    inp_torch[0, 3] = 3.0  # b_false
    inp_torch[0, 4] = 6.0  # threshold (2.0 < 6.0 is true)

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    dm_if_else_reassign_f32_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(7.0, abs=1e-5)


def test_dm_if_else_reassign_false_branch_f32(device):
    """f32 existing variable reassigned in both branches -- false branch taken."""
    inp_torch = torch.zeros(32, 32, dtype=torch.float32)
    inp_torch[0, 0] = 5.0  # initial b
    inp_torch[0, 1] = 8.0  # a (condition value)
    inp_torch[0, 2] = 7.0  # b_true
    inp_torch[0, 3] = 3.0  # b_false
    inp_torch[0, 4] = 6.0  # threshold (8.0 < 6.0 is false)

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.float32), device)

    dm_if_else_reassign_f32_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    assert result[0, 0].item() == pytest.approx(3.0, abs=1e-5)


# =============================================================================
# Pattern 4: DM if-else with ttl.copy (ISSUE #380 reproducer)
# =============================================================================
# Transfer handle assigned in both if/else branches, then waited on after
# the block.  No raw_element ops -- purely ttl.copy + ttl.node().
#
# The tracer yields transfer handles from both scf.if branches so
# tx.wait() after the block uses the correct SSA value.


@ttl.operation(grid=(2, 1))
def dm_if_else_copy_kernel(inp, out):
    """Per-node copy: node 0 reads/writes tile [0,0], node 1 reads/writes [0,1]."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as a, out_dfb.reserve() as o:
            o.store(a)

    @ttl.datamovement()
    def dm_read():
        node_x, _ = ttl.node(dims=2)
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            if node_x == 1:
                tx = ttl.copy(inp[0, 1], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        node_x, _ = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            if node_x == 1:
                tx = ttl.copy(blk, out[0, 1])
            tx.wait()


def test_dm_if_else_copy(device):
    """Transfer handle assigned in if-only with pre-init is usable after."""
    inp = to_l1(torch.full((32, 64), 0.5, dtype=torch.bfloat16), device)
    out = to_l1(torch.zeros((32, 64), dtype=torch.bfloat16), device)

    dm_if_else_copy_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    # Compute is passthrough (conditional stores blocked by ISSUE #683).
    # Once #683 is fixed, replace passthrough with the commented-out compute
    # above and use these assertions instead:
    #   expected_0 = torch.exp(torch.tensor(0.5)).item()
    #   expected_1 = torch.tanh(torch.tensor(0.5)).item()
    #   assert result[0, 0].item() == pytest.approx(expected_0, abs=1e-2)
    #   assert result[0, 32].item() == pytest.approx(expected_1, abs=1e-2)
    assert result[0, 0].item() == pytest.approx(0.5, abs=1e-2)
    assert result[0, 32].item() == pytest.approx(0.5, abs=1e-2)


# =============================================================================
# Pattern 5: Conditional count variable as loop bound (ISSUE #380 comment)
# =============================================================================
# count is set inside if/elif branches, then used as the bound of a
# for-range loop.  Because count stays 0 in the outer scope, the
# canonicalizer removes the zero-trip loop and produces empty kernels.
#
# Adapted from the reproducer posted in the ISSUE #380 comments:
# https://github.com/tenstorrent/tt-lang/issues/380


@ttl.operation(grid=(2, 1))
def dm_cond_count_kernel(inp, out):
    """Conditional iteration count: node 0 copies 2 tiles, node 1 copies 1."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _ = ttl.node(dims=2)
        count = 0
        if node_x < 1:
            count = 2
        elif node_x < 2:
            count = 1
        for _ in range(count):
            with inp_dfb.wait() as a, out_dfb.reserve() as o:
                o.store(a)

    @ttl.datamovement()
    def dm_read():
        node_x, _ = ttl.node(dims=2)
        count = 0
        if node_x < 1:
            count = 2
        elif node_x < 2:
            count = 1
        for i in range(count):
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[0, i], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        node_x, _ = ttl.node(dims=2)
        count = 0
        if node_x < 1:
            count = 2
        elif node_x < 2:
            count = 1
        for i in range(count):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, i])
                tx.wait()


def test_dm_cond_count_loop(device):
    """Conditionally-set loop count must survive if/elif scope exit."""
    inp = to_l1(torch.full((32, 64), 0.5, dtype=torch.bfloat16), device)
    out = to_l1(torch.zeros((32, 64), dtype=torch.bfloat16), device)

    dm_cond_count_kernel(inp, out)
    result = ttnn.to_torch(out).float()

    # Node 0 copies 2 tiles: out[0,0] and out[0,1] should be ~0.5.
    # With the bug, count stays 0, kernels are no-ops, output stays zero.
    assert result[0, 0].item() == pytest.approx(0.5, abs=1e-2)
    assert result[0, 32].item() == pytest.approx(0.5, abs=1e-2)
