# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the simulator dry-run mode (set_dry_run).

Dry-run mode skips all data movement and numerical computation while
still exercising DFB sequencing, block state machine transitions,
deadlock detection, and copy-wait injection.  The tests verify:

- Results of arithmetic operators are zero tensors of the correct shape.
- ttnn module-level ops (multiply, matmul, relu) return correct shapes.
- ttl.copy().wait() completes without moving payload bytes.
- Block.store() completes without overwriting the destination's bytes.
- A full kernel run (eltwise add pattern) succeeds under dry-run.
- Structural violations (block state machine errors) are still caught.
- Switching dry_run off restores normal computation.
"""

import torch
import pytest

from sim import ttl, ttnn
from sim.context import get_context, reset_context, set_dry_run


@pytest.fixture(autouse=True)
def _reset():
    reset_context()
    yield
    reset_context()


# ---------------------------------------------------------------------------
# Tensor arithmetic — correct shape, zeroed payload
# ---------------------------------------------------------------------------


class TestDryRunTensorArithmetic:
    """Tensor binary/unary ops return zero tensors of the right shape in dry-run."""

    @pytest.fixture(autouse=True)
    def _enable(self):
        set_dry_run(True)

    def test_add_returns_correct_shape(self):
        a = ttnn.from_torch(torch.ones(32, 64))
        b = ttnn.from_torch(torch.ones(32, 64))
        result = a + b
        assert result.shape == (32, 64)
        assert torch.all(result.to_torch() == 0)

    def test_sub_returns_correct_shape(self):
        a = ttnn.from_torch(torch.ones(32, 64))
        b = ttnn.from_torch(torch.ones(32, 64))
        result = a - b
        assert result.shape == (32, 64)
        assert torch.all(result.to_torch() == 0)

    def test_mul_returns_correct_shape(self):
        a = ttnn.from_torch(torch.ones(32, 64))
        result = a * 2.0
        assert result.shape == (32, 64)
        assert torch.all(result.to_torch() == 0)

    def test_matmul_returns_correct_shape(self):
        a = ttnn.from_torch(torch.ones(32, 64))
        b = ttnn.from_torch(torch.ones(64, 96))
        result = a @ b
        assert result.shape == (32, 96)
        assert torch.all(result.to_torch() == 0)

    def test_neg_returns_correct_shape(self):
        a = ttnn.from_torch(torch.ones(32, 32))
        result = -a
        assert result.shape == (32, 32)
        assert torch.all(result.to_torch() == 0)

    def test_abs_returns_correct_shape(self):
        a = ttnn.from_torch(-torch.ones(32, 32))
        result = abs(a)
        assert result.shape == (32, 32)
        assert torch.all(result.to_torch() == 0)

    def test_radd_returns_correct_shape(self):
        a = ttnn.from_torch(torch.ones(32, 32))
        result = 1.0 + a
        assert result.shape == (32, 32)
        assert torch.all(result.to_torch() == 0)

    def test_multiply_module_level(self):
        a = ttnn.from_torch(torch.ones(32, 64))
        b = ttnn.from_torch(torch.ones(32, 64))
        result = ttnn.multiply(a, b)
        assert result.shape == (32, 64)
        assert torch.all(result.to_torch() == 0)

    def test_matmul_module_level(self):
        a = ttnn.from_torch(torch.ones(32, 64))
        b = ttnn.from_torch(torch.ones(64, 96))
        result = ttnn.matmul(a, b)
        assert result.shape == (32, 96)
        assert torch.all(result.to_torch() == 0)

    def test_relu_module_level(self):
        a = ttnn.from_torch(-torch.ones(32, 32))
        result = ttnn.relu(a)
        assert result.shape == (32, 32)
        assert torch.all(result.to_torch() == 0)

    def test_dry_run_off_restores_computation(self):
        """Disabling dry-run restores actual math results."""
        set_dry_run(False)
        a = ttnn.from_torch(torch.ones(32, 32))
        result = a + a
        assert torch.all(result.to_torch() == 2.0)


# ---------------------------------------------------------------------------
# Full kernel run — structural checks still fire, output is not verified
# ---------------------------------------------------------------------------


class TestDryRunKernelRun:
    """A kernel using ttl.copy and ttnn arithmetic runs to completion under dry-run."""

    @pytest.fixture(autouse=True)
    def _enable(self):
        set_dry_run(True)

    def test_eltwise_add_kernel_completes(self):
        """The eltwise-add pattern runs without error; output is not checked."""
        inp_a = ttnn.from_torch(torch.ones(32, 32))
        inp_b = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp_a, inp_b, out):
            a_dfb = ttl.make_dataflow_buffer_like(inp_a, shape=(1, 1), block_count=2)
            b_dfb = ttl.make_dataflow_buffer_like(inp_b, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                a_blk = a_dfb.wait()
                b_blk = b_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(a_blk + b_blk)

            @ttl.datamovement()
            def dm_read():
                a_blk = a_dfb.reserve()
                ttl.copy(inp_a[0, 0], a_blk).wait()
                b_blk = b_dfb.reserve()
                ttl.copy(inp_b[0, 0], b_blk).wait()

            @ttl.datamovement()
            def dm_write():
                ob = out_dfb.wait()
                ttl.copy(ob, out[0, 0]).wait()

        op(inp_a, inp_b, out)
        # Dry-run: output is NOT the sum, but the kernel must complete cleanly.
        # We just verify it ran without error (no assertion on values).

    def test_multi_iteration_kernel_completes(self):
        """Multi-iteration kernel (loop over tiles) completes under dry-run."""
        ITERS = 4
        inp = ttnn.from_torch(torch.ones(ITERS * 32, 32))
        out = ttnn.from_torch(torch.zeros(ITERS * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for _ in range(ITERS):
                    blk = in_dfb.wait()
                    ob = out_dfb.reserve()
                    ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                for i in range(ITERS):
                    blk = in_dfb.reserve()
                    ttl.copy(inp[i, 0], blk).wait()

            @ttl.datamovement()
            def dm_write():
                for i in range(ITERS):
                    blk = out_dfb.wait()
                    ttl.copy(blk, out[i, 0]).wait()

        op(inp, out)

    def test_block_state_violation_still_caught_in_dry_run(self):
        """Dry-run does not suppress block state machine errors."""
        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = in_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = in_dfb.reserve()
                ttl.copy(inp[0, 0], blk).wait()
                # Intentional violation: push twice
                blk.push()
                blk.push()

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        with pytest.raises(RuntimeError):
            op(inp, out)

    def test_copy_payload_not_transferred(self):
        """In dry-run mode the output tensor retains its original value."""
        sentinel = 42.0
        inp = ttnn.from_torch(torch.full((32, 32), sentinel))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = in_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = in_dfb.reserve()
                ttl.copy(inp[0, 0], blk).wait()

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        op(inp, out)
        # The output must NOT have been overwritten with the sentinel value.
        assert torch.all(ttnn.to_torch(out) == 0.0)
