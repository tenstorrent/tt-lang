# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the simulator dry-run mode (set_dry_run).

Dry-run mode skips all data movement and numerical computation while
still exercising DFB sequencing, block state machine transitions,
deadlock detection, and copy-wait injection.  The tests verify:

- Results of arithmetic operators are zero tensors of the correct shape.
- ttnn module-level ops (multiply, matmul, relu) return correct shapes.
- Matmul derives batched/broadcast output shapes -- and rejects incompatible
  dims -- exactly as the non-dry path does.
- ttl.copy().wait() completes without moving payload bytes.
- Block.store() completes without overwriting the destination's bytes.
- ttl.block / ttl.math ops return correctly shaped blocks and propagate the
  source layout (so chained layout checks remain meaningful).
- DataflowBuffer.reserve() slots carry the buffer's layout under dry-run.
- A full kernel run (eltwise add pattern) succeeds under dry-run.
- Structural violations (block state machine errors, deadlock) are still caught.
- Pipe send/receive bookkeeping and shape checks run symmetrically under dry-run.
- Switching dry_run off restores normal computation.
"""

import torch
import pytest
from test_utils import make_element_for_buffer_shape, make_ones_tile, make_rand_tensor

from sim import ttl, ttnn
from sim.context import get_context, reset_context, set_dry_run
from sim.copy import copy
from sim.dfb import Block, DataflowBuffer
from sim.pipe import Pipe
from sim.ttnnsim import ROW_MAJOR_LAYOUT, TILE_LAYOUT, Tensor


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

    def test_matmul_batched_shape(self):
        """Batched matmul derives the correct (batch, m, n) output shape."""
        a = ttnn.from_torch(torch.ones(4, 32, 64))
        b = ttnn.from_torch(torch.ones(4, 64, 96))
        result = a @ b
        assert result.shape == (4, 32, 96)

    def test_matmul_broadcast_batch_shape(self):
        """Matmul broadcasts a 2-D operand across the batch dim of the other."""
        a = ttnn.from_torch(torch.ones(2, 3, 32, 64))
        b = ttnn.from_torch(torch.ones(64, 96))
        result = a @ b
        assert result.shape == (2, 3, 32, 96)

    def test_matmul_dry_shape_matches_real(self):
        """The dry-run matmul shape must match the shape the non-dry path derives."""
        a_t = torch.ones(2, 3, 32, 64)
        b_t = torch.ones(2, 1, 64, 96)
        set_dry_run(False)
        real_shape = (ttnn.from_torch(a_t) @ ttnn.from_torch(b_t)).shape
        set_dry_run(True)
        dry_shape = (ttnn.from_torch(a_t) @ ttnn.from_torch(b_t)).shape
        assert dry_shape == real_shape

    def test_matmul_incompatible_dims_raises(self):
        """Dry-run shape derivation rejects incompatible dims like the real path.

        The meta-tensor matmul reproduces torch's contraction-dim check, so a
        structural error is surfaced even though no values are computed.
        """
        a = ttnn.from_torch(torch.ones(32, 64))
        b = ttnn.from_torch(torch.ones(32, 96))  # inner dims 64 vs 32 mismatch
        with pytest.raises(RuntimeError):
            _ = a @ b

    def test_relu_module_level(self):
        # Positive input: real relu would preserve the value, so an all-zero
        # result confirms dry-run skipped the computation (rather than relu
        # merely clamping a negative input to zero).
        a = ttnn.from_torch(torch.ones(32, 32))
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

    def test_deadlock_still_detected_in_dry_run(self):
        """Dry-run does not suppress deadlock detection.

        The compute kernel waits on a DFB that no datamovement kernel ever
        fills, so all kernels block forever; the scheduler must still surface
        this as a deadlock.
        """
        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # in_dfb is never produced into -> blocks forever.
                blk = in_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                # Waits on out_dfb (never produced) instead of filling in_dfb,
                # so compute can never proceed.
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

            @ttl.datamovement()
            def dm_write():
                # out_dfb is never produced into -> blocks forever.
                ob = out_dfb.wait()
                ttl.copy(ob, out[0, 0]).wait()

        with pytest.raises(RuntimeError, match="[Dd]eadlock"):
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


# ---------------------------------------------------------------------------
# Block-level ops — shape-changing ops and layout propagation
# ---------------------------------------------------------------------------


def _tile_block(shape, layout=TILE_LAYOUT):
    """Build a Block of the given grid shape backed by single-element tiles."""
    import math as _math

    tiles = [Tensor(torch.zeros(1, 1), layout) for _ in range(_math.prod(shape))]
    return Block.from_list(tiles, shape=shape)


class TestDryRunBlockOps:
    """ttl.block / ttl.math ops return correctly shaped, correctly laid-out blocks."""

    @pytest.fixture(autouse=True)
    def _enable(self):
        set_dry_run(True)

    def test_squeeze_shape(self):
        blk = _tile_block((1, 3, 1, 4))
        result = ttl.block.squeeze(blk, dims=[0, 2])
        assert result.shape == (3, 4)

    def test_unsqueeze_shape(self):
        blk = _tile_block((3, 4))
        result = ttl.block.unsqueeze(blk, dims=[0, 2])
        assert result.shape == (1, 3, 1, 4)

    def test_transpose_shape(self):
        blk = _tile_block((2, 5))
        result = ttl.block.transpose(blk)
        assert result.shape == (5, 2)

    def test_fill_shape(self):
        result = ttl.block.fill(1.0, (2, 3))
        assert result.shape == (2, 3)

    def test_math_unary_shape(self):
        blk = _tile_block((2, 3))
        result = ttl.math.sqrt(blk)
        assert result.shape == (2, 3)

    def test_math_binary_shape(self):
        a = _tile_block((2, 3))
        b = _tile_block((2, 3))
        result = ttl.math.max(a, b)
        assert result.shape == (2, 3)

    def test_unary_op_preserves_tile_layout(self):
        blk = _tile_block((1, 1), TILE_LAYOUT)
        assert ttl.math.relu(blk).layout == TILE_LAYOUT

    def test_unary_op_preserves_row_major_layout(self):
        """Dry-run results must carry the source layout, not default to TILE."""
        blk = _tile_block((1, 1), ROW_MAJOR_LAYOUT)
        assert ttl.math.relu(blk).layout == ROW_MAJOR_LAYOUT

    def test_binary_op_preserves_row_major_layout(self):
        a = _tile_block((1, 1), ROW_MAJOR_LAYOUT)
        b = _tile_block((1, 1), ROW_MAJOR_LAYOUT)
        assert (a + b).layout == ROW_MAJOR_LAYOUT

    def test_layout_mismatch_caught_after_chained_op(self):
        """A chained dry-run result keeps its layout so a later mismatch is caught.

        Before layout propagation, ``rm1 + rm2`` produced a TILE-layout result,
        which masked the mismatch when combined with a row-major block. With the
        layout propagated, ``check_same_layout`` fires as it would in a real run.
        """
        rm1 = _tile_block((1, 1), ROW_MAJOR_LAYOUT)
        rm2 = _tile_block((1, 1), ROW_MAJOR_LAYOUT)
        tile = _tile_block((1, 1), TILE_LAYOUT)
        chained = rm1 + rm2
        assert chained.layout == ROW_MAJOR_LAYOUT
        with pytest.raises(ValueError, match="same layout"):
            _ = chained + tile


# ---------------------------------------------------------------------------
# Block.store — dry-run is a structural no-op on payload bytes
# ---------------------------------------------------------------------------


def test_store_does_not_overwrite_destination_in_dry_run(compute_kernel_context):
    """Block.store() skips the payload copy under dry-run.

    The destination slot is reserved with dry-run OFF so it has real backing
    bytes (zeros); storing a non-zero source under dry-run must leave those
    bytes untouched, confirming store() is a structural no-op on data while its
    state-machine transition still fires.
    """
    element = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)

    set_dry_run(False)
    dst = dfb.reserve()
    assert torch.all(dst._buf.to_torch() == 0.0)  # fresh slot starts zeroed

    set_dry_run(True)
    dst.store(Block.from_tensor(make_ones_tile()))  # non-zero source

    # The destination is still zeros: the dry-run store moved no bytes.
    assert torch.all(dst._buf.to_torch() == 0.0)


# ---------------------------------------------------------------------------
# DataflowBuffer.reserve — dry-run slot carries the buffer's layout
# ---------------------------------------------------------------------------


def test_reserve_propagates_row_major_layout_in_dry_run(dm_kernel_context):
    """A reserved block in dry-run reports the buffer's layout, not the TILE default.

    The reserve slot uses a layout-specific sentinel so chained layout checks
    (check_same_layout) stay meaningful even when no payload is allocated.
    """
    set_dry_run(True)

    rm_dfb = DataflowBuffer(
        likeness_tensor=Tensor(torch.zeros(32, 32), ROW_MAJOR_LAYOUT),
        shape=(1, 1),
        block_count=2,
    )
    blk = rm_dfb.reserve()
    assert blk.layout == ROW_MAJOR_LAYOUT


# ---------------------------------------------------------------------------
# Pipe send/receive symmetry under dry-run
# ---------------------------------------------------------------------------


def test_pipe_send_receive_drains_queue_in_dry_run(dm_kernel_context):
    """Dry-run pipe receive performs queue bookkeeping symmetrically with send.

    The send enqueues a zero-payload marker and the receive must dequeue it
    (rather than being skipped wholesale), so the pipe queue ends up empty.

    The ``dm_kernel_context`` fixture installs the scheduler/kernel context;
    the module-level autouse ``_reset`` fixture has already reset the context
    before it ran, so dry-run is enabled here without resetting again.
    """
    set_dry_run(True)

    pipe = Pipe(6000, 6001)
    src_dfb = DataflowBuffer(
        likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
    )
    dst_dfb = DataflowBuffer(
        likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
    )

    with src_dfb.reserve() as src_block:
        copy(make_ones_tile(), src_block).wait()
    with src_dfb.wait() as src_block:
        copy(src_block, pipe).wait()

    pipe_buffer = get_context().copy_state.pipe_buffer
    assert len(pipe_buffer[pipe]["queue"]) == 1

    with dst_dfb.reserve() as dst_block:
        copy(pipe, dst_block).wait()

    # The receive drained the marker; without symmetric bookkeeping the message
    # would still be sitting in the queue.
    assert len(pipe_buffer[pipe]["queue"]) == 0


def test_pipe_shape_mismatch_caught_in_dry_run(dm_kernel_context):
    """The structural pipe shape check runs in dry-run, not just in normal mode.

    The sent block has tile-grid shape (2, 1); receiving into a (1, 1) block is
    a structural error that must be reported even though no payload is moved.
    """
    set_dry_run(True)

    pipe = Pipe(6100, 6101)
    src_dfb = DataflowBuffer(
        likeness_tensor=make_element_for_buffer_shape((2, 1)),
        shape=(2, 1),
        block_count=2,
    )
    dst_dfb = DataflowBuffer(
        likeness_tensor=make_element_for_buffer_shape((1, 1)),
        shape=(1, 1),
        block_count=2,
    )

    with src_dfb.reserve() as src_block:
        copy(make_rand_tensor(64, 32), src_block).wait()
    with src_dfb.wait() as src_block:
        copy(src_block, pipe).wait()

    with pytest.raises(ValueError, match="does not match pipe data shape"):
        with dst_dfb.reserve() as dst_block:
            copy(pipe, dst_block).wait()
