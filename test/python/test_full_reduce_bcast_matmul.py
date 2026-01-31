# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Full distributed reduce -> bcast -> matmul with multicore communication.

Unlike reduce_bcast_matmul_kernel which operates per-core independently,
this kernel does:
1. Global reduce: All cores reduce their A slices, gather to coordinator, sum to global scalar
2. Global broadcast: Coordinator scatters the scalar back to all workers
3. Full matmul with K accumulation: Each core computes partial C, gather and sum

Grid: 4x1 (4 cores in a row)
- A: (ROWS_PER_CORE * 4 * 32, 4 * COLS_PER_CORE * 32) - split across cores by column
- B: (128, 128) - 4x4 tiles, same B used by all cores
- out: (128, 128) - output, computed on coordinator after accumulation

Each core handles ROWS_PER_CORE x COLS_PER_CORE blocks of A (each block is 4x1 tiles).
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


# Core role constant
COORDINATOR = 0
# Number of row/column blocks each core processes
ROWS_PER_CORE = 2
COLS_PER_CORE = 2
# Number of workers (non-coordinator cores)
NUM_WORKERS = 3


@ttl.kernel(grid=(4, 1))
def full_reduce_bcast_matmul_kernel(A, B, scaler, out):
    """
    Full distributed reduce -> bcast -> matmul.

    1. Each core reduces ROWS_PER_CORE x COLS_PER_CORE blocks of A, accumulating locally
    2. Gather local sums to coordinator, compute global sum
    3. Scatter global sum back to all cores
    4. Each core broadcasts scalar to full 4x4 shape, then matmuls with B
    5. Gather partial matmul results to coordinator, accumulate for final C

    Each core reads (ROWS_PER_CORE x COLS_PER_CORE) blocks of A (each 4x1 tiles) for reduce,
    and 4x4 tiles of B for matmul. The broadcast outputs 4x4, matmul does (4x4) @ (4x4) -> (4x4).
    Since all cores broadcast the same global sum and do the same matmul, we accumulate 4
    identical results.
    """
    # Pipes for reduce gather (workers -> coordinator)
    reduce_pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    reduce_pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    reduce_pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))

    # Cleaner scatter pipe (simulator limitation prevents multicast):
    # bcast_pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))
    # Workaround: individual pipes for broadcast
    bcast_pipe1 = ttl.Pipe(src=(0, 0), dst=(1, 0))
    bcast_pipe2 = ttl.Pipe(src=(0, 0), dst=(2, 0))
    bcast_pipe3 = ttl.Pipe(src=(0, 0), dst=(3, 0))

    # Pipes for matmul result gather (workers -> Core 0)
    matmul_pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    matmul_pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    matmul_pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))

    # Input CBs for reduce (A slice: 4 rows x 1 col of tiles = 128x32 elements)
    a_cb = ttl.make_circular_buffer_like(A, shape=(4, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

    # Input CB for matmul (B: 4x4 tiles, each core reads its portion)
    b_cb = ttl.make_circular_buffer_like(B, shape=(4, 4), buffer_factor=2)

    # Reduce intermediate CBs (all 1x1 tiles for scalar handling)
    reduce_out_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    reduce_gather_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=6)
    reduce_acc_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

    # Broadcast CBs (1x1 input, 4x4 output for symmetric broadcast)
    bcast_val_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

    # Matmul CBs (4x4 tiles)
    matmul_out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)
    matmul_gather_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=6)
    matmul_acc_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

    # Output CB
    out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)

        # === Stage 1: Local reduce of A slices (all cores) ===
        # Total blocks per core
        blocks_per_core = ROWS_PER_CORE * COLS_PER_CORE

        # Scaler is reused for all reduces
        with scaler_cb.wait() as s:
            # First block: reduce and copy to accumulator
            with a_cb.wait() as a, reduce_out_cb.reserve() as r:
                reduced = ttl.math.reduce_sum(a, s, r, dims=[0, 1])
                r.store(reduced)
            with reduce_out_cb.wait() as t, reduce_acc_cb.reserve() as acc:
                acc.store(ttl.math.abs(t))

            # Additional blocks: reduce and accumulate
            for _ in range(blocks_per_core - 1):
                with a_cb.wait() as a, reduce_out_cb.reserve() as r:
                    reduced = ttl.math.reduce_sum(a, s, r, dims=[0, 1])
                    r.store(reduced)
                with reduce_out_cb.wait() as t, reduce_acc_cb.wait() as acc, reduce_acc_cb.reserve() as new_acc:
                    new_acc.store(acc + t)
        # Now reduce_acc_cb has local sum for this core

        if x == COORDINATOR:
            # === Coordinator: Gather and accumulate worker reductions ===
            for _ in range(NUM_WORKERS):
                with reduce_gather_cb.wait() as t, reduce_acc_cb.wait() as acc, reduce_acc_cb.reserve() as new_acc:
                    new_acc.store(acc + t)

            # Copy global sum to bcast_val_cb
            with reduce_acc_cb.wait() as acc, bcast_val_cb.reserve() as global_sum:
                global_sum.store(ttl.math.abs(acc))

            # === Stage 2: Broadcast scalar to 4x4 tiles ===
            with bcast_val_cb.wait() as bv, bcast_out_cb.reserve() as bout:
                broadcasted = ttl.math.broadcast(bv, bout, dims=[0, 1])
                bout.store(broadcasted)

            # === Stage 3: Matmul (4x4) @ (4x4) -> (4x4) ===
            with bcast_out_cb.wait() as a_bcast, b_cb.wait() as b, matmul_out_cb.reserve() as c:
                result = ttl.math.matmul(a_bcast, b, c)
                c.store(result)

            # === Stage 4: Accumulate gathered matmul results ===
            with matmul_out_cb.wait() as m0, matmul_acc_cb.reserve() as macc:
                macc.store(ttl.math.abs(m0))

            for _ in range(NUM_WORKERS):
                with matmul_gather_cb.wait() as m, matmul_acc_cb.wait() as acc, matmul_acc_cb.reserve() as new_acc:
                    new_acc.store(acc + m)

            # Copy final result to out_cb
            with matmul_acc_cb.wait() as acc, out_cb.reserve() as final_out:
                final_out.store(ttl.math.abs(acc))

        else:
            # === Workers: Send local sum, receive bcast, compute, send matmul ===
            with reduce_acc_cb.wait() as acc, reduce_gather_cb.reserve() as g:
                g.store(ttl.math.abs(acc))

            # Receive broadcast value (DM handles the pipe receive)

            # Broadcast to 4x4 tiles
            with bcast_val_cb.wait() as bv, bcast_out_cb.reserve() as bout:
                broadcasted = ttl.math.broadcast(bv, bout, dims=[0, 1])
                bout.store(broadcasted)

            # Matmul (4x4) @ (4x4) -> (4x4)
            with bcast_out_cb.wait() as a_bcast, b_cb.wait() as b, matmul_out_cb.reserve() as c:
                result = ttl.math.matmul(a_bcast, b, c)
                c.store(result)

            # Copy matmul result to gather CB for sending
            with matmul_out_cb.wait() as mout, matmul_gather_cb.reserve() as mg:
                mg.store(ttl.math.abs(mout))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)

        # Read scaler first (compute waits on this before consuming a_cb)
        with scaler_cb.reserve() as s_blk:
            tx = ttl.copy(scaler[0, 0], s_blk)
            tx.wait()

        # Each core reads ROWS_PER_CORE x COLS_PER_CORE blocks for reduce
        for row in range(ROWS_PER_CORE):
            for col in range(COLS_PER_CORE):
                row_idx = row * 4  # 4 tile rows per block
                col_idx = x * COLS_PER_CORE + col
                with a_cb.reserve() as a_blk:
                    tx = ttl.copy(A[row_idx:row_idx+4, col_idx:col_idx+1], a_blk)
                    tx.wait()

        # Each core reads full B for matmul (all cores do same matmul with broadcasted A)
        with b_cb.reserve() as b_blk:
            tx = ttl.copy(B[0:4, 0:4], b_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # === Reduce gather: Workers send to coordinator ===
        if x == 1:
            with reduce_gather_cb.wait() as blk:
                tx = ttl.copy(blk, reduce_pipe1)
                tx.wait()
        elif x == 2:
            with reduce_gather_cb.wait() as blk:
                tx = ttl.copy(blk, reduce_pipe2)
                tx.wait()
        elif x == 3:
            with reduce_gather_cb.wait() as blk:
                tx = ttl.copy(blk, reduce_pipe3)
                tx.wait()

        # === Coordinator: Receive reduce gathers ===
        if x == COORDINATOR:
            with reduce_gather_cb.reserve() as blk:
                tx = ttl.copy(reduce_pipe1, blk)
                tx.wait()
            with reduce_gather_cb.reserve() as blk:
                tx = ttl.copy(reduce_pipe2, blk)
                tx.wait()
            with reduce_gather_cb.reserve() as blk:
                tx = ttl.copy(reduce_pipe3, blk)
                tx.wait()

        # === Broadcast scatter (cleaner version, simulator limitation): ===
        # with bcast_pipe.if_src():
        #     blk = bcast_val_cb.wait()
        #     tx = ttl.copy(blk, bcast_pipe)
        #     tx.wait()
        #     bcast_val_cb.pop()
        # with bcast_pipe.if_dst():
        #     with bcast_val_cb.reserve() as blk:
        #         tx = ttl.copy(bcast_pipe, blk)
        #         tx.wait()

        # Workaround: Coordinator sends to each worker individually
        if x == COORDINATOR:
            with bcast_val_cb.wait() as blk:
                tx1 = ttl.copy(blk, bcast_pipe1)
                tx1.wait()
                tx2 = ttl.copy(blk, bcast_pipe2)
                tx2.wait()
                tx3 = ttl.copy(blk, bcast_pipe3)
                tx3.wait()

        # Workers: Receive broadcast
        if x == 1:
            with bcast_val_cb.reserve() as blk:
                tx = ttl.copy(bcast_pipe1, blk)
                tx.wait()
        elif x == 2:
            with bcast_val_cb.reserve() as blk:
                tx = ttl.copy(bcast_pipe2, blk)
                tx.wait()
        elif x == 3:
            with bcast_val_cb.reserve() as blk:
                tx = ttl.copy(bcast_pipe3, blk)
                tx.wait()

        # === Matmul gather: Workers send to coordinator ===
        if x == 1:
            with matmul_gather_cb.wait() as blk:
                tx = ttl.copy(blk, matmul_pipe1)
                tx.wait()
        elif x == 2:
            with matmul_gather_cb.wait() as blk:
                tx = ttl.copy(blk, matmul_pipe2)
                tx.wait()
        elif x == 3:
            with matmul_gather_cb.wait() as blk:
                tx = ttl.copy(blk, matmul_pipe3)
                tx.wait()

        # === Coordinator: Receive matmul gathers and write output ===
        if x == COORDINATOR:
            with matmul_gather_cb.reserve() as blk:
                tx = ttl.copy(matmul_pipe1, blk)
                tx.wait()
            with matmul_gather_cb.reserve() as blk:
                tx = ttl.copy(matmul_pipe2, blk)
                tx.wait()
            with matmul_gather_cb.reserve() as blk:
                tx = ttl.copy(matmul_pipe3, blk)
                tx.wait()

            # Write final output
            with out_cb.wait() as o_blk:
                tx = ttl.copy(o_blk, out[0:4, 0:4])
                tx.wait()


class TestFullReduceBcastMatmul:
    """Tests for full distributed reduce -> bcast -> matmul."""

    def test_uniform_values(self, device):
        """Test with uniform values for easy verification.

        Each core processes ROWS_PER_CORE x COLS_PER_CORE blocks of A.
        """
        # A dimensions based on grid and blocks per core
        A_height = ROWS_PER_CORE * 4 * 32  # ROWS_PER_CORE blocks of 4 tile rows
        A_width = 4 * COLS_PER_CORE * 32   # 4 cores * COLS_PER_CORE columns

        A_torch = torch.full((A_height, A_width), 0.01, dtype=torch.bfloat16)
        B_torch = torch.full((128, 128), 0.01, dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        A = to_l1(A_torch, device)
        B = to_l1(B_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        full_reduce_bcast_matmul_kernel(A, B, scaler, out)
        result = ttnn.to_torch(out).float()

        # Expected: 4 cores * (global_sum * 0.01 * 128)
        global_sum = 0.01 * A_height * A_width
        single_matmul = global_sum * 0.01 * 128
        expected_value = single_matmul * 4  # 4 cores doing same matmul
        expected = torch.tensor(expected_value, dtype=torch.float32)

        print(f"\n=== Full Reduce-Bcast-Matmul Test ===")
        print(f"ROWS_PER_CORE={ROWS_PER_CORE}, COLS_PER_CORE={COLS_PER_CORE}")
        print(f"A shape: {A_torch.shape}")
        print(f"A sum: {A_torch.float().sum().item():.2f}")
        print(f"Global sum: {global_sum:.2f}")
        print(f"Single matmul result: {single_matmul:.2f}")
        print(f"Expected (4 cores summed): {expected_value:.2f}")
        print(f"Actual output [0,0]: {result[0, 0].item():.2f}")
        print(f"Actual output [64,64]: {result[64, 64].item():.2f}")

        # Check output values (only first 4x4 tiles have data, rest is zeros)
        assert_allclose(result[0, 0], expected, rtol=0.15, atol=500)
        assert_allclose(result[64, 64], expected, rtol=0.15, atol=500)
        assert_allclose(result[127, 127], expected, rtol=0.15, atol=500)

    def test_against_pytorch(self, device):
        """Compare against PyTorch reference implementation.

        Note: The kernel sums 4 identical matmul results (one per core),
        so the expected result is 4 * matmul(broadcast(sum(A)), B).

        Uses positive random values since kernel uses abs() for CB-to-CB copies.
        """
        torch.manual_seed(42)

        # A dimensions based on grid and blocks per core
        A_height = ROWS_PER_CORE * 4 * 32
        A_width = 4 * COLS_PER_CORE * 32

        # Use positive values since kernel uses abs() for CB-to-CB copies
        A_torch = torch.rand((A_height, A_width), dtype=torch.bfloat16) * 0.1
        B_torch = torch.rand((128, 128), dtype=torch.bfloat16) * 0.1
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        # PyTorch reference: kernel sums 4 identical matmuls
        global_sum = A_torch.float().sum().item()
        A_bcast = torch.full((128, 128), global_sum, dtype=torch.float32)
        single_matmul = torch.matmul(A_bcast, B_torch.float())
        expected = single_matmul * 4  # 4 cores doing same matmul

        A = to_l1(A_torch, device)
        B = to_l1(B_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        full_reduce_bcast_matmul_kernel(A, B, scaler, out)
        result = ttnn.to_torch(out).float()

        print(f"\n=== PyTorch Comparison Test ===")
        print(f"ROWS_PER_CORE={ROWS_PER_CORE}, COLS_PER_CORE={COLS_PER_CORE}")
        print(f"A shape: {A_torch.shape}")
        print(f"A sum: {global_sum:.2f}")
        print(f"Single matmul [0,0]: {single_matmul[0,0].item():.2f}")
        print(f"Expected (4x) [0,0]: {expected[0,0].item():.2f}")
        print(f"Actual [0,0]: {result[0,0].item():.2f}")

        # PCC comparison
        result_flat = result.flatten()
        expected_flat = expected.flatten()

        result_mean = result_flat.mean()
        expected_mean = expected_flat.mean()
        result_centered = result_flat - result_mean
        expected_centered = expected_flat - expected_mean

        numerator = (result_centered * expected_centered).sum()
        denominator = torch.sqrt((result_centered**2).sum() * (expected_centered**2).sum())
        pcc = numerator / (denominator + 1e-10)

        print(f"PCC: {pcc.item():.6f}")

        assert pcc > 0.95, f"PCC {pcc:.4f} below threshold"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
