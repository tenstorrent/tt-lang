# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Matmul with bias using explicit CB-spilled accumulation.
# Y = A @ B + C
#
# Data movement access pattern:
#   The DM reader iterates output tiles in (mt, nt) order. For each output
#   tile, it first loads the bias tile C[mt, nt], then streams all KT
#   A-column/B-row tiles: A[mt, 0..KT-1] paired with B[0..KT-1, nt].
#   Each (A, B) pair is a 1x1 tile DFB push that the compute engine
#   consumes in lock-step. The DM writer reads the final output tile
#   Y[mt, nt] after K-accumulation and bias addition complete.
#
# Compute accumulation pattern:
#   Accumulation across the K dimension is performed explicitly using a
#   temporary dataflow buffer (tmp_dfb) rather than store(..., acc=True).
#   Each K-step result is packed to the temp CB and reloaded on the next
#   iteration, with element-wise addition for accumulation. The bias C is
#   added in a separate compute phase after K-accumulation completes.
#
# This pattern maps directly to the tt-metal bmm_large_block_zm.cpp
# approach (pack partials to cb_intermed0, reload on next K-block).

import torch
import ttl
import ttnn

TILE_SIZE = 32


@ttl.kernel(grid=(1, 1))
def matmul_with_bias(
    A: ttnn.Tensor,
    B: ttnn.Tensor,
    C: ttnn.Tensor,
    Y: ttnn.Tensor,
) -> None:
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    MT = M // TILE_SIZE
    KT = K // TILE_SIZE
    NT = N // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1))
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1))
    # Compute-local CBs for K-accumulation (DM threads do not touch these).
    partial_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1))

    @ttl.datamovement()
    def read():
        for mt in range(MT):
            for nt in range(NT):
                with c_dfb.reserve() as c_blk:
                    tx = ttl.copy(C[mt, nt], c_blk)
                    tx.wait()

                for kt in range(KT):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        tx_a = ttl.copy(A[mt, kt], a_blk)
                        tx_a.wait()
                        tx_b = ttl.copy(B[kt, nt], b_blk)
                        tx_b.wait()

    @ttl.compute()
    def compute():
        for _ in range(MT):
            for _ in range(NT):
                # First K-step: matmul directly to accumulator.
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(a_blk @ b_blk)

                # Remaining K-steps: matmul to partial, add with accumulator.
                for _ in range(KT - 1):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        with partial_dfb.reserve() as p:
                            p.store(a_blk @ b_blk)

                    with partial_dfb.wait() as new, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + new)

                # Add bias C to accumulated result, store to output.
                with acc_dfb.wait() as acc_blk, c_dfb.wait() as c_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(acc_blk + c_blk)

    @ttl.datamovement()
    def write():
        for mt in range(MT):
            for nt in range(NT):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, Y[mt, nt])
                    tx.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        M, K, N = 64, 96, 64

        A_torch = torch.randn((M, K), dtype=torch.bfloat16)
        B_torch = torch.randn((K, N), dtype=torch.bfloat16)
        C_torch = torch.randn((M, N), dtype=torch.bfloat16)

        A = ttnn.from_torch(
            A_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        B = ttnn.from_torch(
            B_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        C = ttnn.from_torch(
            C_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        Y = ttnn.from_torch(
            torch.zeros((M, N), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        matmul_with_bias(A, B, C, Y)

        result = ttnn.to_torch(Y)
        expected = A_torch @ B_torch + C_torch

        pcc = torch.corrcoef(
            torch.stack([result.flatten().float(), expected.flatten().float()])
        )[0, 1].item()
        assert pcc > 0.99, (
            f"PCC {pcc:.6f} < 0.99 for matmul+bias. "
            f"Max diff: {(result - expected).abs().max().item()}"
        )
        print("PASSED!")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
