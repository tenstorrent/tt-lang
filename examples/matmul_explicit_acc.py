# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Matmul with bias using explicit CB-spilled accumulation.
# Y = A @ B + C
#
# Accumulation across the K dimension is performed explicitly using a
# temporary dataflow buffer (tmp_dfb) rather than store(..., acc=True).
# Each K-step result is packed to the temp CB and reloaded on the next
# iteration, with element-wise addition for accumulation. The bias C is
# added in a separate compute phase after K-accumulation completes.
#
# This pattern maps directly to the tt-metal bmm_large_block_zm.cpp
# approach (pack partials to cb_intermed0, reload on next K-block).

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from sim import ttl, ttnn


@ttl.kernel(grid=(1, 1))
def matmul_with_bias(
    A: ttnn.Tensor,
    B: ttnn.Tensor,
    C: ttnn.Tensor,
    Y: ttnn.Tensor,
) -> None:
    TILE_SIZE = 32
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    MT = M // TILE_SIZE
    KT = K // TILE_SIZE
    NT = N // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1))
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1))
    # Temporary CB for K-accumulation (compute-local, no DM thread touches it)
    tmp_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1))

    @ttl.datamovement()
    def read():
        for mt in range(MT):
            for nt in range(NT):
                with c_dfb.reserve() as c_blk:
                    ttl.copy(C[mt, nt], c_blk).wait()

                for kt in range(KT):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        ttl.copy(A[mt, kt], a_blk).wait()
                        ttl.copy(B[kt, nt], b_blk).wait()

    @ttl.compute()
    def compute():
        for _ in range(MT):
            for _ in range(NT):
                # Phase 1: Accumulate A@B over K into tmp_dfb.
                # First K-step: store matmul result directly.
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                    with tmp_dfb.reserve() as tmp_blk:
                        tmp_blk.store(a_blk @ b_blk)

                # Remaining K-steps: reload partial, add new product, store back.
                for _ in range(KT - 1):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        with tmp_dfb.wait() as prev_blk:
                            with tmp_dfb.reserve() as tmp_blk:
                                tmp_blk.store(prev_blk + (a_blk @ b_blk))

                # Phase 2: Add bias C to accumulated result, store to output.
                with tmp_dfb.wait() as acc_blk, c_dfb.wait() as c_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(acc_blk + c_blk)

    @ttl.datamovement()
    def write():
        for mt in range(MT):
            for nt in range(NT):
                with y_dfb.wait() as y_blk:
                    ttl.copy(y_blk, Y[mt, nt]).wait()


def main() -> None:
    M, K, N = 64, 96, 64

    A_torch = torch.rand((M, K), dtype=torch.float32)
    B_torch = torch.rand((K, N), dtype=torch.float32)
    C_torch = torch.rand((M, N), dtype=torch.float32)

    A = ttnn.from_torch(A_torch)
    B = ttnn.from_torch(B_torch)
    C = ttnn.from_torch(C_torch)
    Y = ttnn.empty((M, N), dtype=torch.float32)

    matmul_with_bias(A, B, C, Y)

    result = ttnn.to_torch(Y)
    expected = A_torch @ B_torch + C_torch

    assert torch.allclose(
        result, expected, atol=1e-4
    ), f"Mismatch! Max diff: {(result - expected).abs().max().item()}"
    print("PASSED!")


if __name__ == "__main__":
    main()
