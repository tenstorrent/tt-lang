# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Batch matmul with bias: Y = A @ B + C
#
# Based on the matmul example from TTLangSpecification.md.
# Demonstrates multi-dimensional (3D) dataflow buffers and tensor indexing.
#
# Tensor   Torch shape  Shape in tiles
# A        I, M, K      I,  MT, KT      (I batch elements, each a (M, K) matrix)
# B        K, N         KT, NT
# C        M, N         MT, NT
# Y        I, M, N      I,  MT, NT

import torch

import ttl
import ttnn
from utils.correctness import assert_with_ulp

TILE_SIZE = 32


@ttl.kernel(grid=(1, 1))
def matmul_with_bias(
    A: ttnn.Tensor,
    B: ttnn.Tensor,
    C: ttnn.Tensor,
    Y: ttnn.Tensor,
) -> None:
    I = A.shape[0]  # batch dimension (one element per unit, not tile-divided)
    M = A.shape[1]  # rows (tile-aligned)
    K = A.shape[2]  # inner dimension (tile-aligned)
    N = B.shape[1]  # columns (tile-aligned)

    MT = M // TILE_SIZE
    KT = K // TILE_SIZE
    NT = N // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1, 1), buffer_factor=2)

    @ttl.datamovement()
    def matmul_read() -> None:
        for it in range(I):
            for mt in range(MT):
                for nt in range(NT):
                    with c_dfb.reserve() as c_blk:
                        c_xf = ttl.copy(C[mt, nt], c_blk)
                        c_xf.wait()

                    for kt in range(KT):
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            a_xf = ttl.copy(A[it, mt, kt], a_blk)
                            b_xf = ttl.copy(B[kt, nt], b_blk)
                            a_xf.wait()
                            b_xf.wait()

    @ttl.compute()
    def matmul_compute() -> None:
        for _ in range(I):
            for _ in range(MT):
                for _ in range(NT):
                    with y_dfb.reserve() as y_blk:
                        with c_dfb.wait() as c_blk:
                            y_blk.store(c_blk, acc=True)

                        for _ in range(KT):
                            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                                y_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def matmul_write() -> None:
        for it in range(I):
            for mt in range(MT):
                for nt in range(NT):
                    with y_dfb.wait() as y_blk:
                        y_xf = ttl.copy(y_blk, Y[it, mt, nt])
                        y_xf.wait()


def main() -> None:
    I, M, K, N = 2, 64, 96, 128

    A_torch = torch.rand((I, M, K), dtype=torch.float32)
    B_torch = torch.rand((K, N), dtype=torch.float32)
    C_torch = torch.rand((M, N), dtype=torch.float32)

    A = ttnn.from_torch(A_torch)
    B = ttnn.from_torch(B_torch)
    C = ttnn.from_torch(C_torch)
    Y = ttnn.empty((I, M, N), dtype=ttnn.float32)

    matmul_with_bias(A, B, C, Y)

    result = ttnn.to_torch(Y)
    expected = torch.stack([A_torch[i] @ B_torch + C_torch for i in range(I)])

    assert_with_ulp(result, expected, ulp_threshold=1000)
    print("PASSED!")


if __name__ == "__main__":
    main()
