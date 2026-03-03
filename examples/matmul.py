# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
    # ---------------------
    # Matmul with bias: Y = A @ B + C
    #
    # Tensor   Torch shape  Shape in tiles
    # A        I, M, K      IT, MT, KT
    # B        K, N         KT, NT
    # C        M, N         MT, NT
    # Y        I, M, N      IT, MT, NT
    #
    TILE_SIZE = 32
    I = A.shape[0]
    M = A.shape[1]
    K = A.shape[2]
    N = B.shape[1]
    IT = I  # batch dim is iterated directly (not tile-divided like M, K, N)
    MT = M // TILE_SIZE
    NT = N // TILE_SIZE
    KT = K // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1, 1))
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1))
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1, 1))

    @ttl.datamovement()
    def matmul_read():
        for it in range(IT):
            for mt in range(MT):
                for nt in range(NT):

                    # acquire c_blk from c_dfb:

                    with c_dfb.reserve() as c_blk:

                        # then copy:

                        c_xf = ttl.copy(C[mt, nt], c_blk)
                        c_xf.wait()

                        # release c_blk

                    for kt in range(KT):

                        # acquire a_blk and b_blk from a_dfb and b_dfb:

                        with (
                            a_dfb.reserve() as a_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            # then copy:

                            a_xf = ttl.copy(A[it, mt, kt], a_blk)
                            b_xf = ttl.copy(B[kt, nt], b_blk)

                            a_xf.wait()
                            b_xf.wait()

                            # release a_blk and b_blk

    @ttl.compute()
    def matmul_compute():
        for _ in range(IT):
            for _ in range(MT):
                for _ in range(NT):

                    # acquire y_blk from y_dfb:

                    with y_dfb.reserve() as y_blk:

                        # acquire c_blk from c_dfb:

                        with c_dfb.wait() as c_blk:

                            # then compute: y = c:

                            y_blk.store(c_blk, acc=True)

                            # release c_blk

                        for _ in range(KT):

                            # acquire a_blk and b_blk from a_dfb and b_dfb:

                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                            ):
                                # then compute y += a @ b:

                                y_blk.store(a_blk @ b_blk, acc=True)

                                # release a_blk and b_blk

                        # release y_blk

    @ttl.datamovement()
    def matmul_write():
        for it in range(IT):
            for mt in range(MT):
                for nt in range(NT):

                    # acquire y_blk from y_dfb:

                    with y_dfb.wait() as y_blk:

                        # then copy:

                        y_xf = ttl.copy(y_blk, Y[it, mt, nt])
                        y_xf.wait()

                        # release y_blk


def main() -> None:
    # I must be small since the batch dim is iterated directly (not tile-divided).
    I, M, K, N = 2, 64, 96, 128

    A_torch = torch.rand((I, M, K), dtype=torch.float32)
    B_torch = torch.rand((K, N), dtype=torch.float32)
    C_torch = torch.rand((M, N), dtype=torch.float32)

    A = ttnn.from_torch(A_torch)
    B = ttnn.from_torch(B_torch)
    C = ttnn.from_torch(C_torch)
    Y = ttnn.empty((I, M, N), dtype=torch.float32)

    matmul_with_bias(A, B, C, Y)

    result = ttnn.to_torch(Y)
    expected = torch.stack([A_torch[i] @ B_torch + C_torch for i in range(I)])

    assert torch.allclose(result, expected, atol=1e-4), "Mismatch!"
    print("PASSED!")


if __name__ == "__main__":
    main()
