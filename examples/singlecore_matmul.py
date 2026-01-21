# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore

import ttl
import ttnn
from sim.testing import assert_pcc


@ttl.kernel(grid=(1, 1))
def tt_lang_singlecore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."

    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    buffering_factor = 2
    a_cb = ttl.make_circular_buffer_like(
        a, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(1, 1), buffer_factor=buffering_factor
    )
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )

    @ttl.compute()
    def mm_compute():
        for _ in range(Mt):
            for _ in range(Nt):
                # Reserve output block once for the entire K accumulation
                # The reserved block is automatically initialized with zeros
                with out_cb.reserve() as out_blk:
                    # Accumulate over K dimension
                    for _ in range(Kt):
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                            # Perform matmul and accumulate using acc=True
                            out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    # Reserve blocks for A and B tiles
                    with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                        # Copy tiles using tile coordinates
                        a_wr = ttl.copy(a[m, k], a_blk)
                        b_wr = ttl.copy(b[k, n], b_blk)

                        a_wr.wait()
                        b_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        for m in range(Mt):
            for n in range(Nt):
                # Wait for computed output tile
                with out_cb.wait() as out_blk:
                    # Copy output tile to result tensor
                    out_wr = ttl.copy(out_blk, out[m, n])
                    out_wr.wait()

def main() -> None:
    # Test with reasonably sized matrices that are multiples of tile size
    M, K, N = 128, 256, 64
    a = ttnn.rand((M, K), dtype=ttnn.float32)
    b = ttnn.rand((K, N), dtype=ttnn.float32)
    out = ttnn.empty((M, N), dtype=ttnn.float32)

    print(f"Matrix multiplication: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"Tiles: A={M//32}x{K//32}, B={K//32}x{N//32}, Out={M//32}x{N//32}")

    tt_lang_singlecore_matmul(a, b, out)

    # Compute golden result
    golden = a @ b

    # Verify correctness with relaxed tolerance for matmul
    assert_pcc(golden, out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    main()
