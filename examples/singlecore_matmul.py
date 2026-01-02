# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

from sim import ttl, ttnn
from sim.testing import assert_pcc

if TYPE_CHECKING:
    pass


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
                out_blk = out_cb.reserve()
                # Initialize output block to zero for first iteration
                # For subsequent iterations, we accumulate
                for k_idx in range(Kt):
                    a_blk = a_cb.wait()
                    b_blk = b_cb.wait()

                    # Perform matmul and accumulate
                    if k_idx == 0:
                        result = a_blk @ b_blk
                    else:
                        result = out_blk + (a_blk @ b_blk)

                    out_blk.store(result)

                    a_cb.pop()
                    b_cb.pop()

                # Push the accumulated result
                out_cb.push()

    @ttl.datamovement()
    def mm_reader():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    # Reserve blocks for A and B tiles
                    a_blk = a_cb.reserve()
                    b_blk = b_cb.reserve()

                    # Copy tiles using tile coordinates
                    a_wr = ttl.copy(a[m : m + 1, k : k + 1], a_blk)
                    b_wr = ttl.copy(b[k : k + 1, n : n + 1], b_blk)

                    a_wr.wait()
                    b_wr.wait()

                    # Push the tiles to make them visible
                    a_cb.push()
                    b_cb.push()

    @ttl.datamovement()
    def mm_writer():
        for m in range(Mt):
            for n in range(Nt):
                # Wait for computed output tile
                out_blk = out_cb.wait()

                # Copy output tile to result tensor
                out_wr = ttl.copy(out_blk, out[m : m + 1, n : n + 1])
                out_wr.wait()

                # Pop the consumed tile
                out_cb.pop()

    # Execute the program
    ttl.Program(mm_compute, mm_reader, mm_writer)(a, b, out)


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
    print("tt_lang_singlecore_matmul: success")


if __name__ == "__main__":
    main()
