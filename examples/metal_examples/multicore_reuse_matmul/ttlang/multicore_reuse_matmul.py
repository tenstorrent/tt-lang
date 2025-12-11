# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest
import torch

from ttl import Program, make_circular_buffer_like, copy, core
from metal_examples.utils import assert_with_ulp


@ttl.kernel(grid=(13, 10))
def tt_lang_multicore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    K_block_size = 2  # k dim block size

    num_cores_x, num_cores_y = ttl.grid_size(dims=2)
    # unused subblock sizes, to be determined by compiler, but using helper function to get better simultaneous comparisons
    (per_core_M, per_core_N, _, _) = get_large_matmul_params(
        Mt, Nt, num_cores_y, num_cores_x, K_block_size
    )
    assert per_core_M != 0, "get_large_matmul_params was not able to find a solution"
    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")
    assert Mt % per_core_M == 0, "per_core_M must divide Mt"
    assert Nt % per_core_N == 0, "per_core_N must divide Nt"
    assert Kt % K_block_size == 0, "K_block_size must divide Kt"
    num_total_blocks = (Mt // per_core_M) * (Nt // per_core_N)
    assert (
        num_total_blocks <= num_cores_x * num_cores_y
    ), "number of total blocks must be less than or equal to num cores"

    buffering_factor = 2
    a_cb = make_circular_buffer_like(
        a, shape=(per_core_M, K_block_size), buffer_factor=buffering_factor
    )
    b_cb = make_circular_buffer_like(
        b, shape=(K_block_size, per_core_N), buffer_factor=buffering_factor
    )
    # non buffered output, matching metal implementation
    out_cb = make_circular_buffer_like(
        out, shape=(per_core_M, per_core_N), buffer_factor=1
    )

    @ttl.compute()
    def mm_compute():
        with out_cb.reserve() as out_blk:  # per_core_M * per_core_N
            for _ in range(Kt // K_block_size):
                with a_cb.wait() as a_blk, b_cb.wait() as b_blk:  # a per_core_M x K_block_size, b K_block_size x per_core_N
                    out_blk.store(out_blk + a_blk @ b_blk)

    @ttl.datamovement()
    def mm_reader():
        core_x, core_y = ttl.core(dims=2)
        out_row = per_core_M * core_y
        out_col = per_core_N * core_x
        for block in range(Kt // K_block_size):
            k = block * K_block_size
            with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                a_wr = copy(
                    a[out_row : (out_row + per_core_M), k : (k + K_block_size)], a_blk
                )
                b_wr = copy(
                    b[k : (k + K_block_size), out_col : (out_col + per_core_N)], b_blk
                )
                a_wr.wait()
                b_wr.wait()

    # blocking only occurs on the k dim, so each core writes its entire output block at once
    @ttl.datamovement()
    def mm_writer():
        core_x, core_y = ttl.core(dims=2)
        out_row = per_core_M * core_y
        out_col = per_core_N * core_x
        with out_cb.wait() as out_blk:
            out_wr = copy(
                out_blk,
                out[out_row : (out_row + per_core_M), out_col : (out_col + per_core_N)],
            )
            out_wr.wait()

    return Program(mm_compute, mm_reader, mm_writer)(a, b, out)


@pytest.mark.parametrize("M,K,N", [(256, 256, 256), (512, 512, 512)])
def test_multicore_matmul_tt_lang(M, K, N):
    """Test multicore matmul kernel."""
    device = ttnn.open_device(device_id=0)
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_multicore_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)

    ttnn.close_device(device)
