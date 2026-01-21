# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from metal_examples.utils import assert_with_ulp
from ttl import Program, copy, core, make_circular_buffer_like


@ttl.kernel(grid=(13, 10))
def tt_lang_multicore_reuse_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    assert b.shape[1] == out.shape[1], "Output matrix has incorrect number of columns."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    K_block_size = 2  # k dim block size

    num_cores_x, num_cores_y = ttl.grid_size(dims=2)
    # unused subblock sizes, to be determined by compiler, but using helper function to get better simultaneous comparisons
    block_params = get_large_matmul_params(
        Mt, Nt, num_cores_y, num_cores_x, K_block_size
    )
    per_core_M = block_params.block_h
    per_core_N = block_params.block_w
    assert per_core_M != 0, "get_large_matmul_params was not able to find a solution"
    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")
    assert Mt % per_core_M == 0, "per_core_M must divide Mt"
    assert Nt % per_core_N == 0, "per_core_N must divide Nt"
    assert Kt % K_block_size == 0, "K_block_size must divide Kt"
    num_blocks_y = Mt // per_core_M
    num_blocks_x = Nt // per_core_N
    assert (
        num_blocks_x <= num_cores_x and num_blocks_y <= num_cores_y
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
        core_x, core_y = ttl.core(dims=2)
        out_row = per_core_M * core_y
        out_col = per_core_N * core_x
        if (out_row < Mt) and (out_col < Nt):
            with out_cb.reserve() as out_blk:  # per_core_M * per_core_N
                for _ in range(Kt // K_block_size):
                    with (
                        a_cb.wait() as a_blk,
                        b_cb.wait() as b_blk,
                    ):  # a per_core_M x K_block_size, b K_block_size x per_core_N
                        out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        core_x, core_y = ttl.core(dims=2)
        out_row = per_core_M * core_y
        out_col = per_core_N * core_x
        if (out_row < Mt) and (out_col < Nt):
            for block in range(Kt // K_block_size):
                k = block * K_block_size
                with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                    a_wr = copy(
                        a[out_row : (out_row + per_core_M), k : (k + K_block_size)],
                        a_blk,
                    )
                    b_wr = copy(
                        b[k : (k + K_block_size), out_col : (out_col + per_core_N)],
                        b_blk,
                    )
                    a_wr.wait()
                    b_wr.wait()

    # blocking only occurs on the k dim, so each core writes its entire output block at once
    @ttl.datamovement()
    def mm_writer():
        core_x, core_y = ttl.core(dims=2)
        out_row = per_core_M * core_y
        out_col = per_core_N * core_x
        if (out_row < Mt) and (out_col < Nt):
            with out_cb.wait() as out_blk:
                out_wr = copy(
                    out_blk,
                    out[
                        out_row : (out_row + per_core_M),
                        out_col : (out_col + per_core_N),
                    ],
                )
                out_wr.wait()


@pytest.mark.parametrize("M,K,N", [(640, 640, 640)])
def test_multicore_reuse_matmul_tt_lang(M, K, N):
    """Test multicore matmul kernel."""
    device = ttnn.open_device(device_id=0)
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_multicore_reuse_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)

    ttnn.close_device(device)


@ttl.kernel(grid=(13, 10))
def tt_lang_multicore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    num_cores_x, num_cores_y = ttl.grid_size(dims=2)
    # this simplified non-reuse multicore matmul is limited to 1 tile per core, to highlight differences with the reuse version
    assert num_cores_x >= Nt
    assert num_cores_y >= Mt

    buffering_factor = 2
    a_cb = make_circular_buffer_like(a, shape=(1, 1), buffer_factor=buffering_factor)
    b_cb = make_circular_buffer_like(b, shape=(1, 1), buffer_factor=buffering_factor)
    # non buffered output, matching metal implementation
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def mm_compute():
        core_x, core_y = ttl.core(dims=2)
        out_row = core_y
        out_col = core_x
        if (out_row < Mt) and (out_col < Nt):
            with out_cb.reserve() as out_blk:
                for _ in range(Kt):
                    with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                        out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        core_x, core_y = ttl.core(dims=2)
        out_row = core_y
        out_col = core_x
        if (out_row < Mt) and (out_col < Nt):
            for k in range(Kt):
                with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                    a_wr = copy(a[out_row, k], a_blk)
                    b_wr = copy(b[k, out_col], b_blk)
                    a_wr.wait()
                    b_wr.wait()

    # blocking only occurs on the k dim, so each core writes its entire output block at once
    @ttl.datamovement()
    def mm_writer():
        core_x, core_y = ttl.core(dims=2)
        out_row = core_y
        out_col = core_x
        if (out_row < Mt) and (out_col < Nt):
            with out_cb.wait() as out_blk:
                out_wr = copy(
                    out_blk,
                    out[out_row, out_col],
                )
                out_wr.wait()
