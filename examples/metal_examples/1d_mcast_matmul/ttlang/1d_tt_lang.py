# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from metal_examples.utils import assert_with_ulp
from ttl import Program, copy, core, make_circular_buffer_like, Pipe, PipeNet

"""
will be multicasting a block from input_t to multiple cores, with each core writing to its own block in output_t
"""


@ttl.kernel(grid=auto)
def matmul_1d(
    a_tensor: ttnn.Tensor,
    b_tensor: ttnn.Tensor,
    output_t: ttnn.Tensor,
    block_h: int,
    block_w: int,
    block_inner_dim: int,
    blocks_per_core_n: int,
):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    assert b.shape[1] == out.shape[1], "Output matrix has incorrect number of columns."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    # tiling checks
    assert ttl.grid_size(dims=1) >= Nt // (
        blocks_per_core_n * block_w
    ), "Not enough cores for the given tiling configuration."

    num_working_cores = Nt // (blocks_per_core_n * block_w)
    num_blocks_m = Mt // block_h
    num_blocks_k = Kt // block_inner_dim

    buffering_factor = 2
    a_cb = make_circular_buffer_like(
        a, shape=(block_h, block_inner_dim), buffer_factor=buffering_factor
    )
    b_cb = make_circular_buffer_like(
        b, shape=(block_inner_dim, block_w), buffer_factor=buffering_factor
    )
    # non buffered output, matching metal implementation
    out_cb = make_circular_buffer_like(out, shape=(block_h, block_w), buffer_factor=1)

    mcast_pipe = ttl.Pipe((0), slice(1, num_working_cores))
    net = PipeNet(mcast_pipe)

    def block_slice(block_offset, block_size):
        return slice(block_offset * block_size, (block_offset + 1) * block_size)

    @ttl.compute()
    def mm_compute():
        core = ttl.core(dims=1)
        if core >= num_working_cores:
            return
        for block_m in range(num_blocks_m):
            for block_n in range(blocks_per_core_n):
                with out_cb.reserve() as out_blk:
                    for block_k in range(num_blocks_k):
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                            out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        core = ttl.core(dims=1)
        if core >= num_working_cores:
            return
        for block_m in range(num_blocks_m):
            for _ in range(blocks_per_core_n):
                for block_k in range(num_blocks_k):
                    with a_cb.reserve() as a_blk:

                        def pipe_src(pipe):
                            in_rd = copy(
                                a_tensor[
                                    block_slice(block_m, block_h),
                                    block_slice(block_k, block_inner_dim),
                                ],
                                a_blk,
                            )
                            in_rd.wait()
                            mcast_wr = copy(a_blk, pipe)
                            mcast_wr.wait()

                        def pipe_dst(pipe):
                            mcast_rd = copy(pipe, a_blk)
                            mcast_rd.wait()

                        net.if_src(pipe_src)
                        net.if_dst(pipe_dst)

    @ttl.datamovement()
    def mm_writer():
        core = ttl.core(dims=1)
        if core >= num_working_cores:
            return
        for block_m in range(num_blocks_m):
            for block_n in range(blocks_per_core_n):
                with out_cb.wait() as out_blk:
                    for block_k in range(num_blocks_k):
                        with b_cb.reserve() as b_blk:
                            b_rd = copy(
                                b_tensor[
                                    block_slice(block_k, block_inner_dim),
                                    block_slice(
                                        block_n + core * blocks_per_core_n, block_w
                                    ),
                                ],
                                b_blk,
                            )
                            b_rd.wait()
                    out_wr = copy(
                        out_blk,
                        out[
                            block_slice(block_m, block_h),
                            block_slice(block_n + core * blocks_per_core_n, block_w),
                        ],
                    )
                    out_wr.wait()


@pytest.mark.parametrize(
    "M,N,K,block_h,block_w,block_inner_dim,blocks_per_core_n",
    [
        # N dim is written out as # of cores * TS * blocks_per_core * block_n
        (TS, 2 * TS, TS, 1, 1, 1, 1, 1, 1),  # trivial base case
        (TS, 14 * TS, TS, 1, 1, 1, 1, 1, 1),  # just over 1 row for all arch
        (TS, 8 * TS, TS * 2, 1, 1, 1, 1, 1, 1),  # 2 blocks in k dim
        (TS * 2, 8 * TS, TS, 1, 1, 1, 1, 1, 1),  # 2 blocks in m dim
        (TS, 8 * TS * 2, TS, 2, 1, 1, 1, 1, 1),  # 2 blocks per core in n dim
        pytest.param(
            TS * 6,
            2 * TS,
            TS * 2,
            1,
            2,
            1,
            1,
            2,
            1,
            marks=pytest.mark.skip(
                reason="Having M iterate >1 causes problems when also doing k dim accumulation, needs investigation"
            ),
        ),
        (
            TS,
            8 * TS * 2,
            TS * 2,
            2,
            1,
            1,
            1,
            1,
            1,
        ),  # 2 blocks per core in n dim, with 2 blocks in k dim
        (
            TS * 16,
            8 * TS,
            TS * 8,
            1,
            16,
            1,
            8,
            8,
            1,
        ),  # bigger blocks in m and k dims, with 2 subblocks per block in m/h dim
        (
            TS,
            8 * TS * 16,
            TS * 8,
            1,
            1,
            16,
            8,
            1,
            8,
        ),  # bigger blocks in n and k dims, with 2 subblocks per block in n/w dim
        # stress tests
        (
            TS * 4,
            8 * TS * 4,
            TS * 4 * 2,
            1,
            4,
            4,
            2,
            2,
            2,
        ),  # 4 tile blocks, with 2 subblocks in each dim
        (
            TS * 4,
            8 * TS * 2 * 4,
            TS * 4 * 2,
            2,
            4,
            4,
            2,
            2,
            2,
        ),  # above but with 2 blocks per core in n dim
        (
            TS * 4,
            64 * TS * 2 * 4,
            TS * 4 * 2,
            2,
            4,
            4,
            2,
            2,
            2,
        ),  # above but all cores wh
        (
            TS * 8,
            120 * TS * 2 * 8,
            TS * 16,
            2,
            8,
            8,
            16,
            4,
            2,
        ),  # all cores small bh 640/768 L1 tile limit
        pytest.param(
            TS * 8 * 2,
            120 * TS * 2 * 8,
            TS * 16 * 2,
            2,
            8,
            8,
            16,
            4,
            2,
            marks=pytest.mark.skip(
                reason="Having M iterate >1 causes problems when also doing k dim accumulation, needs investigation"
            ),
        ),  # above, but with 2 blocks in m and k dim
    ],
)
def test_matmul_1d(M, N, K, block_h, block_w, block_inner_dim, blocks_per_core_n):
    device = ttnn.open_device(device_id=0)

    A = ttnn.rand(
        (M, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    B = ttnn.rand(
        (K, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output_t = ttnn.empty(
        (M, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    matmul_1d(A, B, output_t, block_h, block_w, block_inner_dim, blocks_per_core_n)

    golden_output = A.to_torch() @ B.to_torch()

    assert_with_ulp(output_t.to_torch(), golden_output)
