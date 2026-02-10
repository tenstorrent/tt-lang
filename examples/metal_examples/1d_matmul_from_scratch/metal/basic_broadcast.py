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
def block_broadcast_multicast(
    input_t: ttnn.Tensor, output_t: ttnn.Tensor, block_h: int, block_w: int
):
    assert input_t.shape[1] == block_h, "input tensor must be 1 block high."
    assert input_t.shape[0] == block_w, "input tensor must be 1 block wide."
    assert (
        output_t.shape[1] % ttnn.TILE_SIZE == 0
    ), "Output tensor height must be multiple of TILE_SIZE"
    assert (
        output_t.shape[0] % ttnn.TILE_SIZE == 0
    ), "Output tensor width must be multiple of TILE_SIZE"
    Ht = output_t.shape[1] // ttnn.TILE_SIZE
    Wt = output_t.shape[0] // ttnn.TILE_SIZE
    assert Ht % block_h == 0, "block_h must divide output height"
    assert Wt % block_w == 0, "block_w must divide output width"

    in_cb = make_circular_buffer_like(input_t, shape=(block_h, block_w))
    out_cb = make_circular_buffer_like(output_t, shape=(block_h, block_w))

    num_cores = ttl.grid_size(dims=1)
    num_blocks = (Ht // block_h) * (Wt // block_w)
    assert num_blocks <= num_cores, "Not enough cores"

    reciever_cores = (
        (0, slice(1, (Ht // block_h))),
        (slice(1, (Wt // block_w)), slice(0, (Ht // block_h))),
    )

    mcast_pipe = ttl.Pipe((0), slice(1, num_blocks))
    net = PipeNet(mcast_pipe)

    def block_slice(block_offset, block_size):
        return slice(block_offset * block_size, (block_offset + 1) * block_size)

    @ttl.compute()
    def mm_compute():
        core = ttl.core(dims=1)
        if core < num_blocks:
            with in_cb.wait() as in_blk, out_cb.reserve() as out_blk:
                out_blk.store(in_blk)

    @ttl.datamovement()
    def mm_reader():
        core = ttl.core(dims=1)
        if core >= num_blocks:
            return
        with in_cb.reserve() as in_blk:

            def pipe_src(pipe):
                in_rd = copy(
                    input_t[block_slice(0, block_h), block_slice(0, block_w)],
                    in_blk[block_slice(0, block_h), block_slice(0, block_w)],
                )
                mcast_wr = copy(
                    in_blk[block_slice(0, block_h), block_slice(0, block_w)], pipe
                )
                in_rd.wait()
                mcast_wr.wait()

            def pipe_dst(pipe):
                mcast_rd = copy(
                    pipe, in_blk[block_slice(0, block_h), block_slice(0, block_w)]
                )
                mcast_rd.wait()

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)

    @ttl.datamovement()
    def mm_writer():
        core = ttl.core(dims=1)
        if core < num_blocks:
            out_row = core // (Nt // block_w)
            out_col = core % (Nt // block_w)
            with out_cb.wait() as out_blk:
                out_wr = copy(
                    out_blk,
                    output_t[
                        block_slice(out_row, block_h), block_slice(out_col, block_w)
                    ],
                )


@pytest.mark.parametrize(
    "H,W,block_h,block_w",
    [
        (128, 128, 1, 1),
    ],
)
def test_block_broadcast_mcast(H, W, block_h, block_w):
    device = ttnn.open_device(device_id=0)

    input_t = ttnn.rand(
        (block_h, block_w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output_t = ttnn.empty(
        (W, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    block_broadcast_multicast(input_t, output_t, block_h, block_w)

    num_blocks_y = H // block_h
    num_blocks_x = W // block_w
    golden_output = input_t.to_torch().repeat(num_blocks_y, num_blocks_x).contiguous()

    assert_with_ulp(output_t.to_torch(), golden_output, max_ulp=1)
