# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from utils import assert_with_ulp
import ttl
from ttl import Program, copy, core, make_circular_buffer_like, Pipe, PipeNet

"""
will be multicasting a block from input_t to multiple cores, with each core writing to its own block in output_t
"""
@ttl.kernel(grid="auto")
def block_broadcast_multicast(input_t: ttnn.Tensor, output_t: ttnn.Tensor, block_h: int, block_w: int):
    assert input_t.shape[1]//ttnn.TILE_SIZE == block_h, "input tensor must be 1 block high."
    assert input_t.shape[0]//ttnn.TILE_SIZE == block_w, "input tensor must be 1 block wide."
    assert output_t.shape[1] % ttnn.TILE_SIZE == 0, "Output tensor height must be multiple of TILE_SIZE"
    assert output_t.shape[0] % ttnn.TILE_SIZE == 0, "Output tensor width must be multiple of TILE_SIZE"
    Ht = output_t.shape[1] // ttnn.TILE_SIZE
    Wt = output_t.shape[0] // ttnn.TILE_SIZE
    assert Ht % block_h == 0, "block_h must divide output height"
    assert Wt % block_w == 0, "block_w must divide output width"

    in_cb = make_circular_buffer_like(
        input_t, shape=(block_h, block_w)
    )
    out_cb = make_circular_buffer_like(
        output_t, shape=(block_h, block_w)
    )

    num_cores= ttl.grid_size(dims=1)
    num_blocks = (Ht // block_h) * (Wt // block_w)
    assert num_blocks <= num_cores, "Not enough cores"

    mcast_pipe = ttl.Pipe((0), (slice(1, num_blocks-1)))
    print(str(num_blocks) + " "+ str(mcast_pipe))
    net = PipeNet([mcast_pipe])


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
        with in_cb.reserve() as in_blk:
            def pipe_src(pipe):
                print("core:", core, "is source")
                in_rd = copy(input_t[0, 0], in_blk)
                in_rd.wait()
                mcast_wr = copy(in_blk, pipe)
                mcast_wr.wait()

            def pipe_dst(pipe):
                print("core:", core, "is dst")
                mcast_rd = copy(pipe, in_blk)
                mcast_rd.wait()

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)
    
    @ttl.datamovement()
    def mm_writer():
        core = ttl.core(dims=1)
        if core < num_blocks:
            out_row = core // (Wt // block_w)
            out_col = core % (Wt // block_w)
            print("core:", core, "writing to block at row:", out_row, "col:", out_col)
            with out_cb.wait() as out_blk:
                out_wr = copy(out_blk, output_t[out_row, out_col])
                out_wr.wait()


def test_block_broadcast_mcast(H, W, block_h, block_w):

    input_t = ttnn.rand(
        (block_h*ttnn.TILE_SIZE, block_w*ttnn.TILE_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    output_t = ttnn.empty(
        (W, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    block_broadcast_multicast(input_t, output_t, block_h, block_w)

    num_blocks_y = H // (block_h * ttnn.TILE_SIZE)
    num_blocks_x = W // (block_w * ttnn.TILE_SIZE)
    golden_output = input_t.to_torch().repeat(num_blocks_y, num_blocks_x).contiguous()
    #torch.set_printoptions(threshold=100000)
    print(output_t.shape)
    print(golden_output.shape)
    print(input_t.to_torch())
    print(output_t.to_torch())
    assert_with_ulp(output_t.to_torch(), golden_output, allow_nonfinite=True)

# test_block_broadcast_mcast(32, 64, 1, 1)
'''
get RuntimeError: ValueError: Tensor shape (32, 0) (=(1, 0) tiles) does not match Block shape (1, 1) tiles (=(32, 32) elements) for
  --> examples/block_broadcast_mcast.py:77:21
   |
77 |                 out_wr = copy(out_blk, output_t[out_row, out_col])
'''
test_block_broadcast_mcast(64, 64, 1, 1)