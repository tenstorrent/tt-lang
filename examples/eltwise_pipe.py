# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore
from typing import TYPE_CHECKING
import torch
import math

from sim import ttl
from sim.typedefs import Pipe

if TYPE_CHECKING:
    from sim.pykernel_env import granularity


@ttl.kernel(
    grid="auto",  # NOTE: allow compiler to choose grid
    granularity=2,  # compute granularity. could be passed by user, or left for auto-tuning
)
def eltwise_pipe(
    a_in: torch.Tensor,
    b_in: torch.Tensor,
    c_in: torch.Tensor,
    out: torch.Tensor,
    mode=None,  # Optional execution mode
) -> None:
    # Assuming lightweight op input validation should be here
    assert a_in.shape == b_in.shape == out.shape
    assert all(ttl.is_tiled(tensor, ttl.TILE_SHAPE) for tensor in [a_in, b_in, out])
    assert a_in.shape[0] % granularity == 0

    # Check that c_in is 1x1 and expand it to ttl.TILE_SHAPE
    assert c_in.shape == (1, 1), f"c_in must be 1x1, got {c_in.shape}"
    c_expanded = c_in.expand(ttl.TILE_SHAPE[0], ttl.TILE_SHAPE[1])

    row_tiles = a_in.shape[0] // ttl.TILE_SHAPE[0]
    col_tiles = a_in.shape[1] // ttl.TILE_SHAPE[1]

    # Parallelizing by columns here to get reuse on C
    grid_h, grid_w = ttl.grid_size()
    cols_per_core = math.ceil(col_tiles / (grid_h * grid_w))
    buffer_factor = (
        2  # TODO: Should buffer factor be tunable by the user? Or tuned by kernel?
    )

    a_accessor = ttl.TensorAccessor(a_in, index_type=ttl.IndexType.TILE)
    b_accessor = ttl.TensorAccessor(b_in, index_type=ttl.IndexType.TILE)
    c_accessor = ttl.TensorAccessor(c_expanded, index_type=ttl.IndexType.TILE)
    out_accessor = ttl.TensorAccessor(out, index_type=ttl.IndexType.TILE)

    # Create circular buffers
    a_in_cb = ttl.make_circular_buffer_like(
        a_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    b_in_cb = ttl.make_circular_buffer_like(
        b_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    c_in_cb = ttl.make_circular_buffer_like(
        c_in, shape=(1, 1), buffer_factor=buffer_factor
    )
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(granularity, 1), buffer_factor=buffer_factor
    )

    # Create multicast address for C using 2D coordinates
    # Source: (0,0), Destinations: rectangular range from (0,1) to (0,3)
    # This expands to cores (0,1), (0,2), (0,3)
    pipe = ttl.Pipe((0, 0), ((0, 1), (0, 3)))

    @ttl.compute()
    def compute_func():
        if not ttl.core_in_pipe(pipe):
            return  # This core is not participating in C multicast
        core_num = ttl.core(dims=1)  # linear core index
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)

        c_block = c_in_cb.wait()  # blocking

        for ct in range(start_col_tile, end_col_tile):
            # Reuse C across rows of A, B
            # TODO: Perhaps consider making Block pointers that come from wait()/reserve() read/write only respectively?
            for rt_block in range(row_tiles // granularity):
                print(
                    "Compute: ", f"core={core_num}", f"column={ct}", f"block={rt_block}"
                )
                # again, these return Block pointers:
                a_block = a_in_cb.wait()  # blocking
                b_block = b_in_cb.wait()  # blocking
                # NOTE: Please consider making non-approx the default for eltwise unary, but leave the option for the user to specify approx=True
                out_block = out_cb.reserve()  # blocking

                # Use store() to properly populate the Block with computed results
                result = a_block * b_block + c_block
                out_block.store(result)

                # finalize push, this advances the cb pointers, the writing happened at the line above
                out_cb.push()
                # finalize pop, this advances the cb pointers, essentially freeing the memory
                # After poping, the corresponding Block(a_block) points to stale data. Should probably make it an error to access it at that point
                a_in_cb.pop()
                # ditto
                b_in_cb.pop()
        c_in_cb.pop()

    @ttl.datamovement()
    def dm0():
        if not ttl.core_in_pipe(pipe):
            return  # This core is not participating in C multicast

        def pipe_src(p: Pipe) -> None:
            print("dm0 (C multicast SRC): ", f"core={core_num}")
            # C is only 1 tile
            c_block = c_in_cb.reserve()
            tx = ttl.copy(c_accessor[slice(0, 1), slice(0, 1)], c_block)
            tx.wait()
            tx2 = ttl.copy(
                c_block, p
            )  # start sending the data to all cores in the mcast address. Non-blocking
            tx2.wait()  # wait for all cores to do their corresponding copy receive
            # NoC layer meaning: receive ACKs from all cores in the mcast(?)
            c_in_cb.push()

        def pipe_dst(p: Pipe) -> None:
            print("dm0 (C multicast DST): ", f"core={core_num}")
            c_block = c_in_cb.reserve()
            tx = ttl.copy(
                p, c_block
            )  # start receiving data from the mcast address and store them in c_block. Non-blocking
            # NoC layer meaning: wait for packets with that mcast address
            tx.wait()  # Wait until all data from the mcast address is in c_block and sender is informed
            # NoC layer meaning: all data received and ACK is sent back to sender core, assuming reliable delivery
            c_in_cb.push()

        core_num = ttl.core(dims=1)  # linear core index
        ttl.if_pipe_src(pipe, pipe_src)
        ttl.if_pipe_dst(pipe, pipe_dst)

        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print("dm0: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
                col_slice = slice(ct, ct + 1)
                # Write the cbs just as above
                a_block = a_in_cb.reserve()
                tx = ttl.copy(a_accessor[row_slice, col_slice], a_block)
                tx.wait()
                a_in_cb.push()
                b_block = b_in_cb.reserve()
                tx = ttl.copy(b_accessor[row_slice, col_slice], b_block)
                tx.wait()
                b_in_cb.push()

    @ttl.datamovement()
    def dm1():
        if not ttl.core_in_pipe(pipe):
            return  # This core is not participating in C multicast
        core_num = ttl.core(dims=1)  # linear core index
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print("dm1: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
                col_slice = slice(ct, ct + 1)

                out_block = out_cb.wait()

                tx = ttl.copy(out_block, out_accessor[row_slice, col_slice])
                tx.wait()
                out_cb.pop()

    # Execute the program across all cores
    if mode is not None:
        ttl.Program(compute_func, dm0, dm1, execution_mode=mode)(a_in, b_in, out)
    else:
        ttl.Program(compute_func, dm0, dm1)(a_in, b_in, out)
