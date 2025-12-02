# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING
import torch
import math

from sim import (
    TILE_SHAPE,
    TensorAccessor,
    IndexType,
    CoreIndex,
    CircularBuffer,
    dma,
    Program,
    core_index,
    is_tiled,
    ttl,
)

if TYPE_CHECKING:
    from sim.pykernel_env import granularity


@ttl.kernel(
    grid="auto",  # NOTE: allow compiler to choose grid
    granularity=2,  # compute granularity. could be passed by user, or left for auto-tuning
)
def eltwise_add(
    a_in: torch.Tensor,
    b_in: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # Assuming lightweight op input validation should be here
    assert a_in.shape == b_in.shape == out.shape
    assert all(is_tiled(tensor, TILE_SHAPE) for tensor in [a_in, b_in, out])
    assert a_in.shape[0] % granularity == 0

    row_tiles = a_in.shape[0] // TILE_SHAPE[0]
    col_tiles = a_in.shape[1] // TILE_SHAPE[1]

    # Parallelizing by columns here to get reuse on C
    grid_h, grid_w = ttl.grid_size()
    cols_per_core = math.ceil(col_tiles / (grid_h * grid_w))
    buffer_factor = 2

    a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)
    b_accessor = TensorAccessor(b_in, index_type=IndexType.TILE)
    out_accessor = TensorAccessor(out, index_type=IndexType.TILE)

    # Create circular buffers
    a_in_cb = CircularBuffer[type(a_in)](
        shape=(granularity, 1), buffer_factor=buffer_factor
    )
    b_in_cb = CircularBuffer[type(b_in)](
        shape=(granularity, 1), buffer_factor=buffer_factor
    )
    out_cb = CircularBuffer[type(out)](
        shape=(granularity, 1), buffer_factor=buffer_factor
    )

    @ttl.compute()
    def compute_func():
        core_num: CoreIndex = core_index()  # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
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
                result = a_block + b_block
                out_block.store(result)

                # finalize push, this advances the cb pointers, the writing happened at the line above
                out_cb.push()
                # finalize pop, this advances the cb pointers, essentially freeing the memory
                # After poping, the corresponding Block(a_block) points to stale data. Should probably make it an error to access it at that point
                a_in_cb.pop()
                # ditto
                b_in_cb.pop()

    @ttl.datamovement()
    def dm0():
        core_num: CoreIndex = core_index()  # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print("dm0: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
                col_slice = slice(ct, ct + 1)
                # Write the cbs just as above
                a_block = a_in_cb.reserve()
                tx = dma(a_accessor[row_slice, col_slice], a_block)
                tx.wait()
                a_in_cb.push()
                b_block = b_in_cb.reserve()
                tx = dma(b_accessor[row_slice, col_slice], b_block)
                tx.wait()
                b_in_cb.push()

    @ttl.datamovement()
    def dm1():
        core_num: CoreIndex = core_index()  # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print("dm1: ", f"core={core_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
                col_slice = slice(ct, ct + 1)

                out_block = out_cb.wait()
                # out_block[100] # accessing out of bounds should fail

                tx = dma(out_block, out_accessor[row_slice, col_slice])
                tx.wait()
                out_cb.pop()
                # TODO: We might want better error messages, most of them come from the underlying CBAPI
                #       which might be confusing to the higher level CircularBuffer user.
                # TODO: What if another thread writes to the same positions this Block points to?
                # out_block[0] # using pointer on stale data should fail
                # out_cb.pop() # double pop should fail

    # Execute the program across all cores
    return Program(compute_func, dm0, dm1)(a_in, b_in, out)
