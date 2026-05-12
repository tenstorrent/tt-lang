# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# INTENTIONAL ERROR: demonstrates writing into a buffer block while it is still a
# live copy *source* (ROR). After filling the reserved block, we start
# copy(block, tensor) but attempt store() before tx.wait() completes.
#
# TTLANG_HARDWARE_CI: skip-compiler
# type: ignore
import math

import ttl
import ttnn
from utils.correctness import assert_with_ulp


@ttl.operation(
    grid="auto",
)
def eltwise_add(
    a_in: ttnn.Tensor,
    b_in: ttnn.Tensor,
    out: ttnn.Tensor,
) -> None:

    granularity = 2

    assert a_in.shape == b_in.shape == out.shape
    assert a_in.shape[0] % granularity == 0

    row_tiles = a_in.shape[0] // ttl.TILE_SHAPE[0]
    col_tiles = a_in.shape[1] // ttl.TILE_SHAPE[1]

    grid_h, grid_w = ttl.grid_size()
    cols_per_node = math.ceil(col_tiles / (grid_h * grid_w))
    block_count = 2

    a_in_dfb = ttl.make_dataflow_buffer_like(
        a_in, shape=(granularity, 1), block_count=block_count
    )
    b_in_dfb = ttl.make_dataflow_buffer_like(
        b_in, shape=(granularity, 1), block_count=block_count
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(granularity, 1), block_count=block_count
    )

    @ttl.compute()
    def compute_func():
        node_num = ttl.node(dims=1)
        start_col_tile = node_num * cols_per_node
        end_col_tile = min(start_col_tile + cols_per_node, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print(
                    "Compute: ", f"node={node_num}", f"column={ct}", f"block={rt_block}"
                )
                a_block = a_in_dfb.wait()
                b_block = b_in_dfb.wait()
                out_block = out_dfb.reserve()

                result = a_block + b_block
                out_block.store(result)

                out_block.push()
                a_block.pop()
                b_block.pop()

    @ttl.datamovement()
    def dm0():
        node_num = ttl.node(dims=1)
        start_col_tile = node_num * cols_per_node
        end_col_tile = min(start_col_tile + cols_per_node, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print("dm0: ", f"node={node_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
                col_slice = slice(ct, ct + 1)
                a_block = a_in_dfb.reserve()
                tx_fill = ttl.copy(a_in[row_slice, col_slice], a_block)
                tx_fill.wait()
                tx_src = ttl.copy(a_block, out[row_slice, col_slice])
                # INTENTIONAL ERROR: write while block is still a copy source (ROR)
                a_block.store(a_block)
                tx_src.wait()
                a_block.push()
                b_block = b_in_dfb.reserve()
                tx_b = ttl.copy(b_in[row_slice, col_slice], b_block)
                tx_b.wait()
                b_block.push()

    @ttl.datamovement()
    def dm1():
        node_num = ttl.node(dims=1)
        start_col_tile = node_num * cols_per_node
        end_col_tile = min(start_col_tile + cols_per_node, col_tiles)

        for ct in range(start_col_tile, end_col_tile):
            for rt_block in range(row_tiles // granularity):
                print("dm1: ", f"node={node_num}", f"column={ct}", f"block={rt_block}")
                row_slice = slice(rt_block * granularity, (rt_block + 1) * granularity)
                col_slice = slice(ct, ct + 1)

                out_block = out_dfb.wait()

                tx = ttl.copy(out_block, out[row_slice, col_slice])

                tx.wait()
                out_block.pop()


def main() -> None:
    dim = 256
    a_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
    b_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
    out = ttnn.empty((dim, dim), dtype=ttnn.float32)

    eltwise_add(a_in, b_in, out)

    golden = a_in + b_in
    assert_with_ulp(ttnn.to_torch(golden), ttnn.to_torch(out))


if __name__ == "__main__":
    main()
