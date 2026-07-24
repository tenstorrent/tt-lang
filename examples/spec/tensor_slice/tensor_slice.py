# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, scaffolding, the @ttl.operation
# wrapper, device setup) exists so the file can run standalone; it is not
# copied into the specification. The marked lines are nested inside
# @ttl.operation and dedented on render, so the rendered spec is unchanged.

import math

import torch

import ttl
import ttnn

# Concrete compile-time sizes for a standalone run (scaffolding, not rendered).
A_ROWS, A_COLS = 64, 64


@ttl.operation(grid=(1, 2))
def tensor_slice(A: ttnn.Tensor) -> None:  # input matrix (A_ROWS, A_COLS)
    # The rendered spec spells the grid helper unqualified; bind it to the ttl
    # helper so the marked lines below stay verbatim.
    grid_size = ttl.grid_size
    # spec:begin
    g = 2  # granularity
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(g, 1))

    row_tiles = A.shape[0] // ttl.TILE_SHAPE[0]
    col_tiles = A.shape[1] // ttl.TILE_SHAPE[1]
    cols_per_node = math.ceil(col_tiles / (grid_size(dims=1)))

    node_num = ttl.node(dims=1)
    start_ct = node_num * cols_per_node
    end_ct = min(start_ct + cols_per_node, col_tiles)

    @ttl.datamovement()
    def dm():
        for ct in range(start_ct, end_ct):
            for rt in range(row_tiles // g):

                # acquire a_blk from a_dfb:

                with a_dfb.reserve() as a_blk:

                    # then copy from a tensor slice of matching shape:

                    row_slice = slice(rt * g, (rt + 1) * g)  # explicit row slice
                    a_xf = ttl.copy(
                        A[row_slice, ct : ct + 1], a_blk
                    )  # in-line col slice
                    a_xf.wait()

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        # This slice demonstration is pure data movement; a no-op compute kernel
        # plus a second (empty) DM kernel satisfy the simulator's 3-kernel
        # (compute + 2 DM) operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


device = ttnn.open_device(device_id=0)

try:
    A_t = ttnn.rand(
        ttnn.Shape([A_ROWS, A_COLS]), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Each node reads its assigned column tiles; the run exercises node-dependent
    # setup (ttl.node()/ttl.grid_size()) and tile-coordinate tensor slicing.
    tensor_slice(A_t)

finally:
    ttnn.close_device(device)
