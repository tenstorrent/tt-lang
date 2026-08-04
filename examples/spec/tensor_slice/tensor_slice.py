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
# @ttl.operation and dedented on render, so these mechanics add nothing to the
# rendered text.

import math

import torch

import ttl
import ttnn

# Concrete compile-time sizes for a standalone run. More than one group of g row
# tiles per column, so that the row slice the marked kernel computes has to
# advance: at 64 rows there is a single group and any row arithmetic reads the
# same tiles.
A_ROWS, A_COLS = 128, 64


@ttl.operation(grid=(1, 2))
def tensor_slice(
    A: ttnn.Tensor,  # input matrix (A_ROWS, A_COLS)
    out: ttnn.Tensor,  # scaffolding: receives what the marked kernel loaded
) -> None:
    # The marked lines below are the specification's, which calls grid_size
    # without the ttl prefix.  This alias is what makes them run.
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
    def _drain_to_out() -> None:
        # Writes each block the marked kernel loaded back out, in the order it
        # loaded them, so the run can be checked against torch. Nothing else
        # observes what was loaded: without this the slice arithmetic above could
        # read the wrong tiles, or the same tile every time, and the example would
        # still finish.
        for ct in range(start_ct, end_ct):
            for rt in range(row_tiles // g):
                with a_dfb.wait() as a_blk:
                    ttl.copy(a_blk, out[rt * g : (rt + 1) * g, ct : ct + 1]).wait()


device = ttnn.open_device(device_id=0)

try:
    A_t = ttnn.rand(
        ttnn.Shape([A_ROWS, A_COLS]), layout=ttnn.TILE_LAYOUT, device=device
    )
    out_t = ttnn.zeros(
        ttnn.Shape([A_ROWS, A_COLS]), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Each node reads its assigned column tiles; the run exercises node-dependent
    # setup (ttl.node()/ttl.grid_size()) and tile-coordinate tensor slicing.
    tensor_slice(A_t, out_t)

    # The two nodes' column ranges cover A between them, and the slices only move
    # tiles, so every element comes back bit for bit.
    assert torch.equal(
        ttnn.to_torch(A_t), ttnn.to_torch(out_t)
    ), "the sliced reads did not cover A exactly once"

finally:
    ttnn.close_device(device)
