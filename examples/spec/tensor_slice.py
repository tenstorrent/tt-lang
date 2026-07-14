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
# Everything outside the markers (imports, scaffolding) exists so the file can
# stand on its own and is not copied into the specification.

import math

import torch

import ttl
import ttnn

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
                a_xf = ttl.copy(A[row_slice, ct : ct + 1], a_blk)  # in-line col slice
                a_xf.wait()


# spec:end
