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
# Grid:
#
# column
# x == 0
#   |
#   V
# (0, 0) (1, 0) (2, 0) (3, 0) <-- row y == 0
# (0, 1) (1, 1) (2, 1) (3, 1)
# (0, 2) (1, 2) (2, 2) (3, 2)
# (0, 3) (1, 3) (2, 3) (3, 3)

# ---------------------
# gather from row y to (0, y) with unicast.
#
# The pipe net is sized from the active set, not the launch extent.
# ROWS and COLS describe the rectangle that bounds the active set.
# Nodes outside the active rectangle (row 0..ROWS-1, column 0..COLS-1)
# skip the operation body.

ROWS = ...  # rows participating in the gather
COLS = ...  # columns participating in the gather

net = ttl.PipeNet(
    [ttl.Pipe(src=(x, y), dst=(0, y)) for x in range(1, COLS) for y in range(ROWS)]
)

# (1, 0) -> (0, 0) |             |
# (2, 0) -> (0, 0) | sequential  |
# (3, 0) -> (0, 0) |             |
# ...              |             | concurrent
#                                |
# (1, 1) -> (0, 1)               |
# ...                            |


@ttl.datamovement()
def dm():
    with dfb.reserve() as blk:

        def pipe_src(pipe):

            # write data into blk
            # ...

            # then copy blk to pipe:

            xf = ttl.copy(blk, pipe)
            xf.wait()

        def pipe_dst(pipe):

            # copy blk from pipe:

            xf = ttl.copy(pipe, blk)
            xf.wait()

            # then read data from blk
            # ...

        net.if_src(pipe_src)
        net.if_dst(pipe_dst)


# spec:end
