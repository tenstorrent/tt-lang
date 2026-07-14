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
# ---------------------
# forward to a +1 neighbor in a column x

grid_x, grid_y = ttl.grid_size()

net = ttl.PipeNet(
    [
        ttl.Pipe(src=(x, y), dst=(x, (y + 1) % grid_y))
        for x in range(grid_x)
        for y in range(grid_y)
    ]
)

# (0, 0) => (0, 1)  |
# (0, 1) => (0, 2)  |
# ...               |
# (0, 7)* => (0, 0) |
# ...               | concurrent
#                   |
# (1, 0) => (1, 1)  |
# ...               |
#
# * - assuming (8, 8) grid


@ttl.datamovement()
def dm():

    with (
        dfb_to_send.reserve() as blk_to_send,
        dfb_received.reserve() as blk_received,
    ):

        def pipe_src(pipe):

            # write data into blk_to_send
            # ...

            # then copy blk_to_send to pipe:

            xf = ttl.copy(blk_to_send, pipe)
            xf.wait()

        def pipe_dst(pipe):

            # copy blk_received from pipe:

            xf = ttl.copy(pipe, blk_received)
            xf.wait()

            # then read data from blk_received
            # ...

        net.if_src(pipe_src)
        net.if_dst(pipe_dst)


# spec:end
