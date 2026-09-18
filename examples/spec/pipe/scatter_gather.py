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
# Everything outside the markers -- imports, the @ttl.operation wrapper, the
# DFB, the hidden data read/write that fills in the "..." placeholders, the no-
# op kernels, device setup, and the correctness check -- exists so the file can
# run standalone; it is not copied into the specification. The marked lines are
# nested inside @ttl.operation and dedented on render, and hidden lines are
# fenced with spec:end/spec:begin, so these mechanics add nothing to the
# rendered text.

import torch

import ttl
import ttnn

# Concrete grid for a standalone run.
GRID_X, GRID_Y = 2, 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def scatter_gather(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # One tile per node, shared by the node's send and receive.
    dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    # spec:begin
    # ---------------------
    # scatter-gather column x with multicast/loopback

    grid_x, grid_y = ttl.grid_size()

    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(x, y), dst=(x, slice(0, grid_y)))
            for x in range(grid_x)
            for y in range(grid_y)
        ]
    )

    # (0, 0) => (0, 0) (0, 1) (0, 2) ... |            |
    # (0, 1) => (0, 0) (0, 1) (0, 2) ... | sequential |
    # (0, 2) => (0, 0) (0, 1) (0, 2) ... |            |
    # ...                                |            | concurrent
    #                                                 |
    # (1, 0) => (1, 0) (1, 1) (1, 2) ...              |
    # ...                                             |

    @ttl.datamovement()
    def dm():
        with dfb.reserve() as blk:

            def pipe_src(pipe):

                # write data into blk
                # ...
                # spec:end
                # Source node (nx, ny) loads its payload tile before broadcasting
                # it to the whole column (including itself).
                nx, ny = ttl.node(dims=2)
                ttl.copy(inp[nx : nx + 1, ny : ny + 1], blk).wait()
                # spec:begin

                # then copy blk to pipe:

                xf = ttl.copy(blk, pipe)
                xf.wait()

            def pipe_dst(pipe):

                # copy blk from pipe:

                xf = ttl.copy(pipe, blk)
                xf.wait()

                # then read data from blk
                # ...
                # spec:end
                # Destination node (nx, ny) stores each gathered tile. Every
                # source in the column carries that column's value, so the
                # result is well-defined regardless of arrival order.
                nx, ny = ttl.node(dims=2)
                ttl.copy(blk, out[nx : nx + 1, ny : ny + 1]).wait()
                # spec:begin

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        # Scatter-gather is pure data movement; a no-op compute kernel plus a
        # second (empty) DM kernel satisfy the 3-kernel operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


torch.manual_seed(0)

device = ttnn.open_device(device_id=0)

try:
    TILE = 32
    # Every node in column x carries the same payload (x + 1), so the gathered
    # result at each node in the column is well-defined (== x + 1) no matter the
    # order in which the column's broadcasts arrive.
    inp_torch = torch.zeros(GRID_X * TILE, GRID_Y * TILE, dtype=torch.float32)
    for x in range(GRID_X):
        inp_torch[x * TILE : (x + 1) * TILE, :] = float(x + 1)

    inp_t = ttnn.from_torch(inp_torch, layout=ttnn.TILE_LAYOUT, device=device)
    out_t = ttnn.zeros(
        ttnn.Shape([GRID_X * TILE, GRID_Y * TILE]),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    scatter_gather(inp_t, out_t)

    # Each node in column x gathers that column's value (x + 1) into its tile.
    result = ttnn.to_torch(out_t)
    expected = torch.zeros_like(inp_torch)
    for x in range(GRID_X):
        expected[x * TILE : (x + 1) * TILE, :] = float(x + 1)

    # The loopback copies tiles without computing on them, so the comparison is
    # exact: a tolerance would accept a node that gathered something slightly
    # wrong, or that gathered nothing where the column's value is small.
    #
    # It distinguishes the columns and not the nodes within one: every node in a
    # column sends the same value, which is what makes the gathered result
    # independent of arrival order, and so a source that read a column
    # neighbour's tile instead of its own would pass. Giving the nodes in a
    # column distinct payloads would pin that, and would leave the result
    # dependent on which broadcast arrived last.
    assert torch.equal(
        expected, result
    ), "scatter-gather column loopback did not match torch reference"

finally:
    ttnn.close_device(device)
