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
# DFB, the concrete ROWS/COLS, the hidden data read/write that fills in the
# "..." placeholders, the no-op kernels, device setup, and the correctness
# check -- exists so the file can run standalone; it is not copied into the
# specification. The marked lines are nested inside @ttl.operation and dedented
# on render, and hidden lines are fenced with spec:end/spec:begin, so these
# mechanics add nothing to the rendered text.

import torch

import ttl
import ttnn

# Concrete active rectangle for a standalone run:
# COLS columns (x in [0, COLS)) x ROWS rows (y in [0, ROWS)).
GRID_COLS, GRID_ROWS = 3, 2


@ttl.operation(grid=(GRID_COLS, GRID_ROWS))
def gather(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # One tile per node, shared by the node's send and receive.
    dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
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
    # spec:end
    ROWS = GRID_ROWS
    COLS = GRID_COLS
    # spec:begin

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
                # spec:end
                # Source node (nx, ny) loads its payload tile before sending it
                # to the row's gather node (0, ny).
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
                # Gather node (0, ny) stores each tile it receives. Every source
                # in row ny carries that row's value, so the result is
                # well-defined regardless of arrival order.
                nx, ny = ttl.node(dims=2)
                ttl.copy(blk, out[nx : nx + 1, ny : ny + 1]).wait()
                # spec:begin

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        # The gather is pure data movement; a no-op compute kernel plus a second
        # (empty) DM kernel satisfy the 3-kernel operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


torch.manual_seed(0)

device = ttnn.open_device(device_id=0)

try:
    TILE = 32
    # Every source in row y carries the same payload (y + 1), so the gather node
    # (0, y) has a well-defined result regardless of arrival order.
    inp_torch = torch.zeros(GRID_COLS * TILE, GRID_ROWS * TILE, dtype=torch.float32)
    for x in range(1, GRID_COLS):
        for y in range(GRID_ROWS):
            inp_torch[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = float(
                y + 1
            )

    inp_t = ttnn.from_torch(inp_torch, layout=ttnn.TILE_LAYOUT, device=device)
    out_t = ttnn.zeros(
        ttnn.Shape([GRID_COLS * TILE, GRID_ROWS * TILE]),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    gather(inp_t, out_t)

    # Only the gather nodes (0, y) write output, receiving row y's value (y + 1);
    # every other node is a source and leaves its output tile zero.
    result = ttnn.to_torch(out_t)
    expected = torch.zeros_like(inp_torch)
    for y in range(GRID_ROWS):
        expected[0:TILE, y * TILE : (y + 1) * TILE] = float(y + 1)

    # The gather copies tiles without computing on them, so the comparison is
    # exact: a tolerance could not tell a source node's output tile that stayed
    # zero from one a mis-addressed unicast wrote something small into.
    assert torch.equal(expected, result), "gather did not match torch reference"

finally:
    ttnn.close_device(device)
