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
# DFBs, the hidden data read/write that fills in the "..." placeholders, the
# no-op kernels, device setup, and the correctness check -- exists so the file
# can run standalone; it is not copied into the specification. The marked lines
# are nested inside @ttl.operation and dedented on render, and hidden lines are
# fenced with spec:end/spec:begin, so these mechanics add nothing to the
# rendered text.

import torch

import ttl
import ttnn

# Concrete grid for a standalone run.
GRID_X, GRID_Y = 2, 4


@ttl.operation(grid=(GRID_X, GRID_Y))
def forward_neighbor(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # One tile per node for the sent and received payloads.
    dfb_to_send = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    dfb_received = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
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
                # spec:end
                # Source node (nx, ny) loads its own payload tile to forward.
                nx, ny = ttl.node(dims=2)
                ttl.copy(inp[nx : nx + 1, ny : ny + 1], blk_to_send).wait()
                # spec:begin

                # then copy blk_to_send to pipe:

                xf = ttl.copy(blk_to_send, pipe)
                xf.wait()

            def pipe_dst(pipe):

                # copy blk_received from pipe:

                xf = ttl.copy(pipe, blk_received)
                xf.wait()

                # then read data from blk_received
                # ...
                # spec:end
                # Destination node (nx, ny) stores the tile from its -1 neighbor.
                nx, ny = ttl.node(dims=2)
                ttl.copy(blk_received, out[nx : nx + 1, ny : ny + 1]).wait()
                # spec:begin

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        # Forwarding is pure data movement; a no-op compute kernel plus a second
        # (empty) DM kernel satisfy the simulator's 3-kernel operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


torch.manual_seed(0)

device = ttnn.open_device(device_id=0)

try:
    TILE = 32
    # inp tile (x, y) holds a distinct payload for node (x, y).
    inp_torch = torch.zeros(GRID_X * TILE, GRID_Y * TILE, dtype=torch.float32)
    for x in range(GRID_X):
        for y in range(GRID_Y):
            inp_torch[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = float(
                x * GRID_Y + y + 1
            )

    inp_t = ttnn.from_torch(inp_torch, layout=ttnn.TILE_LAYOUT, device=device)
    out_t = ttnn.zeros(
        ttnn.Shape([GRID_X * TILE, GRID_Y * TILE]),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    forward_neighbor(inp_t, out_t)

    # Node (x, y) forwards its tile to (x, (y+1) % GRID_Y), so node (x, y)
    # receives the tile of its -1 neighbor (x, (y-1) % GRID_Y).
    result = ttnn.to_torch(out_t)
    expected = torch.zeros_like(inp_torch)
    for x in range(GRID_X):
        for y in range(GRID_Y):
            src_y = (y - 1) % GRID_Y
            expected[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = float(
                x * GRID_Y + src_y + 1
            )

    # Forwarding copies tiles without computing on them, so the comparison is
    # exact: a tolerance would accept a tile that arrived from the wrong neighbor
    # whenever the two payloads are close, or one that never arrived at all.
    assert torch.equal(
        expected, result
    ), "forward-neighbor ring did not match torch reference"

finally:
    ttnn.close_device(device)
