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
GRID_X, GRID_Y = 2, 4


@ttl.operation(grid=(GRID_X, GRID_Y))
def scatter(src: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # DFB shared by the node's kernels; one tile per node.
    dfb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), block_count=2)
    # spec:begin
    # ---------------------
    # scatter from (x, 0) to column x with multicast

    grid_x, grid_y = ttl.grid_size()

    net = ttl.PipeNet(
        [ttl.Pipe(src=(x, 0), dst=(x, slice(1, grid_y))) for x in range(grid_x)]
    )

    # (0, 0) => (0, 1) (0, 2) (0, 3) ... |
    # (1, 0) => (1, 1) (1, 2) (1, 3) ... | concurrent
    # ...                                |

    @ttl.datamovement()
    def dm():
        with dfb.reserve() as blk:

            def pipe_src(pipe):

                # write data into blk
                # ...
                # spec:end
                # Source node (nx, 0) loads its row's payload tile.
                nx, _ = ttl.node(dims=2)
                ttl.copy(src[nx : nx + 1, 0:1], blk).wait()
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
                # Destination node (nx, ny) stores the tile it received.
                nx, ny = ttl.node(dims=2)
                ttl.copy(blk, out[nx : nx + 1, ny : ny + 1]).wait()
                # spec:begin

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)

    # spec:end

    @ttl.compute()
    def _noop_compute() -> None:
        # The scatter is pure data movement; a no-op compute kernel plus a second
        # (empty) DM kernel satisfy the simulator's 3-kernel operation contract.
        pass

    @ttl.datamovement()
    def _noop_dm1() -> None:
        pass


torch.manual_seed(0)

device = ttnn.open_device(device_id=0)

try:
    # src tile (x, 0) holds row x's distinct payload; all other tiles are zero.
    TILE = 32
    src_torch = torch.zeros(GRID_X * TILE, GRID_Y * TILE, dtype=torch.float32)
    for x in range(GRID_X):
        src_torch[x * TILE : (x + 1) * TILE, 0:TILE] = float(x + 1)

    src_t = ttnn.from_torch(src_torch, layout=ttnn.TILE_LAYOUT, device=device)
    out_t = ttnn.zeros(
        ttnn.Shape([GRID_X * TILE, GRID_Y * TILE]),
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    scatter(src_t, out_t)

    # Each source (x, 0) multicasts its tile to (x, 1..GRID_Y-1); column 0 is a
    # source and is never written by a destination, so it stays zero.
    result = ttnn.to_torch(out_t)
    expected = torch.zeros_like(src_torch)
    for x in range(GRID_X):
        for y in range(1, GRID_Y):
            expected[x * TILE : (x + 1) * TILE, y * TILE : (y + 1) * TILE] = float(
                x + 1
            )

    # The multicast copies tiles without computing on them, so the comparison is
    # exact: a tolerance could not tell a column-0 tile that stayed zero from one
    # written with something small.
    assert torch.equal(
        expected, result
    ), "scatter multicast did not match torch reference"

finally:
    ttnn.close_device(device)
