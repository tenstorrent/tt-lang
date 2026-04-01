# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 1: Single Node, Single-Tile Block
# ================================================
# Introduces the core TT-Lang programming model:
#   - @ttl.operation   — declares a kernel and the grid it runs on
#   - @ttl.compute  — the compute thread: arithmetic on tiles
#   - @ttl.datamovement — DM threads: move data between DRAM and L1
#   - ttl.make_dataflow_buffer_like — creates an in-L1 dataflow buffer (DFB)
#     that synchronizes data passing between threads
#   - ttl.copy / tx.wait — initiates and awaits a transfer
#
# The kernel fuses a * b + c into a single operation, processing one 32×32 tile
# at a time.  The outer TT-NN multiply by d is left to TT-NN (see Step 0).

import ttnn
import torch


def from_torch(tensor: torch.Tensor):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


import ttl

# Tenstorrent hardware operates on 32×32 tiles.  Tensor dimensions in tile
# coordinates are obtained by dividing the element-count by TILE_SIZE.

TILE_SIZE = 32


# @ttl.operation marks a Python function as a TT-Lang operation.
# grid=(1, 1) means the kernel runs on a single node (one Tensix core).
# The function signature lists the tensors the kernel reads and writes;
# these live in DRAM and are passed by the host at call time.


@ttl.operation(grid=(1, 1))
def __tutorial_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor):

    # Compute iteration counts in tile coordinates.

    rows = a.shape[0] // TILE_SIZE
    cols = a.shape[1] // TILE_SIZE

    # Dataflow buffers (DFBs) are L1 buffers shared between threads.
    # shape=(1, 1) means each entry holds exactly one 32×32 tile.
    # buffer_factor=2 allocates two entries, enabling double-buffering: while the
    # compute thread processes one entry, the DM thread can fill the other.

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), buffer_factor=2)

    # The compute thread runs concurrently with the DM threads.
    # It waits for filled input blocks and reserves empty output blocks.

    @ttl.compute()
    def tutorial_compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    # wait() blocks until the DM reader has pushed a filled tile.
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    c_dfb.wait() as c_blk,
                    # reserve() blocks until the DM writer has popped the previous
                    # output tile, freeing an entry for the next result.
                    y_dfb.reserve() as y_blk,
                ):
                    # Fused elementwise operation: a * b + c, written to y.
                    # The `with` block implicitly calls pop() on inputs and
                    # push() on the output when the scope exits.

                    y_blk.store(a_blk * b_blk + c_blk)

    # The first DM thread reads input tiles from DRAM into DFBs.

    @ttl.datamovement()
    def tutorial_read():
        for row in range(rows):
            for col in range(cols):
                with (
                    # reserve() blocks until the compute thread has freed an entry.
                    a_dfb.reserve() as a_blk,
                    b_dfb.reserve() as b_blk,
                    c_dfb.reserve() as c_blk,
                ):

                    # ttl.copy initiates a transfer from DRAM tensor slice to L1 block.
                    # a[row, col] selects the tile at position (row, col) in tile
                    # coordinates (not element coordinates).

                    tx_a = ttl.copy(
                        a[row, col],
                        a_blk,
                    )
                    tx_b = ttl.copy(
                        b[row, col],
                        b_blk,
                    )
                    tx_c = ttl.copy(
                        c[row, col],
                        c_blk,
                    )

                    # Wait for all three trasfers to finish before the `with` block
                    # exits and implicitly calls push(), signalling the compute
                    # thread that the tiles are ready.

                    tx_a.wait()
                    tx_b.wait()
                    tx_c.wait()

    # The second DM thread writes computed output tiles from L1 back to DRAM.

    @ttl.datamovement()
    def tutorial_write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    # Copy the computed tile from L1 to the output DRAM tensor.
                    tx = ttl.copy(
                        y_blk,
                        y[row, col],
                    )
                    tx.wait()


def tutorial_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor):
    y = from_torch(torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16))
    __tutorial_kernel(a, b, c, y)
    return y


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    shape = (2048, 2048)

    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand(shape, dtype=torch.bfloat16)
    d = torch.rand(shape, dtype=torch.bfloat16)

    expected_y = (a * b + c) * d

    a = from_torch(a)
    b = from_torch(b)
    c = from_torch(c)
    d = from_torch(d)

    y = ttnn.multiply(tutorial_kernel(a, b, c), d)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
