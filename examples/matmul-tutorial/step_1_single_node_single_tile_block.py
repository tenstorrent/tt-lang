# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 1: Single Node, Single-Tile Block
# ================================================
# Introduces the core TT-Lang programming model for matmul:
#   - @ttl.operation   — declares an operation and the grid it runs on
#   - @ttl.compute     — the compute kernel: tile-level matrix multiply and add
#   - @ttl.datamovement — DM kernels: move data between DRAM and L1
#   - ttl.make_dataflow_buffer_like — creates an in-L1 dataflow buffer (DFB)
#     that synchronizes data passing between kernels
#   - ttl.copy / tx.wait — initiates and awaits a transfer
#   - ttl.math.fill    — fills a block with a scalar value (used to zero the
#     accumulator before the k-reduction loop)
#
# The operation fuses a @ b + c followed by relu into a single kernel,
# processing one 32×32 tile at a time.  The outer m×n loop iterates over
# output tiles; the inner k loop accumulates partial products.

import ttnn
import torch


def from_torch(tensor: torch.Tensor):

    # Upload a bfloat16 torch tensor to DRAM on the device in tiled layout.

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
# grid=(1, 1) means the operation runs on a single node (one Tensix core).
# The function signature lists the tensors the operation reads and writes;
# these live in DRAM and are passed by the host at call time.


@ttl.operation(grid=(1, 1))
def __tutorial_operation(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:

    # Compute iteration counts in tile coordinates.

    m_tiles = a.shape[0] // TILE_SIZE
    n_tiles = b.shape[1] // TILE_SIZE
    k_tiles = a.shape[1] // TILE_SIZE

    # Dataflow buffers (DFBs) are L1 buffers shared between threads.
    # shape=(1, 1) means each entry holds exactly one 32×32 tile.
    # block_count=2 allocates two blocks, enabling double-buffering: while the
    # compute kernel processes one entry, the DM kernel can fill the other.
    #
    # acc_dfb is the running accumulator for the k-reduction.  It is both
    # produced and consumed by the compute kernel in a ping-pong pattern:
    # each k-step reads the previous partial sum (pre_acc_blk) and writes a
    # new one (acc_blk), so block_count=2 allows the two slots to alternate.

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    # The DM reader runs concurrently with the compute kernel.
    # For each output tile (m, n) it first reads the bias tile c[m, n], then
    # streams all k input tiles for a and b into their respective DFBs.

    @ttl.datamovement()
    def read():
        for m_tile in range(m_tiles):
            for n_tile in range(n_tiles):

                # Read the bias tile for this (m, n) output position first so
                # it is available when the compute kernel finishes accumulating.

                with c_dfb.reserve() as c_blk:
                    tx_c = ttl.copy(
                        c[m_tile, n_tile],
                        c_blk,
                    )

                    tx_c.wait()

                for k_tile in range(k_tiles):

                    # Stream a[m, k] and b[k, n] tiles into L1 for each step
                    # of the k-reduction.

                    with (
                        a_dfb.reserve() as a_blk,
                        b_dfb.reserve() as b_blk,
                    ):
                        tx_a = ttl.copy(
                            a[m_tile, k_tile],
                            a_blk,
                        )
                        tx_b = ttl.copy(
                            b[k_tile, n_tile],
                            b_blk,
                        )

                        tx_a.wait()
                        tx_b.wait()

    # The compute kernel accumulates partial matmul products across k, then
    # adds the bias and applies relu before writing the result to y_dfb.

    @ttl.compute()
    def compute():
        for _ in range(m_tiles):
            for _ in range(n_tiles):

                # Initialize the accumulator to zero before the k loop.
                # ttl.math.fill produces a block expression; store() materializes
                # it into acc_blk and pushes it so the k loop can consume it.

                with acc_dfb.reserve() as acc_blk:
                    acc_blk.store(ttl.math.fill(acc_blk, 0))

                for _ in range(k_tiles):

                    # Consume the previous partial sum (pre_acc_blk) along with
                    # the next a and b tiles, compute the updated partial sum,
                    # and push it back into acc_dfb for the next k-step.

                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        acc_dfb.wait() as pre_acc_blk,
                    ):
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(pre_acc_blk + a_blk @ b_blk)

                # After k is exhausted, add the bias and apply relu in one step.

                with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(ttl.math.relu(c_blk + acc_blk))

    # The DM writer reads completed output tiles from y_dfb and writes them
    # back to the output tensor in DRAM.

    @ttl.datamovement()
    def write():
        for m_tile in range(m_tiles):
            for n_tile in range(n_tiles):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[m_tile, n_tile],
                    )
                    tx.wait()


def tutorial_operation(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor):
    y = from_torch(torch.zeros((a.shape[0], b.shape[1]), dtype=torch.bfloat16))
    __tutorial_operation(a, b, c, y)
    return y


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    M, K, N = 8192, 8192, 8192

    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.randn((M, N), dtype=torch.bfloat16)

    expected_y = torch.relu(a @ b + c)

    a = from_torch(a)
    b = from_torch(b)
    c = from_torch(c)

    y = tutorial_operation(a, b, c)

    y = ttnn.to_torch(y)

    pcc = torch.corrcoef(
        torch.stack([y.flatten().float(), expected_y.flatten().float()])
    )[0, 1].item()

    print(f"PCC {pcc:.6f}")

    assert pcc > 0.99

finally:
    ttnn.close_device(device)
