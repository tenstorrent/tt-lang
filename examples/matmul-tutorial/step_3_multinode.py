# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 3: Multi-Node, Fixed Grid
# ========================================
# Extends Step 2 by running the operation across a grid of nodes in parallel.
#
# New concepts introduced:
#   - grid=(4, 4)           — run the operation on a 4×4 grid of nodes (16 cores)
#   - ttl.grid_size(dims=2) — query the (n, m) grid dimensions at runtime
#   - ttl.node(dims=2)      — query this node's (node_n, node_m) position
#
# Each node processes an independent rectangular region of the output tensor,
# partitioned along the M and N dimensions.  The K dimension is not partitioned:
# every node iterates over all k_blocks to accumulate its full partial product.
# This requires tensor dimensions to be evenly divisible by the grid (see Step
# 4 for a version that handles remainders).

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

TILE_SIZE = 32
M_GRANULARITY = 4
N_GRANULARITY = 4
K_GRANULARITY = 4


# grid=(4, 4) launches the operation body on every node of a 4-column × 4-row
# grid.  All nodes execute the same code; they differentiate their work via
# ttl.node().


@ttl.operation(grid=(4, 4))
def __tutorial_operation(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    m_tiles_per_block = M_GRANULARITY
    n_tiles_per_block = N_GRANULARITY
    k_tiles_per_block = K_GRANULARITY

    m_blocks = a.shape[0] // TILE_SIZE // m_tiles_per_block
    n_blocks = b.shape[1] // TILE_SIZE // n_tiles_per_block
    k_blocks = a.shape[1] // TILE_SIZE // k_tiles_per_block

    # ttl.grid_size returns (grid_n, grid_m) matching the (n, m) convention
    # used by ttl.node.  The grid is partitioned so each node handles an
    # independent slice of the M×N output space.

    grid_n, grid_m = ttl.grid_size(dims=2)

    # Divide the total block count evenly across the grid.
    # Assumes the tensor is evenly divisible by the grid size.

    m_blocks_per_node = m_blocks // grid_m
    n_blocks_per_node = n_blocks // grid_n

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(m_tiles_per_block, k_tiles_per_block), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(k_tiles_per_block, n_tiles_per_block), block_count=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(
        c, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2
    )
    acc_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(m_tiles_per_block, n_tiles_per_block), block_count=2
    )

    @ttl.datamovement()
    def read():

        # ttl.node() returns the zero-based coordinates of this specific node.
        # node_n and node_m are used to offset into the global tensor.

        node_n, node_m = ttl.node(dims=2)

        for local_m_block in range(m_blocks_per_node):

            # Map local block index to global block index.

            m_block = node_m * m_blocks_per_node + local_m_block
            start_m_tile = m_block * m_tiles_per_block
            end_m_tile = (m_block + 1) * m_tiles_per_block

            for local_n_block in range(n_blocks_per_node):
                n_block = node_n * n_blocks_per_node + local_n_block
                start_n_tile = n_block * n_tiles_per_block
                end_n_tile = (n_block + 1) * n_tiles_per_block

                with c_dfb.reserve() as c_blk:
                    tx_c = ttl.copy(
                        c[
                            start_m_tile:end_m_tile,
                            start_n_tile:end_n_tile,
                        ],
                        c_blk,
                    )

                    tx_c.wait()

                # All nodes iterate over the full k dimension to accumulate
                # their partial matmul result independently.

                for k_block in range(k_blocks):
                    start_k_tile = k_block * k_tiles_per_block
                    end_k_tile = (k_block + 1) * k_tiles_per_block
                    with (
                        a_dfb.reserve() as a_blk,
                        b_dfb.reserve() as b_blk,
                    ):
                        tx_a = ttl.copy(
                            a[
                                start_m_tile:end_m_tile,
                                start_k_tile:end_k_tile,
                            ],
                            a_blk,
                        )
                        tx_b = ttl.copy(
                            b[
                                start_k_tile:end_k_tile,
                                start_n_tile:end_n_tile,
                            ],
                            b_blk,
                        )

                        tx_a.wait()
                        tx_b.wait()

    # The compute kernel iterates over the blocks assigned to this node.
    # It does not need to know its node coordinates: the DM kernels already
    # stream only the relevant tiles into the DFBs.

    @ttl.compute()
    def compute():
        for _ in range(m_blocks_per_node):
            for _ in range(n_blocks_per_node):
                with acc_dfb.reserve() as acc_blk:
                    acc_blk.store(ttl.math.fill(acc_blk, 0))

                for _ in range(k_blocks):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        acc_dfb.wait() as pre_acc_blk,
                    ):
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(pre_acc_blk + a_blk @ b_blk)

                with c_dfb.wait() as c_blk, acc_dfb.wait() as acc_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(ttl.math.relu(c_blk + acc_blk))

    @ttl.datamovement()
    def write():
        node_n, node_m = ttl.node(dims=2)

        for local_m_block in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m_block
            start_m_tile = m_block * m_tiles_per_block
            end_m_tile = (m_block + 1) * m_tiles_per_block

            for local_n_block in range(n_blocks_per_node):
                n_block = node_n * n_blocks_per_node + local_n_block
                start_n_tile = n_block * n_tiles_per_block
                end_n_tile = (n_block + 1) * n_tiles_per_block

                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[
                            start_m_tile:end_m_tile,
                            start_n_tile:end_n_tile,
                        ],
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
