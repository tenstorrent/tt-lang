# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 2: Single Node, Multi-Tile Block
# ===============================================
# Builds on Step 1 by processing multiple tiles per dataflow buffer entry
# instead of one tile at a time.
#
# New concepts introduced:
#   - Multi-tile blocks: each DFB entry holds a granularity-sized patch of
#     tiles.  Fewer, larger memory transfers reduce per-transfer overhead and
#     give the compute kernel more work per synchronization round-trip.
#   - Asymmetric block shapes: a, b, and c have different tile dimensions
#     (M×K, K×N, and M×N respectively), so their DFBs use matching shapes.
#
# Everything else (single node, same three-kernel structure) is identical to
# Step 1.  The loop bodies are unchanged; only the DFB shapes and the tensor
# slice ranges differ.

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

# M_GRANULARITY, N_GRANULARITY, K_GRANULARITY control how many tiles fit along
# each matmul dimension per block.  With all set to 4, each a-block is a 4×4
# patch of tiles (128×128 elements), each b-block is 4×4, and each c/y-block
# is 4×4 in M×N space.

M_GRANULARITY = 4
N_GRANULARITY = 4
K_GRANULARITY = 4


@ttl.operation(grid=(1, 1))
def __tutorial_operation(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    m_tiles_per_block = M_GRANULARITY
    n_tiles_per_block = N_GRANULARITY
    k_tiles_per_block = K_GRANULARITY

    # m_blocks, n_blocks, k_blocks now count blocks, not individual tiles.

    m_blocks = a.shape[0] // TILE_SIZE // m_tiles_per_block
    n_blocks = b.shape[1] // TILE_SIZE // n_tiles_per_block
    k_blocks = a.shape[1] // TILE_SIZE // k_tiles_per_block

    # DFB shapes match the tile dimensions of each tensor operand:
    #   a: M×K → shape (m_tiles_per_block, k_tiles_per_block)
    #   b: K×N → shape (k_tiles_per_block, n_tiles_per_block)
    #   c, acc, y: M×N → shape (m_tiles_per_block, n_tiles_per_block)

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
        for m_block in range(m_blocks):

            # Convert block index to tile index range for the tensor slice.

            start_m_tile = m_block * m_tiles_per_block
            end_m_tile = (m_block + 1) * m_tiles_per_block

            for n_block in range(n_blocks):
                start_n_tile = n_block * n_tiles_per_block
                end_n_tile = (n_block + 1) * n_tiles_per_block

                # Slice with a range to copy the entire M×N block in one transfer.

                with c_dfb.reserve() as c_blk:
                    tx_c = ttl.copy(
                        c[
                            start_m_tile:end_m_tile,
                            start_n_tile:end_n_tile,
                        ],
                        c_blk,
                    )

                    tx_c.wait()

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

    # The compute kernel is unchanged in structure from Step 1.  The hardware
    # now operates on full multi-tile blocks per iteration rather than single
    # tiles, amortizing synchronization overhead over more compute work.

    @ttl.compute()
    def compute():
        for _ in range(m_blocks):
            for _ in range(n_blocks):
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
        for m_block in range(m_blocks):
            start_m_tile = m_block * m_tiles_per_block
            end_m_tile = (m_block + 1) * m_tiles_per_block

            for n_block in range(n_blocks):
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
