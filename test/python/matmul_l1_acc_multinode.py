# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Multinode matmul with L1 packer accumulation. Auto grid, split DMA (reader=A,
writer=B+output), 8x8x8 blocks, K_num_blocks=4 at 3072x1024x3072.

The larger dimensions (96x32x96 tiles, 12x4x12 blocks) ensure each core
handles multiple output blocks (ceil(12/8)=2 per axis on an 8x8 grid),
exercising the per-block L1 acc disable/re-enable sequence.

The compute thread uses += for accumulation across K iterations. The
compiler inserts pack_reconfig_l1_acc guards so each K iteration packs
additively to L1.

Verifies the L1 packer accumulation pattern in generated C++: disable before
K loop, conditional enable after first iteration, disable after cb_push_back.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

TILE = 32
M_BLOCK = 8
K_BLOCK = 8
N_BLOCK = 8


@ttl.operation(grid="auto")
def matmul_l1_acc(a, b, out):
    Mt = a.shape[0] // TILE
    Kt = a.shape[1] // TILE
    Nt = b.shape[1] // TILE

    K_num_blocks = Kt // K_BLOCK
    M_num_blocks = Mt // M_BLOCK
    N_num_blocks = Nt // N_BLOCK

    grid_n, grid_m = ttl.grid_size(dims=2)
    m_blocks_per_node = -(-M_num_blocks // grid_m)
    n_blocks_per_node = -(-N_num_blocks // grid_n)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(M_BLOCK, K_BLOCK), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K_BLOCK, N_BLOCK), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(M_BLOCK, N_BLOCK), block_count=2
    )

    @ttl.compute()
    def compute():
        node_n, node_m = ttl.node(dims=2)
        for local_m in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m
            if m_block < M_num_blocks:
                for local_n in range(n_blocks_per_node):
                    n_block = node_n * n_blocks_per_node + local_n
                    if n_block < N_num_blocks:
                        out_blk = out_dfb.reserve()
                        for _ in range(K_num_blocks):
                            a_blk = a_dfb.wait()
                            b_blk = b_dfb.wait()
                            out_blk += a_blk @ b_blk
                            a_blk.pop()
                            b_blk.pop()
                        out_blk.push()

    @ttl.datamovement()
    def reader():
        node_n, node_m = ttl.node(dims=2)
        for local_m in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m
            if m_block < M_num_blocks:
                m_off = m_block * M_BLOCK
                for local_n in range(n_blocks_per_node):
                    n_block = node_n * n_blocks_per_node + local_n
                    if n_block < N_num_blocks:
                        for kb in range(K_num_blocks):
                            k_off = kb * K_BLOCK
                            with a_dfb.reserve() as a_blk:
                                ttl.copy(
                                    a[
                                        m_off : m_off + M_BLOCK,
                                        k_off : k_off + K_BLOCK,
                                    ],
                                    a_blk,
                                ).wait()

    @ttl.datamovement()
    def writer():
        node_n, node_m = ttl.node(dims=2)
        for local_m in range(m_blocks_per_node):
            m_block = node_m * m_blocks_per_node + local_m
            if m_block < M_num_blocks:
                m_off = m_block * M_BLOCK
                for local_n in range(n_blocks_per_node):
                    n_block = node_n * n_blocks_per_node + local_n
                    if n_block < N_num_blocks:
                        n_off = n_block * N_BLOCK
                        for kb in range(K_num_blocks):
                            k_off = kb * K_BLOCK
                            with b_dfb.reserve() as b_blk:
                                ttl.copy(
                                    b[
                                        k_off : k_off + K_BLOCK,
                                        n_off : n_off + N_BLOCK,
                                    ],
                                    b_blk,
                                ).wait()
                        with out_dfb.wait() as out_blk:
                            ttl.copy(
                                out_blk,
                                out[
                                    m_off : m_off + M_BLOCK,
                                    n_off : n_off + N_BLOCK,
                                ],
                            ).wait()


# =============================================================================
# C++ output: L1 packer accumulation pattern
#   1. Disable before the K loop
#   2. Conditional enable after the first iteration (iv == lb)
#   3. Disable after cb_push_back following the loop
# =============================================================================

# CHECK-CPP-DAG:  int32_t [[ENABLE:v[0-9]+]] = 1;
# CHECK-CPP-DAG:  int32_t [[DISABLE:v[0-9]+]] = 0;
# CHECK-CPP:      PACK((llk_pack_reconfig_l1_acc([[DISABLE]])));
# CHECK-CPP:      for
# CHECK-CPP:        matmul_block(
# CHECK-CPP:        pack_tile
# CHECK-CPP:        if (
# CHECK-CPP-NEXT:   PACK((llk_pack_reconfig_l1_acc([[ENABLE]])));
# CHECK-CPP:      cb_push_back(
# CHECK-CPP:      PACK((llk_pack_reconfig_l1_acc([[DISABLE]])));

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        # 96x32x96 tiles = 3072x1024x3072, 8x8x8 blocks -> 12x4x12 blocks.
        # With an 8x8 grid each core handles ceil(12/8)=2 M-blocks and
        # 2 N-blocks (4 output blocks), exercising the per-block L1 acc
        # disable/re-enable sequence.
        Mt, Kt, Nt = 96, 32, 96
        M, K, N = Mt * TILE, Kt * TILE, Nt * TILE

        a_torch = torch.randn(M, K, dtype=torch.bfloat16)
        b_torch = torch.randn(K, N, dtype=torch.bfloat16)
        golden = (a_torch.float() @ b_torch.float()).float()

        a_dev = ttnn.from_torch(
            a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b_dev = ttnn.from_torch(
            b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out_dev = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        matmul_l1_acc(a_dev, b_dev, out_dev)

        result = ttnn.to_torch(out_dev).float()
        pcc = torch.corrcoef(torch.stack([result.flatten(), golden.flatten()]))[
            0, 1
        ].item()
        if pcc > 0.999:
            print("PASS")
        else:
            print(f"FAIL: PCC {pcc:.6f} < 0.999")

    finally:
        ttnn.close_device(device)
