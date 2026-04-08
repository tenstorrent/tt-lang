# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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


@ttl.operation(grid=(1, 1))
def __tutorial_operation(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    m_tiles = a.shape[0] // TILE_SIZE
    n_tiles = b.shape[1] // TILE_SIZE
    k_tiles = a.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(1, 1), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, 1), block_count=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(
        c, shape=(1, 1), block_count=2
    )
    acc_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(1, 1), block_count=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(1, 1), block_count=2
    )

    @ttl.datamovement()
    def read():
        for m_tile in range(m_tiles):
            for n_tile in range(n_tiles):
                with c_dfb.reserve() as c_blk:
                    tx_c = ttl.copy(
                        c[m_tile, n_tile],
                        c_blk,
                    )

                    tx_c.wait()

                for k_tile in range(k_tiles):
                    with (
                        a_dfb.reserve() as a_blk,
                        b_dfb.reserve() as b_blk,
                    ):
                        tx_a = ttl.copy(
                            a[m_tile,k_tile],
                            a_blk,
                        )
                        tx_b = ttl.copy(
                            b[k_tile, n_tile],
                            b_blk,
                        )

                        tx_a.wait()
                        tx_b.wait()

    @ttl.compute()
    def compute():
        for _ in range(m_tiles):
            for _ in range(n_tiles):
                with acc_dfb.reserve() as acc_blk:
                    acc_blk.store(ttl.math.fill(acc_blk, 0))

                for _ in range(k_tiles):
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
    
    print(y)
    print(expected_y)

    pcc = torch.corrcoef(
        torch.stack([y.flatten().float(), expected_y.flatten().float()])
    )[0, 1].item()

    print(f"PCC {pcc:.6f}")

    assert pcc > 0.99

finally:
    ttnn.close_device(device)
