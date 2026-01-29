# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch


def from_torch(t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


import ttl

TILE_SIZE = 32


@ttl.kernel(grid=(1, 1))
def __demo_kernel(a, b, c, y):
    rows = a.shape[0] // TILE_SIZE
    cols = a.shape[1] // TILE_SIZE

    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    y_cb = ttl.make_circular_buffer_like(y, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def demo_compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a_cb.wait() as a_blk,
                    b_cb.wait() as b_blk,
                    c_cb.wait() as c_blk,
                    y_cb.reserve() as y_blk,
                ):
                    y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def demo_read():
        for row in range(rows):
            for col in range(cols):
                with (
                    a_cb.reserve() as a_blk,
                    b_cb.reserve() as b_blk,
                    c_cb.reserve() as c_blk,
                ):
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

                    tx_a.wait()
                    tx_b.wait()
                    tx_c.wait()

    @ttl.datamovement()
    def demo_write():
        for row in range(rows):
            for col in range(cols):
                with y_cb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[row, col],
                    )
                    tx.wait()


def demo_kernel(a, b, c):
    y = from_torch(torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16))
    __demo_kernel(a, b, c, y)
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

    y = ttnn.multiply(demo_kernel(a, b, c), d)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
