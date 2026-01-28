#
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
    rows = y.shape[0] // TILE_SIZE
    cols = y.shape[1] // TILE_SIZE

    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    y_cb = ttl.make_circular_buffer_like(y, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def demo_compute():
        with c_cb.wait() as c_blk:
            for _ in range(rows):
                for _ in range(cols):
                    with (
                        a_cb.wait() as a_blk,
                        b_cb.wait() as b_blk,
                        y_cb.reserve() as y_blk,
                    ):
                        a_bcast = ttl.math.broadcast(a_blk, y_blk, dims=[1])
                        b_bcast = ttl.math.broadcast(b_blk, y_blk, dims=[0])
                        c_bcast = ttl.math.broadcast(c_blk, y_blk, dims=[0, 1])
                        y_blk.store(a_bcast * b_bcast + c_bcast)

    @ttl.datamovement()
    def demo_read():
        with c_cb.reserve() as c_blk:
            tx_c = ttl.copy(
                c[0, 0],
                c_blk,
            )

        tx_c.wait()

        for row in range(rows):
            for col in range(cols):
                with (
                    a_cb.reserve() as a_blk,
                    b_cb.reserve() as b_blk,
                ):
                    tx_a = ttl.copy(
                        a[row, 0],
                        a_blk,
                    )
                    tx_b = ttl.copy(
                        b[0, col],
                        b_blk,
                    )

                    tx_a.wait()
                    tx_b.wait()

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
    y = from_torch(
        torch.zeros(
            (
                max(a.shape[0], b.shape[0], c.shape[0]),
                max(a.shape[1], b.shape[1], c.shape[1]),
            ),
            dtype=torch.bfloat16,
        )
    )
    __demo_kernel(a, b, c, y)
    return y


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    shape = (2048, 2048)

    a = torch.rand((shape[0], 1), dtype=torch.bfloat16)
    b = torch.rand((1, shape[1]), dtype=torch.bfloat16)
    c = torch.rand((1, 1), dtype=torch.bfloat16)
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
