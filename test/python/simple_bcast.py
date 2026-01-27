# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Simple fused bcast test case: (a * b) + bcast(c)."""

import torch
import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def fused_bcast_kernel(a, b, c, out):
    """Compute (a * b) + bcast(c) where c is a row-broadcast tile.

    Two-stage pattern:
    - Stage 1: bcast c to intermediate CB
    - Stage 2: compute (a * b) + c_bcast
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    c_bcast_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Stage 1: Bcast c and store to intermediate CB
        with c_cb.wait() as c_tile, c_bcast_cb.reserve() as c_out:
            c_bcast = ttl.math.broadcast(c_tile, c_out, dims=[0])
            c_out.store(c_bcast)

        # Stage 2: Compute (a * b) + c_bcast
        with (
            a_cb.wait() as a_tile,
            b_cb.wait() as b_tile,
            c_bcast_cb.wait() as c_bcast_tile,
            out_cb.reserve() as o,
        ):
            ab = a_tile * b_tile
            result = ab + c_bcast_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

        c_blk = c_cb.reserve()
        tx_c = ttl.copy(c[0, 0], c_blk)
        tx_c.wait()
        c_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


def main():
    device = ttnn.open_device(device_id=0)

    # a = full tile of 2.0
    a_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)

    # b = full tile of 3.0
    b_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)

    # c = row tile with first row = 1.0 (for row broadcast)
    c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    c_torch[0, :] = 1.0

    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    c = ttnn.from_torch(
        c_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    fused_bcast_kernel(a, b, c, out)

    result = ttnn.to_torch(out)
    # Expected: (2.0 * 3.0) + 1.0 = 7.0
    expected = torch.full((32, 32), 7.0, dtype=torch.bfloat16)

    print(f"Result unique values: {torch.unique(result).tolist()}")
    print(f"Expected: all 7.0")

    if torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
