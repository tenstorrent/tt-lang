# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Invalid test: bcast on elementwise result (not CB-attached).

This should fail because bcast must read from CB, not from DST.
When bcast's input is an elementwise result, the input is in DST,
which is not supported. The DSL catches this early with a clear error:
"bcast input must be attached to a circular buffer".
"""

import ttl


@ttl.kernel(grid=(1, 1))
def invalid_bcast_kernel(a, b, out):
    """INVALID: bcast on elementwise result."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, out_cb.reserve() as o:
            # First do elementwise
            ab = a_tile * b_tile
            # Then try to bcast the elementwise result - INVALID
            result = ttl.math.broadcast(ab, o, dims=[0])
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

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


def main():
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)

    a_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    b_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
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
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    try:
        invalid_bcast_kernel(a, b, out)
        print("FAIL: Expected error for bcast on elementwise result")
    except RuntimeError as e:
        # DSL catches this early with a clear error
        if "bcast input must be attached to a circular buffer" in str(e):
            print("PASS: Got expected error for bcast on non-CB-attached input")
        else:
            print(f"FAIL: Got unexpected error: {e}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
