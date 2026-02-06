# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Debug broadcast after reduce_max."""

import torch
import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def test_reduce_then_broadcast(inp, scaler, out):
    """Test reduce_max followed by broadcast."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=1)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    max_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    max_bcast_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as iv, scaler_cb.wait() as sc:
            # Step 1: reduce_max
            with max_cb.reserve() as mx:
                mx.store(ttl.math.reduce_max(iv, sc, mx, dims=[0]))
            # Step 2: broadcast the max values
            with max_cb.wait() as mxv, max_bcast_cb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, mxb, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with max_bcast_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def to_ttnn(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    print("=== Debug reduce_max + broadcast ===\n")

    device = ttnn.open_device(device_id=0)

    # Input with varying values per row
    inp = torch.zeros(32, 32, dtype=torch.bfloat16)
    inp[0, :] = torch.arange(32, dtype=torch.bfloat16)  # Row 0: 0,1,2,...,31
    inp[1, :] = torch.arange(32, dtype=torch.bfloat16) + 100  # Row 1: 100,101,...,131
    inp[2, :] = torch.arange(32, dtype=torch.bfloat16) + 50   # Row 2: 50,51,...,81

    scaler = torch.ones(32, 32, dtype=torch.bfloat16)
    out = torch.zeros(32, 32, dtype=torch.bfloat16)

    print("Input (first 3 rows, 8 cols):")
    print(f"  Row 0: {inp[0, :8].tolist()}")
    print(f"  Row 1: {inp[1, :8].tolist()}")
    print(f"  Row 2: {inp[2, :8].tolist()}")

    inp_tt = to_ttnn(inp, device)
    scaler_tt = to_ttnn(scaler, device)
    out_tt = to_ttnn(out, device)

    test_reduce_then_broadcast(inp_tt, scaler_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()

    print("\nOutput after reduce_max + broadcast (first 3 rows, 8 cols):")
    print(f"  Row 0: {result[0, :8].tolist()}")
    print(f"  Row 1: {result[1, :8].tolist()}")
    print(f"  Row 2: {result[2, :8].tolist()}")

    # Expected: each row should have the same value (the max of that row) across all columns
    # Row 0 max = 31, Row 1 max = 131, Row 2 max = 81
    print("\nExpected (max of each row broadcast across all columns):")
    print(f"  Row 0: all 31.0")
    print(f"  Row 1: all 131.0")
    print(f"  Row 2: all 81.0")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
