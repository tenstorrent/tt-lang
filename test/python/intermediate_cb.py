# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
P3 BUG REPRO: Intermediate CB storage pattern fails to compile

Compiler Error:
    static assertion failed: Index out of range
    static_assert(Idx < kernel_compile_time_args.size(), "Index out of range");
    note: the comparison reduces to '(2 < 2)'

The compiler generates code referencing compile-time arg index 2,
but only allocated indices 0 and 1.

This pattern is meant to break fusion chains to avoid register clobbering (P1).
Since it doesn't compile, there's no clean in-kernel workaround for P1.
"""

import ttnn
import torch
import ttl


@ttl.kernel(grid=(1, 1))
def intermediate_cb_test(x, out):
    """
    Attempt to use intermediate CB to break fusion.

    Goal: compute exp(relu(x)) by storing relu result to intermediate CB
    """
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    intermediate_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Step 1: Read input, compute relu, store to intermediate
        with x_cb.wait() as xv:
            with intermediate_cb.reserve() as inter:
                relu_result = ttl.math.relu(xv)
                inter.store(relu_result)

        # Step 2: Read from intermediate, compute exp, store to output
        with intermediate_cb.wait() as rv:
            with out_cb.reserve() as o:
                result = ttl.math.exp(rv)  # Using exp, not power
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

    return ttl.Program(compute, dm_read, dm_write)(x, out)


def test_intermediate_cb():
    """Test intermediate CB pattern."""
    device = ttnn.open_device(device_id=0)

    print("=" * 60)
    print("P3 TEST: Intermediate CB storage pattern")
    print("=" * 60)

    x_torch = torch.tensor(
        [[-1.0, 0.0, 1.0, 2.0, 3.0] + [1.0] * 27] * 32, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    x_t = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_t = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Expected: exp(relu(x))
    expected = torch.exp(torch.relu(x_torch))

    print(f"Input:    {x_torch[0, :5]}")
    print(f"Expected: {expected[0, :5]}")

    intermediate_cb_test(x_t, out_t)
    result = ttnn.to_torch(out_t)

    print(f"Actual:   {result[0, :5]}")

    diff = (result.float() - expected.float()).abs().max()
    print(f"Max diff: {diff}")
    print("PASSED" if diff < 0.5 else "FAILED or BUG")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_intermediate_cb()
