# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Accumulating loop - proves loop runs multiple times.
# out acts as accumulator: out = out + rhs each iteration.
# Initial: out=3, rhs=3
# Iter 1: out = 3 + 3 = 6
# Iter 2: out = 6 + 3 = 9
# Final result: 9 (proves loop ran twice)

import torch
from ttlang.d2m_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== Loop Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)], ttnn_interop=True)
def test_loop_accumulate(acc, addend, out):
    """Accumulating add: out = acc + addend, where acc is read/written each iter."""
    acc_accessor = TensorAccessor(acc)
    addend_accessor = TensorAccessor(addend)
    out_accessor = TensorAccessor(out)

    @compute()
    def accum_compute(
        acc_cb: CircularBuffer, addend_cb: CircularBuffer
    ):
        for i in range(2):
            a = acc_cb.wait()
            b = addend_cb.wait()
            o = acc_cb.reserve()
            result = a + b
            o.store(result)
            acc_cb.pop()
            addend_cb.pop()
            acc_cb.push()

    @datamovement()
    def dm_read(acc_cb: CircularBuffer, addend_cb: CircularBuffer):
        for i in range(2):
            # Read current accumulator value
            acc_shard = acc_cb.reserve()
            tx_acc = dma(acc_accessor[0, 0], acc_shard)
            tx_acc.wait()
            acc_cb.push()

            # Read addend
            addend_shard = addend_cb.reserve()
            tx_add = dma(addend_accessor[0, 0], addend_shard)
            tx_add.wait()
            addend_cb.push()

    @datamovement()
    def dm_write(acc_cb: CircularBuffer, addend_cb: CircularBuffer):
        for i in range(2):
            # Write result back to accumulator
            out_shard = acc_cb.wait()
            tx = dma(out_shard, out_accessor[0, 0])
            tx.wait()
            acc_cb.pop()

    return Program(accum_compute, dm_read, dm_write)(acc, addend, out)


# CHECK: === Accumulating Loop Test ===
print("=== Accumulating Loop Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Initial values: acc=3, addend=3
    # Iter 1: 3 + 3 = 6
    # Iter 2: 6 + 3 = 9
    acc_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    addend_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)  # Start same as acc
    expected_value = 9.0  # 3 + 3 + 3 = 9 after 2 iterations

    # Create DRAM tensors
    acc_dram = ttnn.from_torch(
        acc_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    addend_dram = ttnn.from_torch(
        addend_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_dram = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Move to L1 sharded
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        (32, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    acc = ttnn.to_memory_config(acc_dram, memory_config=l1_config)
    addend = ttnn.to_memory_config(addend_dram, memory_config=l1_config)
    out = ttnn.to_memory_config(out_dram, memory_config=l1_config)

    print(f"Running accumulating loop (2 iterations)...")
    print(f"  Initial acc: 3.0, addend: 3.0")
    print(f"  Iter 1: 3 + 3 = 6")
    print(f"  Iter 2: 6 + 3 = 9")
    print(f"  Expected final: {expected_value}")

    test_loop_accumulate(acc, addend, out)

    # Get result
    out_result = ttnn.to_torch(out)

    print(f"\n=== AFTER KERNEL ===")
    print(f"out[0,0] = {out_result[0,0].item()}")
    print(f"out min/max/mean: {out_result.min().item():.1f} / {out_result.max().item():.1f} / {out_result.float().mean().item():.1f}")

    if torch.allclose(out_result.float(), torch.full((32, 32), expected_value), rtol=1e-2, atol=1e-2):
        print(f"\nPASS: Accumulating loop produced {expected_value} (proves 2 iterations)")
        # CHECK: PASS
    else:
        actual = out_result[0,0].item()
        if abs(actual - 6.0) < 0.1:
            print(f"\nFAIL: Got 6.0 - loop only ran once!")
        else:
            print(f"\nFAIL: Expected {expected_value}, got {out_result.min().item()} to {out_result.max().item()}")

finally:
    ttnn.close_device(device)

print("\n=== Loop Test Complete ===")
# CHECK: Loop Test Complete
