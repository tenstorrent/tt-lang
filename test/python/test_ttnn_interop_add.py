# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: TTNN interop path with ttnn.Tensors on device.

import torch
import ttnn
import ttl
from test_helpers import to_l1


@ttl.kernel(grid=(1, 1))
def test_ttnn_interop_add(lhs, rhs, out):
    """Simple add kernel compiled for TTNN interop (C++ output)."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Read both inputs - reserve CB and copy directly from tensor to CB
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_out():
        # Write output - wait for data in CB and copy directly from CB to device
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


# CHECK: TTNN INTEROP
# CHECK: Created ProgramDescriptor

print("=== Testing TTNN Interop Path ===")

device = ttnn.open_device(device_id=0)

try:
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.full((32, 32), -999.0, dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch

    # Create tensors in L1
    lhs = to_l1(lhs_torch, device)
    rhs = to_l1(rhs_torch, device)
    out = to_l1(out_torch, device)

    test_ttnn_interop_add(lhs, rhs, out)
    out_result = ttnn.to_torch(out)

    print(f"\nResult[0,0] = {out_result[0, 0].item()}")
    print(f"Expected[0,0] = {expected[0, 0].item()}")

    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: Output matches expected!")
        # CHECK: PASS
    else:
        max_err = (out_result.float() - expected.float()).abs().max().item()
        print(f"\nFAIL: Max error = {max_err:.6f}")

finally:
    ttnn.close_device(device)

print("\n=== TTNN Interop Test Complete ===")
# CHECK: TTNN Interop Test Complete
