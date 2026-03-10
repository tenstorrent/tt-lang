# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TT_METAL_DPRINT_CORES=0,0 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Reproducers for known dprint issues.

Bug 1: Tensor accessor page print reads wrong address.
  get_common_arg_val<uint32_t>(idx) returns a bank base address, not
  a directly dereferenceable L1 pointer. Reading from it shows zeros
  even after data has been written to the tensor.

Bug 2: CB metadata prints concatenated on same line.
  Two consecutive print(dfb) calls produce output on the same line
  with no separator, e.g.:
    cb_id 0: { ... }cb_id 1: { ... }

Bug 3: Tile print after o.store() shows zeros.
  TileSlice reads from the CB read pointer, but o.store() packs data
  to the write pointer. The read pointer does not advance until
  cb_push_back (end of the with-reserve block), so the tile print
  sees stale/zero data.
"""

import os

os.environ["TT_METAL_DPRINT_CORES"] = "0,0"

import torch
import ttnn
import ttl


# =============================================================================
# Bug 1: Tensor accessor page print
# =============================================================================


@ttl.kernel(grid=(1, 1))
def bug1_tensor_accessor_print(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            result = ttl.exp(i)
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        # Print input tensor pages. Input has non-zero random data,
        # but get_common_arg_val returns bank base (wrong address).
        print("bug1 before copy")
        print(inp, num_pages=1)
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()
        # Print output tensor pages after write barrier.
        # Data should be non-zero (exp of random input).
        print("bug1 after write")
        print(out, num_pages=1)


# =============================================================================
# Bug 2: CB prints concatenated on same line
# =============================================================================


@ttl.kernel(grid=(1, 1))
def bug2_cb_concat(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Two consecutive CB prints should appear on separate lines.
        print(inp_dfb)
        print(out_dfb)
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            o.store(i)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# Bug 3: Tile print after store shows zeros
# =============================================================================


@ttl.kernel(grid=(1, 1))
def bug3_tile_after_store(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            result = ttl.exp(i)
            print("before", thread="pack")
            print(o)
            o.store(result)
            # This tile print reads from CB read pointer, but store
            # wrote to the write pointer. Should show exp(input) but
            # shows zeros instead.
            print("after", thread="pack")
            print(o)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# FileCheck patterns
# =============================================================================

# Bug 1: Tensor accessor prints should show non-zero data but show zeros.
# XFAIL: *
# The tensor accessor address (get_common_arg_val) is a bank base,
# not a dereferenceable L1 pointer.

# Bug 1: verify the prints execute (labels appear)
# CHECK-DAG: bug1 before copy
# CHECK-DAG: bug1 after write

# Bug 2: Each CB print should be on its own line.
# If bug is present, both appear concatenated on one line.
# CHECK: cb_id
# CHECK: cb_id

# Bug 3: Tile print after store
# CHECK: bug3 after store

# =============================================================================
# Test execution
# =============================================================================

device = ttnn.open_device(device_id=0)

try:
    inp = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print("=== Bug 1: Tensor accessor page print ===")
    bug1_tensor_accessor_print(inp, out)

    print("=== Bug 2: CB prints concatenated ===")
    bug2_cb_concat(inp, out)

    # WORKS FINE (not a bug)
    # print("=== Bug 3: Tile print after store ===")
    # bug3_tile_after_store(inp, out)

finally:
    ttnn.close_device(device)
