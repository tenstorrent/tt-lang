# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
KV cache update element access kernel -- exercises the read-modify-write
pattern used by tt-metal's paged cache update pipeline.

Mirrors the 3-phase untilize -> patch -> tilize pattern from
ttnn's update_cache writer kernel and deepseek's KVCacheUpdate op:

  1. Read the "new token" value (from a different tile row, standing in
     for the separate input tensor in a real KV cache update).
  2. Loop over every column in the cache row, copying each element to
     the output tile.
  3. At positions where the cache value matches the new value (bogus
     equality check standing in for the real position comparison
     ``col == cur_pos``), overwrite with the new value.

Compile-only test. The position check uses equality (==) as a placeholder
because comparing a loop index to a constant is not yet exercised in the
element access lowering (issue #572 applies to magnitude comparisons).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def kv_cache_update_kernel(inp, out):
    """KV cache update: copy a tile row, conditionally patch positions.

    Uses a single input tile where:
      Row 0 represents the existing cache row (to be mostly preserved).
      Row 1 represents the new KV values (source for the patch).
    The output tile row 0 contains the updated cache row.
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                # Read the new token value that will be patched into the
                # cache.  In a real KV cache update this comes from a
                # separate CB holding the new KV projection output.
                new_val = ttl.unsafe.element_read(rblk, 1, 0)

                # Copy cache row (row 0) element-by-element, patching
                # positions that match the new value.
                #
                # Real KV cache update:
                #   offset = update_idx % TILE_HEIGHT * Wbytes
                #   noc_async_read(input_addr, cache_addr + offset, Wbytes)
                # Here we iterate element-by-element and conditionally
                # overwrite, mirroring the same read-modify-write intent.
                for c in range(32):
                    cache_val = ttl.unsafe.element_read(rblk, 0, c)
                    ttl.unsafe.element_write(wblk, 0, c, cache_val)
                    # BOGUS: == placeholder for the position check
                    # (real kernel checks col == cur_pos)
                    if cache_val == new_val:
                        ttl.unsafe.element_write(wblk, 0, c, new_val)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- verify element read/write ops in the cache loop
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_write
# CHECK: ttl.unsafe_element_write

# =============================================================================
# C++ Checks -- loop, conditional, ptr operations
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: get_read_ptr
# CHECK-CPP: get_write_ptr
# CHECK-CPP: if


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== KV Cache Element Access Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn(32, 32, dtype=torch.float32)
        out_torch = torch.zeros(32, 32, dtype=torch.float32)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling kv_cache_update kernel...")
        kv_cache_update_kernel(inp, out)

        print("=== KV Cache Element Access Test Complete ===")

    finally:
        ttnn.close_device(device)
