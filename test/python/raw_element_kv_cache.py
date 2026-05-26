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

Compile-only test.

WARNING: The equality comparison (==) on raw_element_read results requires
arith.cmpf lowering which is not yet implemented for TTKernel. This test
will fail to compile until that lowering is added.
See https://github.com/tenstorrent/tt-lang/issues/572
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
                new_val = ttl.raw_element_read(rblk, 1, 0)

                for c in range(32):
                    cache_val = ttl.raw_element_read(rblk, 0, c)
                    ttl.raw_element_write(wblk, 0, c, cache_val)
                    if cache_val == new_val:
                        ttl.raw_element_write(wblk, 0, c, new_val)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- verify element read/write ops in the cache loop
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_write
# CHECK: ttl.raw_element_write

# =============================================================================
# C++ Checks -- loop, conditional, ptr operations
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: reinterpret_cast<tt_l1_ptr uint32_t*>


device = ttnn.open_device(device_id=0)

inp = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").float32),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
out = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").float32),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

kv_cache_update_kernel(inp, out)
ttnn.close_device(device)
