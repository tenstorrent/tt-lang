# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_DEBUG_LOCATIONS=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Test that TTLANG_DEBUG_LOCATIONS=1 prints file locations in MLIR output.

Source locations are always generated for error messages, but setting
TTLANG_DEBUG_LOCATIONS=1 enables printing them in dumped MLIR files.
When printed, ops have locations like loc("debug_locations.py":N:M).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"
os.environ["TTLANG_DEBUG_LOCATIONS"] = "1"

from ttlang import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def debug_loc_kernel(lhs, out):
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_thread():
        l = lhs_cb.wait()
        o = out_cb.reserve()
        o.store(l)
        lhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_cb.reserve()
        tx = ttl.copy(lhs[0, 0], lhs_cb)
        tx.wait()
        lhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = ttl.copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_thread, dm_read, dm_write)(lhs, out)


# Verify function definitions exist
# CHECK: func.func @compute_thread

# Verify operations have source locations
# CHECK: ttl.cb_wait %{{.*}} loc(#loc2)
# CHECK: ttl.cb_reserve %{{.*}} loc(#loc3)

# Verify function closing brace has location
# CHECK: } loc(#loc1)

# Verify location aliases are defined with actual file line numbers (at end of MLIR)
# TODO: Figure out a machine-independent way to verify the BASE line number more precisely
# CHECK: #loc1 = loc("{{.*}}debug_locations.py":[[#BASE:]]:1)
# CHECK: #loc2 = loc("{{.*}}debug_locations.py":[[#BASE+1]]:9)
# CHECK: #loc3 = loc("{{.*}}debug_locations.py":[[#BASE+2]]:9)
# CHECK: #loc4 = loc("{{.*}}debug_locations.py":[[#BASE+3]]:5)
# CHECK: #loc5 = loc("{{.*}}debug_locations.py":[[#BASE+4]]:5)
# CHECK: #loc6 = loc("{{.*}}debug_locations.py":[[#BASE+5]]:5)


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        debug_loc_kernel(lhs, out)

    finally:
        ttnn.close_device(device)
