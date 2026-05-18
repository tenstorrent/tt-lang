# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Argmax element access kernel -- exercises cross-scope variable tracking and
multi-row reduction using ttl.unsafe.element_read/write.

Mirrors the control flow of tt-metal's argmax kernels:
  - reader_argmax_interleaved.cpp scans elements within tiles,
    comparing each to the current best via compare_values().
  - argmax_common.hpp tracks both the max value and its column index
    using update_max_if_greater / calculate_argmax_index.

Two reduction passes over different tile rows:
  Row 0: track max value, write it to output.
  Row 1: track max value and encode the column position as c * 32 + 1
         (matching the position encoding from reader_argmax_interleaved).

Compile-only test. Comparisons use equality (==) as a placeholder because
i32 magnitude comparison is incorrect for f32 bit patterns (issue #572).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def argmax_element_kernel(inp, out):
    """Argmax-style kernel: multi-row reduction with position tracking.

    Scans two tile rows element-by-element.  For each row, tracks the
    "best" value via a cross-scope variable update and records the column
    position.  Writes the max value and encoded position to the output.
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
                # -- Row 0 reduction: find the max value --
                max_val = ttl.unsafe.element_read(rblk, 0, 0)
                for c in range(32):
                    val = ttl.unsafe.element_read(rblk, 0, c)
                    # BOGUS: == placeholder for magnitude comparison
                    if val == max_val:
                        max_val = val
                ttl.unsafe.element_write(wblk, 0, 0, max_val)

                # -- Row 1 reduction: find max and encode position --
                # Mirrors argmax_common.hpp's update_max_if_greater()
                # and calculate_argmax_index() pattern.
                best = ttl.unsafe.element_read(rblk, 1, 0)
                for c in range(32):
                    val = ttl.unsafe.element_read(rblk, 1, c)
                    if val == best:
                        best = val
                        # Position encoding from reader_argmax_interleaved
                        ttl.unsafe.element_write(wblk, 1, 0, c * 32 + 1)
                ttl.unsafe.element_write(wblk, 1, 1, best)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- verify element access ops in both reduction passes
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_write
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_read
# CHECK: ttl.unsafe_element_write
# CHECK: ttl.unsafe_element_write

# =============================================================================
# C++ Checks -- cross-scope vars, conditionals, ptr operations
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: get_read_ptr
# CHECK-CPP: get_write_ptr
# CHECK-CPP: if


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Argmax Element Access Test ===")
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

        print("Compiling argmax element kernel...")
        argmax_element_kernel(inp, out)

        print("=== Argmax Element Access Test Complete ===")

    finally:
        ttnn.close_device(device)
