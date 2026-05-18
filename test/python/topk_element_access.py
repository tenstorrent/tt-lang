# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
TopK-style element access kernel -- exercises pairwise compare-and-swap
and reduction scan patterns using ttl.unsafe.element_read/write.

Mirrors the control flow of tt-metal's bitonic topk (ckernel_sfpu_topk.h):

  Phase 1: Reduction scan finding the "best" element (like the local
           argmax in deepseek sampling's phase1_reduce_local_values and
           ttnn argmax's compare_values loop).
  Phase 2: Pairwise compare-and-swap at a fixed distance (like one step
           of a bitonic sorting network -- SFPLOAD/SFPSWAP/SFPSTORE in
           _bitonic_topk_merge).

Compile-only test. Comparisons use equality (==) as a placeholder because
i32 magnitude comparison is incorrect for f32 bit patterns (issue #572).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def topk_element_kernel(inp, out):
    """TopK-style kernel: reduction scan + pairwise compare-and-swap.

    Reads a single input tile and produces an output tile containing:
      Row 0, col 0: the "best" value from the reduction scan.
      Row 1: the input row after a pairwise compare-and-swap pass at
             distance 16 (half-tile bitonic merge step).
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
                # -- Phase 1: Reduction scan --
                # Iterate over all elements in the tile row and track the
                # "best" value seen.  Real topk uses bfloat16_greater or
                # hardware SFPSWAP; here we use == as a placeholder.
                best = ttl.unsafe.element_read(rblk, 0, 0)
                for c in range(32):
                    val = ttl.unsafe.element_read(rblk, 0, c)
                    if val == best:
                        best = val
                ttl.unsafe.element_write(wblk, 0, 0, best)

                # -- Phase 2: Pairwise compare-and-swap at distance 16 --
                # Load element pairs (i, i+16), compare, and conditionally
                # swap.  This mirrors one merge step from the bitonic
                # network where elements at distance K are compared:
                #   bitonic_topk_load8(offset, dist)
                #   SFPSWAP(LREG0, LREG1, ALL_ROWS_MAX)
                #   bitonic_topk_store8(offset, dist)
                for i in range(16):
                    a = ttl.unsafe.element_read(rblk, 0, i)
                    b = ttl.unsafe.element_read(rblk, 0, i + 16)
                    ttl.unsafe.element_write(wblk, 1, i, a)
                    ttl.unsafe.element_write(wblk, 1, i + 16, b)
                    if a == b:
                        ttl.unsafe.element_write(wblk, 1, i, b)
                        ttl.unsafe.element_write(wblk, 1, i + 16, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- verify TTL element access ops in both phases
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
# C++ Checks -- loops, conditionals, and ptr operations
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: get_read_ptr
# CHECK-CPP: get_write_ptr
# CHECK-CPP: for
# CHECK-CPP: if


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== TopK Element Access Test ===")
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

        print("Compiling topk element kernel...")
        topk_element_kernel(inp, out)

        print("=== TopK Element Access Test Complete ===")

    finally:
        ttnn.close_device(device)
