# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
TopK-style element access kernel -- exercises pairwise compare-and-swap
and reduction scan patterns using ttl.raw_element_read/write.

Mirrors the control flow of tt-metal's bitonic topk (ckernel_sfpu_topk.h):

  Phase 1: Reduction scan finding the "best" element (like the local
           argmax in deepseek sampling's phase1_reduce_local_values and
           ttnn argmax's compare_values loop).
  Phase 2: Pairwise compare-and-swap at a fixed distance (like one step
           of a bitonic sorting network -- SFPLOAD/SFPSWAP/SFPSTORE in
           _bitonic_topk_merge).

Compile-only test.

WARNING: Comparisons use equality (==) as a placeholder because
arith.cmpf lowering is not yet implemented for TTKernel. This test will
fail to compile until that lowering is added.
See https://github.com/tenstorrent/tt-lang/issues/572
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
                best = ttl.raw_element_read(rblk, 0, 0)
                for c in range(32):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val == best:
                        best = val
                ttl.raw_element_write(wblk, 0, 0, best)

                # -- Phase 2: Pairwise compare-and-swap at distance 16 --
                for i in range(16):
                    a = ttl.raw_element_read(rblk, 0, i)
                    b = ttl.raw_element_read(rblk, 0, i + 16)
                    ttl.raw_element_write(wblk, 1, i, a)
                    ttl.raw_element_write(wblk, 1, i + 16, b)
                    if a == b:
                        ttl.raw_element_write(wblk, 1, i, b)
                        ttl.raw_element_write(wblk, 1, i + 16, a)

                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks -- verify TTL element access ops in both phases
# =============================================================================

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_write
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_read
# CHECK: ttl.raw_element_write
# CHECK: ttl.raw_element_write

# =============================================================================
# C++ Checks -- loops, conditionals, and ptr operations
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

topk_element_kernel(inp, out)
ttnn.close_device(device)
