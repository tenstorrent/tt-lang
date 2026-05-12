# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for element access scoping bugs in TTLGenericCompiler (ttl_ast.py).

B3: _alloca_scalar emits the initial memref.store at the current insertion
    point (inside the loop body) instead of at function entry alongside
    the alloca.

B4: New i32 variables in nested loops are placed in symbol_tables[-2]
    (the middle scope) instead of symbol_tables[0] (function scope).
    After the outer loop exits, the variable is lost.

B5: _is_integer_scalar matches i1 (boolean from comparisons), causing
    bool results to be promoted to memref<1xi32> when updating an
    outer-scope variable.

Missing memref.load: Variables promoted to memref are never loaded back.
    After loop exit, the outer-scope variable still holds the original SSA
    value, silently discarding all loop updates.
"""

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-B3 < %t.initial.mlir

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl


# === B3: Outer variable updated inside loop ===
# After fix, element_write should use a memref.load result (not %6).

@ttl.operation(grid=(1, 1))
def b3_outer_update_kernel(inp, out):
    """Update an outer-scope variable inside a for loop. After the loop,
    the variable should reflect the last iteration's value."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as blk:
            pass
        with out_dfb.reserve() as oblk:
            pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
            blk.push()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            best = ttl.element_read(rblk, 0, 0)
            with out_dfb.reserve() as wblk:
                for c in range(16):
                    val = ttl.element_read(rblk, 0, c)
                    best = val
                ttl.element_write(wblk, 0, 0, best)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()
                wblk.pop()
            rblk.pop()


# Verify dm_write has proper memref-based cross-scope variable handling.
# The alloca is at function entry; memref.load must appear after the scf.for
# so element_write uses the loop-updated value (not the stale original SSA).
#
# CHECK-B3-LABEL: func.func @dm_write
# CHECK-B3: memref.alloca
# CHECK-B3: ttl.element_read
# CHECK-B3: scf.for
# CHECK-B3: memref.store
# CHECK-B3: }
# CHECK-B3: memref.load
# CHECK-B3: ttl.element_write


inp = ttnn.from_torch(
    torch.randn(32, 32, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)
out = ttnn.from_torch(
    torch.zeros(32, 32, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)

print("Compiling B3 outer update kernel...")
try:
    b3_outer_update_kernel(inp, out)
    print("B3: COMPILED OK")
except Exception as e:
    print(f"B3: COMPILE ERROR: {e}")

print("=== Element Access Scope Tests Complete ===")
