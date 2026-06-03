# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Elementwise add over an 11x1 (prime row dim) tile block.

Exercises remainder peeling in ttl-subblock-compute-for-dst end to end: at the
bf16 DST budget of 8 the divisor heuristic cannot subdivide the prime row dim
(11 > 8) and leaves the whole block unsubblocked; the rescue then raises the row
dim to 8, giving subblock (8,1) which does not divide 11. The pass peels it into
an 8x1 main block plus a 3x1 remainder, and this test confirms the peeled IR
lowers all the way to compute C++ without error.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl

ROW_TILES = 11
COL_TILES = 1


@ttl.operation(grid=(1, 1))
def add_prime(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(ROW_TILES, COL_TILES), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(ROW_TILES, COL_TILES), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(ROW_TILES, COL_TILES), block_count=2)

    @ttl.compute()
    def add_compute():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        o.store(l + r)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        tx_l = ttl.copy(lhs[0:ROW_TILES, 0:COL_TILES], lhs_blk)
        tx_l.wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        tx_r = ttl.copy(rhs[0:ROW_TILES, 0:COL_TILES], rhs_blk)
        tx_r.wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:ROW_TILES, 0:COL_TILES])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Initial IR: the compute is one 11x3 ttl.compute (subblocking runs later in the
# device pipeline, not in the initial dump).
# =============================================================================

# CHECK-LABEL: func.func @add_compute
# CHECK: ttl.add
# CHECK: ttl.store

# =============================================================================
# C++ compute kernel: the peeled kernel must lower to a valid compute kernel
# with add + pack. (Correctness of the offsets is covered numerically below and
# by the lit test subblock_prime.mlir.)
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: pack_tile
# CHECK-CPP: tile_regs_release();


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Prime-block (11x3) Add Test ===")
    require_hardware()

    # Numeric check runs only with a device; the COMPILE_ONLY path above already
    # exercises codegen. Drop COMPILE_ONLY so the op actually executes.
    os.environ.pop("TTLANG_COMPILE_ONLY", None)

    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        shape = (ROW_TILES * 32, COL_TILES * 32)
        lhs_t = torch.randn(shape, dtype=torch.bfloat16)
        rhs_t = torch.randn(shape, dtype=torch.bfloat16)
        out_t = torch.zeros(shape, dtype=torch.bfloat16)

        def to_dev(t):
            d = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.to_memory_config(d, memory_config=ttnn.L1_MEMORY_CONFIG)

        lhs, rhs, out = to_dev(lhs_t), to_dev(rhs_t), to_dev(out_t)

        add_prime(lhs, rhs, out)

        result = ttnn.to_torch(out)
        expected = lhs_t + rhs_t
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), (
            "prime-block add mismatch:\n"
            f"max abs err = {(result.float() - expected.float()).abs().max()}"
        )
        print("PRIME SUBBLOCK PASS")
    finally:
        ttnn.close_device(device)
