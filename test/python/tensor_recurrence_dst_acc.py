# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP --implicit-check-not=add_binary_tile --implicit-check-not=pack_reconfig_l1_acc < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""Additive tensor recurrence lowers to DST-resident accumulation.

`acc = acc + delta` over a compute loop, with the contribution waited once per
iteration, is DST-eligible: the compiler seeds the accumulator DST slot once,
accumulates in place with `binary_dest_reuse_tiles` across the loop, and packs
the result once. This guards against a regression where the recurrence silently
falls back to per-iteration L1-materialized state (per-iteration binary add and
pack), which a numeric-only test cannot detect because both produce the same
result.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

TILE = 32
N_ITERS = 4


@ttl.operation(grid=(1, 1))
def acc_recurrence(initial, delta, out):
    initial_dfb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
    delta_dfb = ttl.make_dataflow_buffer_like(delta, shape=(1, 1), block_count=N_ITERS)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with initial_dfb.wait() as acc:
            for _ in range(N_ITERS):
                with delta_dfb.wait() as delta_blk:
                    acc = acc + delta_blk
            with out_dfb.reserve() as out_blk:
                out_blk.store(acc)

    @ttl.datamovement()
    def reader():
        with initial_dfb.reserve() as initial_blk:
            ttl.copy(initial[0:1, 0:1], initial_blk).wait()
        for _ in range(N_ITERS):
            with delta_dfb.reserve() as delta_blk:
                ttl.copy(delta[0:1, 0:1], delta_blk).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:1, 0:1]).wait()


# DST-resident accumulation in the generated compute C++: seed the accumulator
# DST slot once with copy_tile, accumulate in place with binary_dest_reuse_tiles,
# then pack once. --implicit-check-not rejects the L1-materialized fallback
# (per-iteration add_binary_tile, pack_reconfig_l1_acc).
# CHECK-CPP: copy_tile
# CHECK-CPP: binary_dest_reuse_tiles
# CHECK-CPP: pack_tile

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        initial = torch.randn(TILE, TILE, dtype=torch.bfloat16)
        delta = torch.randn(TILE, TILE, dtype=torch.bfloat16)
        golden = initial.float() + N_ITERS * delta.float()

        def to(t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

        out_dev = to(torch.zeros_like(initial))
        acc_recurrence(to(initial), to(delta), out_dev)

        result = ttnn.to_torch(out_dev).float()
        pcc = torch.corrcoef(
            torch.stack([result.flatten(), golden.flatten()])
        )[0, 1].item()
        if pcc > 0.999:
            print("PASS")
        else:
            print(f"FAIL: PCC {pcc:.6f} < 0.999")
    finally:
        ttnn.close_device(device)
