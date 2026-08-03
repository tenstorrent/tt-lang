# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP --implicit-check-not=add_binary_tile --implicit-check-not=pack_reconfig_l1_acc < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""Additive tensor recurrence lowers to DST-resident accumulation.

`acc = acc + delta` over a compute loop copies the initial value into DST,
accumulates in place with `binary_dest_reuse_tiles`, and packs the result once.
The contribution dataflow buffer has fewer blocks than the loop trip count.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32
N_ITERS = 400
CONTRIBUTION_BLOCK_COUNT = 2
MULTI_TILE_SHAPE = (2, 2)
MULTI_TILE_ROWS = MULTI_TILE_SHAPE[0] * TILE
MULTI_TILE_COLS = MULTI_TILE_SHAPE[1] * TILE
DTYPE_CASES = (
    (torch.bfloat16, "BF16", 0.98),
    (torch.float32, "F32", 0.999),
)


@ttl.operation(grid=(1, 1))
def single_tile_acc_recurrence(initial, delta, out):
    initial_dfb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
    delta_dfb = ttl.make_dataflow_buffer_like(
        delta, shape=(1, 1), block_count=CONTRIBUTION_BLOCK_COUNT
    )
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def single_tile_compute():
        with initial_dfb.wait() as acc:
            for _ in range(N_ITERS):
                with delta_dfb.wait() as delta_blk:
                    acc = acc + delta_blk
            with out_dfb.reserve() as out_blk:
                out_blk.store(acc)

    @ttl.datamovement()
    def single_tile_reader():
        with initial_dfb.reserve() as initial_blk:
            ttl.copy(initial[0:1, 0:1], initial_blk).wait()
        for _ in range(N_ITERS):
            with delta_dfb.reserve() as delta_blk:
                ttl.copy(delta[0:1, 0:1], delta_blk).wait()

    @ttl.datamovement()
    def single_tile_writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def multi_tile_acc_recurrence(initial, delta, out):
    initial_dfb = ttl.make_dataflow_buffer_like(
        initial, shape=MULTI_TILE_SHAPE, block_count=2
    )
    delta_dfb = ttl.make_dataflow_buffer_like(
        delta, shape=MULTI_TILE_SHAPE, block_count=CONTRIBUTION_BLOCK_COUNT
    )
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=MULTI_TILE_SHAPE, block_count=2)

    @ttl.compute()
    def multi_tile_compute():
        with initial_dfb.wait() as acc:
            for _ in range(N_ITERS):
                with delta_dfb.wait() as delta_blk:
                    acc = acc + delta_blk
            with out_dfb.reserve() as out_blk:
                out_blk.store(acc)

    @ttl.datamovement()
    def multi_tile_reader():
        with initial_dfb.reserve() as initial_blk:
            ttl.copy(
                initial[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]],
                initial_blk,
            ).wait()
        for _ in range(N_ITERS):
            with delta_dfb.reserve() as delta_blk:
                ttl.copy(
                    delta[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]],
                    delta_blk,
                ).wait()

    @ttl.datamovement()
    def multi_tile_writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(
                out_blk, out[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]]
            ).wait()


@ttl.operation(grid=(1, 1))
def resident_delta_acc_recurrence(initial, delta, out):
    initial_dfb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
    delta_dfb = ttl.make_dataflow_buffer_like(delta, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def resident_delta_compute():
        with delta_dfb.wait() as delta_blk:
            with initial_dfb.wait() as acc:
                for _ in range(N_ITERS):
                    acc = acc + delta_blk
                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

    @ttl.datamovement()
    def resident_delta_reader():
        with initial_dfb.reserve() as initial_blk:
            ttl.copy(initial[0:1, 0:1], initial_blk).wait()
        with delta_dfb.reserve() as delta_blk:
            ttl.copy(delta[0:1, 0:1], delta_blk).wait()

    @ttl.datamovement()
    def resident_delta_writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:1, 0:1]).wait()


# Initial IR check: the resident contribution is acquired once before the loop
# with a dataflow buffer capacity independent of the trip count.
# CHECK-INITIAL-LABEL: func.func @resident_delta_compute
# CHECK-INITIAL: ttl.bind_cb{{.*}}cb_index = 1{{.*}}block_count = 1
# CHECK-INITIAL: arith.constant 400 : i64
# CHECK-INITIAL: scf.for
# CHECK-INITIAL-NOT: ttl.cb_wait
# CHECK-INITIAL: ttl.add

# Generated C++ checks DST initialization, in-place accumulation, and final
# packing.
# CHECK-CPP-LABEL: === single_tile_compute kernel written to {{.*}} ===
# CHECK-CPP: copy_tile(
# CHECK-CPP: binary_dest_reuse_tiles<
# CHECK-CPP: pack_tile
# CHECK-CPP-LABEL: === multi_tile_compute kernel written to {{.*}} ===
# CHECK-CPP-COUNT-4: copy_tile(
# CHECK-CPP-COUNT-4: binary_dest_reuse_tiles<
# CHECK-CPP: pack_tile_block
# CHECK-CPP-LABEL: === resident_delta_compute kernel written to {{.*}} ===
# CHECK-CPP-NOT: add_binary_tile
# CHECK-CPP-NOT: pack_reconfig_l1_acc
# CHECK-CPP-NOT: llk_pack_reconfig_l1_acc
# CHECK-CPP-COUNT-2: wait_front
# CHECK-CPP: for
# CHECK-CPP-NOT: wait_front
# CHECK-CPP: binary_dest_reuse_tiles<
# CHECK-CPP-NOT: pop_front
# CHECK-CPP: }
# CHECK-CPP: pack_tile
# CHECK-CPP-NOT: add_binary_tile
# CHECK-CPP-NOT: pack_reconfig_l1_acc
# CHECK-CPP-NOT: llk_pack_reconfig_l1_acc

# CHECK-RESULT: SINGLE BF16 PASS
# CHECK-RESULT: MULTI BF16 PASS
# CHECK-RESULT: RESIDENT BF16 PASS
# CHECK-RESULT: SINGLE F32 PASS
# CHECK-RESULT: MULTI F32 PASS
# CHECK-RESULT: RESIDENT F32 PASS
# CHECK-RESULT: PASS


def run_single_tile(device, dtype, dtype_label, threshold):
    initial = torch.randn(TILE, TILE, dtype=dtype)
    delta = torch.randn(TILE, TILE, dtype=dtype)
    golden = initial.float() + N_ITERS * delta.float()
    out_dev = to_dram(torch.zeros_like(initial), device)

    single_tile_acc_recurrence(
        to_dram(initial, device), to_dram(delta, device), out_dev
    )

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden.float(), result.float(), threshold=threshold)
    print(f"SINGLE {dtype_label} PASS")


def run_multi_tile(device, dtype, dtype_label, threshold):
    initial = torch.randn(MULTI_TILE_ROWS, MULTI_TILE_COLS, dtype=dtype)
    delta = torch.randn(MULTI_TILE_ROWS, MULTI_TILE_COLS, dtype=dtype)
    golden = initial.float() + N_ITERS * delta.float()
    out_dev = to_dram(torch.zeros_like(initial), device)

    multi_tile_acc_recurrence(to_dram(initial, device), to_dram(delta, device), out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden.float(), result.float(), threshold=threshold)
    print(f"MULTI {dtype_label} PASS")


def run_resident_delta(device, dtype, dtype_label, threshold):
    initial = torch.randn(TILE, TILE, dtype=dtype)
    delta = torch.randn(TILE, TILE, dtype=dtype)
    golden = initial.float() + N_ITERS * delta.float()
    out_dev = to_dram(torch.zeros_like(initial), device)

    resident_delta_acc_recurrence(
        to_dram(initial, device), to_dram(delta, device), out_dev
    )

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden.float(), result.float(), threshold=threshold)
    print(f"RESIDENT {dtype_label} PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        for dtype, dtype_label, threshold in DTYPE_CASES:
            run_single_tile(device, dtype, dtype_label, threshold)
            run_multi_tile(device, dtype, dtype_label, threshold)
            run_resident_delta(device, dtype, dtype_label, threshold)
        print("PASS")
    finally:
        ttnn.close_device(device)
