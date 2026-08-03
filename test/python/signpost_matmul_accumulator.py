# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.output
# RUN: %python %s > %t.fpu.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-FPU < %t.fpu.output

"""Verify that a user signpost preserves a matmul/add observation boundary."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.operation(grid=(1, 1))
def signpost_matmul_accumulator(a, b, accumulator, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    accumulator_dfb = ttl.make_dataflow_buffer_like(
        accumulator, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as a_block,
            b_dfb.wait() as b_block,
            accumulator_dfb.wait() as accumulator_block,
            out_dfb.reserve() as out_block,
        ):
            product = a_block @ b_block
            with ttl.signpost("accumulator_add"):
                result = product + accumulator_block
            out_block.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as a_block:
            ttl.copy(a[0, 0], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(b[0, 0], b_block).wait()
        with accumulator_dfb.reserve() as accumulator_block:
            ttl.copy(accumulator[0, 0], accumulator_block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 0]).wait()


# Instrumentation between matmul and add prevents accumulator folding. The
# generated kernel retains separate hardware operations with the user scope
# surrounding only the add.

# CHECK:          warning: instrumentation changes code generation: matmul-accumulator folding is disabled
# CHECK:          matmul_block(
# CHECK-NEXT:     {
# CHECK-NEXT:     DeviceZoneScopedN("ttl_accumulator_add");
# CHECK-NEXT:     copy_tile_init(
# CHECK-NEXT:     copy_tile(
# CHECK-NEXT:     add_binary_tile_init();
# CHECK-NEXT:     add_binary_tile(
# CHECK-NEXT:     }

# CHECK-FPU:      warning: instrumentation changes code generation: matmul-accumulator folding is disabled
# CHECK-FPU:      matmul_block(
# CHECK-FPU-NEXT: {
# CHECK-FPU-NEXT: DeviceZoneScopedN("ttl_accumulator_add");
# CHECK-FPU-NEXT: copy_tile_init(
# CHECK-FPU-NEXT: copy_tile(
# CHECK-FPU-NEXT: add_binary_tile_init();
# CHECK-FPU-NEXT: add_binary_tile(
# CHECK-FPU-NEXT: }


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)
    try:

        def to_device(tensor):
            return ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        a = to_device(torch.randn((32, 32), dtype=torch.bfloat16))
        b = to_device(torch.randn((32, 32), dtype=torch.bfloat16))
        accumulator = to_device(torch.randn((32, 32), dtype=torch.bfloat16))
        out = to_device(torch.zeros((32, 32), dtype=torch.bfloat16))
        signpost_matmul_accumulator(a, b, accumulator, out)
    finally:
        ttnn.close_device(device)
