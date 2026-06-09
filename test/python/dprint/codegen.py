# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Test that print() in kernel code generates dprint calls in the C++ output.

Verifies all modes from the dprint spec:
- Scalar: string/int/float constants, integer variables
- CB: ttmlir::CBPrinter
- Tile: inline TileSlice loop
- DST: label with live slot info
- Thread conditioning: DPRINT_PACK wrapping
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.operation(grid=(1, 1))
def dprint_test_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            print("compute start")
            print("magic:", 42)
            print(inp_dfb)
            print("cb state:", inp_dfb)
            print(i)
            print("tile:", i)
            result = ttl.exp(i)
            print(_dump_dst_registers=True, label="after exp")
            print(i, thread="pack")
            print("math only", thread="math")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        print("dm_read core:", x, y)
        print("inp:", inp, num_pages=1)
        print("A:", inp)
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# C++ Kernel Checks - Verify dprint in compute kernel
# =============================================================================

# CHECK: === compute kernel written to
# CHECK: // compute
# CHECK: #include "api/debug/dprint.h"
# CHECK: void kernel_main()

# Scalar prints in compute auto-default to math thread
# CHECK: MATH({
# CHECK: DPRINT("compute start\n");
# CHECK: });
# CHECK: MATH({
# CHECK: DPRINT("magic: 42\n");
# CHECK: });

# CB print uses the self-thread-guarded ttmlir::dprint(CBPrinter) helper
# CHECK: ttmlir::dprint(ttmlir::CBPrinter(get_compile_time_arg_val(

# Mixed-arg: scalar label (math) + CB object (self-guarded helper)
# CHECK: MATH({
# CHECK: DPRINT("cb state:\n");
# CHECK: });
# CHECK: ttmlir::dprint(ttmlir::CBPrinter(get_compile_time_arg_val(

# Tile print in compute auto-defaults to pack thread
# CHECK: PACK({
# CHECK: TSLICE(get_compile_time_arg_val(
# CHECK: });

# Mixed-arg: scalar label + tile object
# CHECK: MATH({
# CHECK: DPRINT("tile:\n");
# CHECK: });
# CHECK: PACK({
# CHECK: TSLICE(get_compile_time_arg_val(
# CHECK: });

# DST dump after exp auto-defaults to math thread
# (no live slots because the dprint is outside the fused compute
# body; the store clears all slots before it)
# CHECK: MATH({
# CHECK: DPRINT("=== after exp ===\n");

# Thread conditioning: explicit pack-only tile print
# CHECK: PACK({
# CHECK: TSLICE(get_compile_time_arg_val(
# CHECK: });

# Thread conditioning: explicit math-only scalar print
# CHECK: MATH({
# CHECK: DPRINT("math only\n");
# CHECK: });

# =============================================================================
# C++ Kernel Checks - Verify dprint with variables in dm_read kernel
# =============================================================================

# CHECK: === dm_read kernel written to
# CHECK: // dm_read
# CHECK: #include "api/debug/dprint.h"
# CHECK: void kernel_main()
# CHECK: DPRINT("dm_read core: {} {}\n", get_absolute_logical_x(), get_absolute_logical_y());

# Mixed-arg: scalar label + tensor accessor pages in datamovement
# CHECK: DPRINT("inp:\n");
# CHECK: TensorAccessorArgs<
# CHECK: noc_async_read_tile(

# Tensor accessor without num_pages defaults to num_pages=1
# CHECK: DPRINT("A:\n");
# CHECK: TensorAccessorArgs<
# CHECK: noc_async_read_tile(

# =============================================================================
# C++ Kernel Checks - Verify no dprint in dm_write kernel (no print calls)
# =============================================================================

# CHECK: === dm_write kernel written to
# CHECK: // dm_write
# CHECK-NOT: #include "api/debug/dprint.h"
# CHECK: void kernel_main()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== DPrint Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        print("Compiling kernel with dprint...")
        dprint_test_kernel(inp, out)

        print("=== DPrint Test Complete ===")

    finally:
        ttnn.close_device(device)
