# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_AUTO_PROFILE=1 TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Test that auto-profiling generates correct signposts in the C++ kernel output.

Verifies:
- Line-based signposts: DeviceZoneScopedN("line_XX_before/after")
- DFB operation signposts: DeviceZoneScopedN("line_XX_dfb_wait_before/after")
- Implicit DFB signposts: DeviceZoneScopedN("line_XX_implicit_cb_pop_before/after")
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"
os.environ["TTLANG_AUTO_PROFILE"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.operation(grid=(1, 1))
def signpost_test_kernel(inp, out):
    """Simple kernel to test signpost generation."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # 'with' generates cb_wait signpost and implicit cb_pop signpost
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            o.store(i)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# C++ Kernel Checks - Verify signposts in generated code
# =============================================================================

# Check compute kernel has profiler header and signposts
# CHECK: === compute kernel written to
# CHECK: // compute
# CHECK: #include "tools/profiler/kernel_profiler.hpp"
# CHECK: void kernel_main()

# Check for cb_wait signpost (scoped around the operation)
# CHECK: DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_wait")
# CHECK: cb_wait_front(

# Check for cb_reserve signpost (scoped around the operation)
# CHECK: DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve")
# CHECK: cb_reserve_back(

# Check for implicit cb_push signpost (from 'with' exit)
# CHECK: DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push")
# CHECK: cb_push_back(

# Check for implicit cb_pop signpost (from 'with' exit)
# CHECK: DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_pop")
# CHECK: cb_pop_front(

# Check dm_read kernel has signposts
# CHECK: === dm_read kernel written to
# CHECK: // dm_read
# CHECK: #include "tools/profiler/kernel_profiler.hpp"
# CHECK: void kernel_main()
# CHECK: DeviceZoneScopedN("dm_read_L{{[0-9]+}}_cb_reserve")

# Check dm_write kernel has signposts
# CHECK: === dm_write kernel written to
# CHECK: // dm_write
# CHECK: #include "tools/profiler/kernel_profiler.hpp"
# CHECK: void kernel_main()
# CHECK: DeviceZoneScopedN("dm_write_L{{[0-9]+}}_cb_wait")


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Auto-Profile Signpost Test ===")

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

        print("Compiling kernel with auto-profiling enabled...")
        signpost_test_kernel(inp, out)

        print("=== Auto-Profile Signpost Test Complete ===")

    finally:
        ttnn.close_device(device)
