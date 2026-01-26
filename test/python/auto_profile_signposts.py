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
- CB operation signposts: DeviceZoneScopedN("line_XX_cb_wait_before/after")
- Implicit CB signposts: DeviceZoneScopedN("line_XX_implicit_cb_pop_before/after")
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


@ttl.kernel(grid=(1, 1))
def signpost_test_kernel(inp, out):
    """Simple kernel to test signpost generation."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # 'with' generates cb_wait signpost and implicit cb_pop signpost
        with inp_cb.wait() as i, out_cb.reserve() as o:
            o.store(i)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
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

# Check for cb_wait signpost (explicit CB operation)
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_cb_wait_before")
# CHECK: cb_wait_front(
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_cb_wait_after")

# Check for cb_reserve signpost (explicit CB operation)
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_cb_reserve_before")
# CHECK: cb_reserve_back(
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_cb_reserve_after")

# Check for implicit cb_push signpost (from 'with' exit)
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_implicit_cb_push_before")
# CHECK: cb_push_back(
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_implicit_cb_push_after")

# Check for implicit cb_pop signpost (from 'with' exit)
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_implicit_cb_pop_before")
# CHECK: cb_pop_front(
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_implicit_cb_pop_after")

# Check dm_read kernel has signposts
# CHECK: === dm_read kernel written to
# CHECK: // dm_read
# CHECK: #include "tools/profiler/kernel_profiler.hpp"
# CHECK: void kernel_main()
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_cb_reserve_before")

# Check dm_write kernel has signposts
# CHECK: === dm_write kernel written to
# CHECK: // dm_write
# CHECK: #include "tools/profiler/kernel_profiler.hpp"
# CHECK: void kernel_main()
# CHECK: DeviceZoneScopedN("line_{{[0-9]+}}_cb_wait_before")


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
