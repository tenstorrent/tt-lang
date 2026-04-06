# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.output
# RUN: %python %s > %t.fpu.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-FPU < %t.fpu.output

"""
Auto profiler on a kernel with 3 stores from a single compute block.
Verifies that multiple tile_stores are correctly placed and profiled.
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
def multi_store_kernel(a, b, out1, out2, out3):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(1, 1), block_count=2)
    out2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(1, 1), block_count=2)
    out3_dfb = ttl.make_dataflow_buffer_like(out3, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as av,
            b_dfb.wait() as bv,
            out1_dfb.reserve() as o1,
            out2_dfb.reserve() as o2,
            out3_dfb.reserve() as o3,
        ):
            result = av + bv
            o1.store(result)
            o2.store(result)
            o3.store(result)

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


# CHECK:          // compute
# CHECK:          void kernel_main()

# CB waits and reserves wrapped in auto-profile scopes
# CHECK-NOT:      DeviceZoneScopedN(
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_wait");
# CHECK-NEXT:     cb_wait_front(get_compile_time_arg_val(0), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_wait");
# CHECK-NEXT:     cb_wait_front(get_compile_time_arg_val(1), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve");
# CHECK-NEXT:     cb_reserve_back(get_compile_time_arg_val(2), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve");
# CHECK-NEXT:     cb_reserve_back(get_compile_time_arg_val(3), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve");
# CHECK-NEXT:     cb_reserve_back(get_compile_time_arg_val(4), {{.*}});

# Compute body with add and 3 pack_tiles (stores) in a single scope
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}");
# CHECK:          add_binary_tile_init();
# CHECK-NEXT:     add_binary_tile(
# CHECK:          tile_regs_commit();
# CHECK-NEXT:     tile_regs_wait();
# CHECK-NEXT:     pack_tile<true>({{.*}}, get_compile_time_arg_val(4), {{.*}});
# CHECK-NEXT:     pack_tile<true>({{.*}}, get_compile_time_arg_val(3), {{.*}});
# CHECK-NEXT:     pack_tile<true>({{.*}}, get_compile_time_arg_val(2), {{.*}});
# CHECK-NEXT:     tile_regs_release();

# 3 cb_push_back and 2 cb_pop_front with auto-profile scopes
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push");
# CHECK-NEXT:     cb_push_back(get_compile_time_arg_val(4), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push");
# CHECK-NEXT:     cb_push_back(get_compile_time_arg_val(3), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push");
# CHECK-NEXT:     cb_push_back(get_compile_time_arg_val(2), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_pop");
# CHECK-NEXT:     cb_pop_front(get_compile_time_arg_val(1), {{.*}});
# CHECK:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_pop");
# CHECK-NEXT:     cb_pop_front(get_compile_time_arg_val(0), {{.*}});
# CHECK-NOT:      DeviceZoneScopedN(

# =============================================================================
# FPU path checks (default: --ttl-maximize-dst --ttl-fpu-binary-ops)
# Single tile add uses FPU binary (add_tiles), 3 stores unchanged
# =============================================================================

# CHECK-FPU:          // compute
# CHECK-FPU:          void kernel_main()
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_wait");
# CHECK-FPU-NEXT:     cb_wait_front(get_compile_time_arg_val(0), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_wait");
# CHECK-FPU-NEXT:     cb_wait_front(get_compile_time_arg_val(1), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve");
# CHECK-FPU-NEXT:     cb_reserve_back(get_compile_time_arg_val(2), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve");
# CHECK-FPU-NEXT:     cb_reserve_back(get_compile_time_arg_val(3), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_cb_reserve");
# CHECK-FPU-NEXT:     cb_reserve_back(get_compile_time_arg_val(4), {{.*}});

# Compute body: FPU binary add with 3 pack_tiles
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}");
# CHECK-FPU:          binary_op_init_common(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(4));
# CHECK-FPU-NEXT:     tile_regs_acquire();
# CHECK-FPU-NEXT:     add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
# CHECK-FPU-NEXT:     add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
# CHECK-FPU-NEXT:     tile_regs_commit();
# CHECK-FPU-NEXT:     tile_regs_wait();
# CHECK-FPU-NEXT:     pack_tile<true>({{.*}}, get_compile_time_arg_val(4), {{.*}});
# CHECK-FPU-NEXT:     pack_tile<true>({{.*}}, get_compile_time_arg_val(3), {{.*}});
# CHECK-FPU-NEXT:     pack_tile<true>({{.*}}, get_compile_time_arg_val(2), {{.*}});
# CHECK-FPU-NEXT:     tile_regs_release();

# 3 cb_push_back and 2 cb_pop_front with auto-profile scopes
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push");
# CHECK-FPU-NEXT:     cb_push_back(get_compile_time_arg_val(4), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push");
# CHECK-FPU-NEXT:     cb_push_back(get_compile_time_arg_val(3), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_push");
# CHECK-FPU-NEXT:     cb_push_back(get_compile_time_arg_val(2), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_pop");
# CHECK-FPU-NEXT:     cb_pop_front(get_compile_time_arg_val(1), {{.*}});
# CHECK-FPU:          DeviceZoneScopedN("compute_L{{[0-9]+}}_implicit_cb_pop");
# CHECK-FPU-NEXT:     cb_pop_front(get_compile_time_arg_val(0), {{.*}});
# CHECK-FPU-NOT:      DeviceZoneScopedN(

if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        a = ttnn.from_torch(
            torch.full((32, 32), 2.0, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            torch.full((32, 32), 3.0, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out1 = ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out2 = ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out3 = ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        multi_store_kernel(a, b, out1, out2, out3)

    finally:
        ttnn.close_device(device)
