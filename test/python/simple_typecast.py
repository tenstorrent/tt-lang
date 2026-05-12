# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: %python %s > %t.hw.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.hw.output

"""
Simple typecast kernel: convert a bf16 input tensor to f32 elementwise.

Verifies that the Python ``ttl.math.typecast`` wrapper lowers to ``ttl.typecast``
in initial IR and to ``typecast_tile_init`` / ``typecast_tile`` in the
generated compute kernel C++. Also verifies numerical correctness against the
torch bf16->f32 reference cast.
"""

import os

import torch
import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def typecast_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(ttl.math.typecast(x, torch.float32))

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
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @compute_fn
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# Bind input (bf16) and output (f32) circular buffers.
# CHECK: %[[IN_CB:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[OUT_CB:.+]] = ttl.bind_cb{cb_index = 1

# Wait for input, attach to its CB.
# CHECK: ttl.cb_wait %[[IN_CB]]
# CHECK: ttl.attach_cb %{{.*}}, %[[IN_CB]]

# Reserve output DFB.
# CHECK: ttl.cb_reserve %[[OUT_CB]]

# Typecast op changes the element type via the result tensor type.
# CHECK: ttl.typecast %{{.*}} : (tensor<{{.*}}, bf16>{{.*}}) -> tensor<{{.*}}, f32>{{.*}}
# CHECK: ttl.store

# =============================================================================
# C++ Kernel Checks
# =============================================================================

# CHECK-CPP: // compute_fn
# CHECK-CPP: void kernel_main()

# init_sfpu uses the input (bf16) and output (f32) CB formats.
# CHECK-CPP: init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(1));

# Tile is loaded into DST then typecast in-place; then packed.
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP: typecast_tile_init<{{.*}}>();
# CHECK-CPP: typecast_tile<{{.*}}>(
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();
# CHECK-CPP: pack_tile<true>(
# CHECK-CPP: tile_regs_release();

# =============================================================================
# Runtime Correctness Checks
# =============================================================================

# CHECK-RESULT: PASSED: bf16->f32 typecast matches torch reference


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        inp_torch = torch.rand((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.float32)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        typecast_kernel(inp, out)

        if not os.environ.get("TTLANG_COMPILE_ONLY"):
            result = ttnn.to_torch(out)
            # bf16->f32 is a lossless widening: every bf16 value has an exact
            # f32 representation, so the hardware result must match exactly.
            expected = inp_torch.to(torch.float32)
            max_diff = (result - expected).abs().max().item()
            assert max_diff == 0.0, f"bf16->f32 typecast mismatch: max_diff={max_diff}"
            print("PASSED: bf16->f32 typecast matches torch reference")
    finally:
        ttnn.close_device(device)
