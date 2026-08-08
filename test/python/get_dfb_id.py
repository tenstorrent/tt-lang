# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Verify that ttl.get_dfb_id() in template_args preserves DFB identity in the
initial MLIR and lowers to a raw integer literal in the generated C++.

The compute thread calls ttl.call_extern_func with DFB IDs and a captured
integer assigned to a local alias. The alias exercises the index-typed
arith.constant template-argument branch. The same DFBs are direct function
arguments so the allocator can retain their storage dependencies. Stub
data-movement threads satisfy the TTNN interop 3-kernel requirement.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


def _make_get_dfb_id_kernel(captured_template_value):
    @ttl.operation(grid=(1, 1))
    def get_dfb_id_kernel(inp):
        in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        scratch = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            local_template_value = captured_template_value
            ttl.call_extern_func(
                FAKE_HEADER,
                "my_shim",
                template_args=[
                    ttl.get_dfb_id(scratch),
                    ttl.get_dfb_id(in_dfb),
                    local_template_value,
                ],
                func_args=[scratch, in_dfb],
            )

        @ttl.datamovement()
        def dm_read():
            blk = in_dfb.reserve()
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
            blk.push()

        @ttl.datamovement()
        def dm_write():
            blk = in_dfb.wait()
            blk.pop()

    return get_dfb_id_kernel


get_dfb_id_kernel = _make_get_dfb_id_kernel(1)


# =============================================================================
# Initial IR checks -- verify ordered DFB index references on opaque_call
# =============================================================================

# CHECK-LABEL: func.func @compute
# CHECK-DAG: %[[SCRATCH:.*]] = ttl.bind_cb{cb_index = 1
# CHECK-DAG: %[[IN_CB:.*]] = ttl.bind_cb{cb_index = 0

# DFB references remain explicit until physical allocation is finalized.
# CHECK: ttl.opaque_call "my_shim" template_args [#ttl.external_template_arg<dfb_index, 0>, #ttl.external_template_arg<dfb_index, 1>, #ttl.external_template_arg<signed_integer, 1>] template_dfbs(%[[SCRATCH]], %[[IN_CB]] : !ttl.cb<{{.*}}>, !ttl.cb<{{.*}}>) (%[[SCRATCH]], %[[IN_CB]])

# =============================================================================
# C++ output checks -- verify raw integers in template args
# =============================================================================

# CHECK-CPP: === compute kernel written to {{.*}} ===
# CHECK-CPP: my_shim<[[SCRATCH_INDEX:[0-9]+]]U, [[IN_INDEX:[0-9]+]]U, 1>
# CHECK-CPP-SAME: (get_compile_time_arg_val([[SCRATCH_INDEX]]), get_compile_time_arg_val([[IN_INDEX]]))


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)

        get_dfb_id_kernel(inp)

    finally:
        ttnn.close_device(device)
