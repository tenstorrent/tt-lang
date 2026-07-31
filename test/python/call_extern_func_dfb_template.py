# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Verify direct DFB and ttnn-created semaphore args in call_extern_func.

The frontend accepts DFBs directly in template_args, materializes them through
the opaque-call pipeline, and lowers DFB/semaphore values to integer template
and function arguments.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


class _FakeGlobalSemaphore:
    __module__ = "ttnn._ttnn.global_semaphore"


class _FakeLocalSemaphore:
    __module__ = "ttnn._ttnn.local_semaphore"


# Compile-only lit tests should not require hardware device bring-up.
# Provide a deterministic semaphore address while still exercising the
# frontend's semaphore auto-detection path.
ttnn.get_global_semaphore_address = lambda _sem: 1234
ttnn.get_local_semaphore_address = lambda _sem: 2345
GLOBAL_SEM = _FakeGlobalSemaphore()
LOCAL_SEM = _FakeLocalSemaphore()


@ttl.operation(grid=(1, 1))
def extern_dfb_template_kernel(inp):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    scratch_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[scratch_dfb, in_dfb, 1, GLOBAL_SEM, LOCAL_SEM],
            func_args=[GLOBAL_SEM, LOCAL_SEM],
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


# CHECK-LABEL: func.func @compute
# CHECK-DAG: %[[SCRATCH:.*]] = ttl.bind_cb{cb_index = 1
# CHECK-DAG: %[[IN_CB:.*]] = ttl.bind_cb{cb_index = 0

# Frontend auto-detects DFB template args and materializes get_dfb_id.
# CHECK: %[[ID_SCRATCH:.*]] = ttl.get_dfb_id %[[SCRATCH]] : <
# CHECK: %[[ID_IN:.*]] = ttl.get_dfb_id %[[IN_CB]] : <
# CHECK: %[[C1:.*]] = arith.constant 1 : i32
# CHECK: %[[GSEM_TPL:.*]] = arith.constant 1234 : i32
# CHECK: %[[LSEM_TPL:.*]] = arith.constant 2345 : i32
# CHECK: %[[GSEM_ARG:.*]] = arith.constant 1234 : i32
# CHECK: %[[LSEM_ARG:.*]] = arith.constant 2345 : i32
# CHECK: ttl.opaque_call "my_shim" template_args(%[[ID_SCRATCH]], %[[ID_IN]], %[[C1]], %[[GSEM_TPL]], %[[LSEM_TPL]]) (%[[GSEM_ARG]], %[[LSEM_ARG]])

# CHECK-CPP: === compute kernel written to {{.*}} ===
# CHECK-CPP: int32_t [[LSEM:[^ ]+]] = 2345;
# CHECK-CPP: int32_t [[GSEM:[^ ]+]] = 1234;
# CHECK-CPP: my_shim<1, 0, 1, 1234, 2345>([[GSEM]], [[LSEM]]);


if __name__ == "__main__":
    import torch

    inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    extern_dfb_template_kernel(inp)
