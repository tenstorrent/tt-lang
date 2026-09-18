# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Verify typed DFB descriptors in call_extern_func.

The frontend preserves descriptor and scalar source order through TTL,
TTKernel, EmitC, and generated C++.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def extern_dfb_template_kernel(inp):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    scratch_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 1), block_count=3)

    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[
                ttl.dfb_descriptor(scratch_dfb),
                ttl.dfb_descriptor(in_dfb),
                -3,
                True,
                -1.0,
            ],
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

# CHECK: ttl.opaque_call "my_shim" template_args [#ttl.external_template_arg<dfb_descriptor, 0>, #ttl.external_template_arg<dfb_descriptor, 1>, #ttl.external_template_arg<signed_integer, -3>, #ttl.external_template_arg<boolean, 1>, #ttl.external_template_arg<unsigned_integer, 3212836864>] template_dfbs(%[[SCRATCH]], %[[IN_CB]] : !ttl.cb<{{.*}}>, !ttl.cb<{{.*}}>) ()

# CHECK-CPP: === compute kernel written to {{.*}} ===
# CHECK-CPP: namespace ttlang {
# CHECK-CPP: struct DFBDescriptor {
# CHECK-CPP: my_shim<ttlang::DFBDescriptor<1, 2, 3, 2048>, ttlang::DFBDescriptor<0, 1, 1, 2048>, -3, true, 3212836864U>();


if __name__ == "__main__":
    import torch
    import ttnn

    inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    extern_dfb_template_kernel(inp)
