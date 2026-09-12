# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Verify expansion of captured static external template arguments."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

FAKE_HEADER = "/dev/null/fake_shim.hpp"


def make_extern_template_expansion_kernel():
    static_template_args = (17, False)

    @ttl.operation(grid=(1, 1))
    def extern_template_expansion_kernel(inp):
        descriptor = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            ttl.call_extern_func(
                FAKE_HEADER,
                "my_shim",
                template_args=[
                    ttl.dfb_descriptor(descriptor),
                    *static_template_args,
                ],
            )

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            pass

    return extern_template_expansion_kernel


extern_template_expansion_kernel = make_extern_template_expansion_kernel()


# CHECK: ttl.opaque_call "my_shim" template_args [#ttl.external_template_arg<dfb_descriptor, 0>, #ttl.external_template_arg<signed_integer, 17>, #ttl.external_template_arg<boolean, 0>]
# CHECK-CPP: my_shim<ttlang::DFBDescriptor<0, 1, 1, 2048>, 17, false>();


if __name__ == "__main__":
    host = torch.ones((32, 32), dtype=torch.bfloat16)
    inp = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    extern_template_expansion_kernel(inp)
