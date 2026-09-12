# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.final.mlir TTLANG_COMPILER_OPTIONS=--ttl-specialize-cores %python %s > %t.specialized.output 2>&1
# RUN: FileCheck %s --check-prefix=SPECIALIZED < %t.final.mlir

"""Compile logical-kernel selectors with every external-call argument class."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import KernelKind, call_extern_func
from ttl import KernelKind as KK

FAKE_HEADER = "/dev/null/fake_shim.hpp"
reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)


@ttl.operation(grid=(2, 1))
def selected_external_calls(inp):
    descriptor_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    drain_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    ttl.call_extern_func(
        FAKE_HEADER,
        "compute_entry",
        template_args=[ttl.dfb_descriptor(descriptor_dfb), -3, True],
        func_args=[ttl.raw_addr(inp), 7],
        dfb_dependencies=[drain_dfb],
        dfb_effects=[
            ttl.DFBEffect.reserve(drain_dfb, tiles=1),
            ttl.DFBEffect.push(drain_dfb, tiles=1),
        ],
        include_paths=["/tmp"],
        kernel=ttl.KernelKind.COMPUTE,
    )
    core_x, _ = ttl.node(dims=2)
    if core_x == 0:
        call_extern_func(
            FAKE_HEADER,
            "reader_entry",
            template_args=[5],
            func_args=[ttl.raw_addr(inp)],
            kernel=reader,
        )
    ttl.call_extern_func(
        FAKE_HEADER,
        "shared_entry",
        kernel=KernelKind.COMPUTE | KernelKind.DATA_MOVEMENT,
    )
    ttl.call_extern_func(FAKE_HEADER, "alias_entry", kernel=KK.COMPUTE)

    unused = drain_dfb.wait()
    unused.pop(kernel=reader)


# CHECK-DAG: ttl.opaque_call "compute_entry"
# CHECK-DAG: ttl.opaque_call "reader_entry"
# CHECK-DAG: ttl.opaque_call "shared_entry"
# CHECK-DAG: ttl.opaque_call "shared_entry"
# CHECK-DAG: ttl.opaque_call "alias_entry"
# CHECK-DAG: ttl.logical_kernel = #ttl.logical_kernel<kind = compute>
# CHECK-DAG: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation =

# CHECK-CPP-DAG: compute_entry<
# CHECK-CPP-DAG: reader_entry<5>
# CHECK-CPP-DAG: shared_entry()
# CHECK-CPP-DAG: shared_entry()
# CHECK-CPP-DAG: alias_entry()

# SPECIALIZED-COUNT-2: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation =


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    selected_external_calls(inp)
