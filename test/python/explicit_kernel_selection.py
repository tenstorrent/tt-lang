# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.final.mlir TTLANG_COMPILER_OPTIONS=--ttl-specialize-cores %python %s > %t.specialized.output 2>&1
# RUN: FileCheck %s --check-prefix=SPECIALIZED < %t.final.mlir

"""Compile explicit thread decorators with named logical kernel selectors."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)


@ttl.operation(grid=(2, 1))
def explicit_selected_add(inp, out):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute(kernel=compute_kernel)
    def compute_thread():
        core_x, _ = ttl.node(dims=2)
        input_block = input_dfb.wait()
        output_block = output_dfb.reserve()
        if core_x == 0:
            output_block.store(input_block + input_block)
        else:
            output_block.store(input_block)
        input_block.pop(kernel=compute_kernel)
        output_block.push(kernel=compute_kernel)

    @ttl.datamovement(kernel=reader_kernel)
    def reader_thread():
        input_block = input_dfb.reserve()
        ttl.copy(inp[0, 0], input_block).wait()
        input_block.push(kernel=reader_kernel)

    @ttl.datamovement()
    def writer_thread():
        output_block = output_dfb.wait()
        ttl.copy(output_block, out[0, 0]).wait()
        output_block.pop(kernel=ttl.KernelKind.DATA_MOVEMENT)


# CHECK-LABEL: func.func @compute_thread
# CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute_kernel", operation = "__main__.explicit_selected_add[captures={{[0-9a-f]+}}]">
# CHECK-LABEL: func.func @reader_thread
# CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader_kernel", operation = "__main__.explicit_selected_add[captures={{[0-9a-f]+}}]">
# CHECK-LABEL: func.func @writer_thread
# CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>

# SPECIALIZED-LABEL: func.func @compute_thread_c0_0
# SPECIALIZED-SAME: ttl.core_coord = {{\[\[}}0, 0]]
# SPECIALIZED-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute_kernel", operation = "__main__.explicit_selected_add[captures={{[0-9a-f]+}}]">
# SPECIALIZED-LABEL: func.func @compute_thread_c1_0
# SPECIALIZED-SAME: ttl.core_coord = {{\[\[}}1, 0]]
# SPECIALIZED-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute_kernel", operation = "__main__.explicit_selected_add[captures={{[0-9a-f]+}}]">
# SPECIALIZED: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader_kernel", operation = "__main__.explicit_selected_add[captures={{[0-9a-f]+}}]">
# SPECIALIZED: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    out = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    explicit_selected_add(inp, out)
