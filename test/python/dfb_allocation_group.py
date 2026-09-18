# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.mlir %python %s
# RUN: FileCheck %s < %t.mlir

"""Verify DFB allocation-group identity through composition and splitting."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


def make_allocation_group_operation():
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation(grid=(1, 1))
    def allocation_group_operation(input_tensor, output_tensor):
        shared_allocation = ttl.make_dfb_allocation_group()
        first_source = ttl.make_dataflow_buffer_like(
            input_tensor,
            shape=(1, 1),
            block_count=2,
            allocation_group=shared_allocation,
        )
        handoff = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
        second_source = ttl.make_dataflow_buffer_like(
            input_tensor,
            shape=(1, 1),
            block_count=4,
            allocation_group=shared_allocation,
        )
        output = ttl.make_dataflow_buffer_like(
            output_tensor, shape=(1, 1), block_count=2
        )

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with first_source.reserve() as destination:
                ttl.copy(input_tensor[0, 0], destination).wait()

            with handoff.wait():
                pass

            with second_source.reserve() as destination:
                ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with first_source.wait():
                pass

            with handoff.reserve() as signal:
                signal.store(ttl.block.fill(0, shape=signal.shape, dtype=signal.dtype))

            with second_source.wait() as source:
                with output.reserve() as destination:
                    destination.store(source)

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with output.wait() as source:
                ttl.copy(source, output_tensor[0, 0]).wait()

    return allocation_group_operation


allocation_group_operation = make_allocation_group_operation()


# CHECK-LABEL: func.func @read
# CHECK: ttl.bind_cb{{.*}}allocation_group = #ttl.dfb_allocation_group<0>{{.*}}dfb_id = 0 : index
# CHECK: ttl.bind_cb{{.*}}allocation_group = #ttl.dfb_allocation_group<0>{{.*}}dfb_id = 2 : index
# CHECK-LABEL: func.func @compute
# CHECK: ttl.bind_cb{{.*}}allocation_group = #ttl.dfb_allocation_group<0>{{.*}}dfb_id = 0 : index
# CHECK: ttl.bind_cb{{.*}}allocation_group = #ttl.dfb_allocation_group<0>{{.*}}dfb_id = 2 : index


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    allocation_group_operation(input_tensor, output_tensor)
