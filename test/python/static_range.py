# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""Compile-only coverage for explicitly unrolled TT-Lang loops."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttnn  # noqa: E402
import ttl  # noqa: E402
from ttl import static_range  # noqa: E402


@ttl.operation(grid=(1, 1))
def static_range_copy(input_tensor, output_tensor):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def static_range_compute():
        for iteration_index in ttl.static_range(1):
            with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
                output_block.store(input_block)
        for iteration_index in static_range(1, 2):
            with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
                output_block.store(input_block)
        for iteration_index in ttl.static_range(3, 2, -1):
            with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
                output_block.store(input_block)

    @ttl.datamovement()
    def read_input():
        for iteration_index in range(3):
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 0], input_block).wait()

    @ttl.datamovement()
    def write_output():
        for iteration_index in range(3):
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()


def host_tensor():
    return ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


if __name__ == "__main__":
    static_range_copy(host_tensor(), host_tensor())


# CHECK-LABEL: func.func @static_range_compute
# CHECK-NOT: scf.for
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve
# CHECK: ttl.store
# CHECK: ttl.cb_push
# CHECK: ttl.cb_pop
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve
# CHECK: ttl.store
# CHECK: ttl.cb_push
# CHECK: ttl.cb_pop
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve
# CHECK: ttl.store
# CHECK: ttl.cb_push
# CHECK: ttl.cb_pop
# CHECK-NOT: scf.for
# CHECK: return
