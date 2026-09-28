# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject repeated DFB publications with no consumer or Pipe send."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


@ttl.operation(grid=(1, 1))
def copy_unbalanced_dfb_publication(input_tensor):
    scratch_dfb = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=1,
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def reader():
        for iteration in range(4):
            with scratch_dfb.reserve() as scratch_block:
                ttl.copy(input_tensor[0, iteration], scratch_block).wait()

    @ttl.datamovement()
    def writer():
        pass


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 4 * 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    # CHECK: error: logical DFB {{[0-9]+}} has capacity-unsafe producer and consumer transactions on core_x=0, core_y=0
    # CHECK: the producer pushes 4 block(s) per launch and the consumer pops 0, leaving 4 outstanding block(s) for capacity 1
    copy_unbalanced_dfb_publication(input_tensor)
