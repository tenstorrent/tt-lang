# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s 2>&1 | FileCheck %s

"""Verify invalid typed DFB allocation-group uses."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


GLOBAL_ALLOCATION_GROUP = ttl.make_dfb_allocation_group()


@ttl.operation(grid=(1, 1))
def invalid_operation(input_tensor):
    target = ttl.make_dfb(
        "bf16",
        shape=(1, 1),
        block_count=2,
        allocation_group=GLOBAL_ALLOCATION_GROUP,
    )

    @ttl.compute()
    def compute():
        with target.wait():
            pass

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


# CHECK: ValueError: @ttl.operation 'invalid_operation': DFBAllocationGroup 'GLOBAL_ALLOCATION_GROUP' must be created by an enclosing factory


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_operation(input_tensor)
