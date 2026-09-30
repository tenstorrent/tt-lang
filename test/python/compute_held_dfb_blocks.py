# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s unused-waits 2>&1 | FileCheck %s --check-prefix=UNUSED-WAITS
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s unused-reserves 2>&1 | FileCheck %s --check-prefix=UNUSED-RESERVES

# UNUSED-WAITS: int32_t [[TWO:v[0-9]+]] = 2;
# UNUSED-WAITS: .wait_front([[TWO]]);
# UNUSED-WAITS-NEXT: .pop_front([[TWO]]);
# UNUSED-WAITS-NOT: pop_front
# UNUSED-WAITS: COMPILED
# UNUSED-RESERVES: int32_t [[TWO:v[0-9]+]] = 2;
# UNUSED-RESERVES: .reserve_back([[TWO]]);
# UNUSED-RESERVES-NEXT: .push_back([[TWO]]);
# UNUSED-RESERVES-NOT: push_back
# UNUSED-RESERVES: COMPILED

"""Accept a compute kernel that holds several blocks of one DFB without using
them.

`ttl-coalesce-dfb-acquires` merges the acquisitions into one multi-block
acquisition, and the stated releases become its one release, so no release
is inserted for the unused blocks.
"""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target
MODE = sys.argv[1]


def make_unused_waits():
    @ttl.operation(grid=(1, 1))
    def unused_waits(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            first = dfb.wait()
            second = dfb.wait()
            first.pop()
            second.pop()

        @ttl.datamovement()
        def reader():
            for column in range(2):
                with dfb.reserve() as blk:
                    ttl.copy(inp[0, column], blk).wait()

        @ttl.datamovement()
        def writer():
            pass

    return unused_waits


def make_unused_reserves():
    @ttl.operation(grid=(1, 1))
    def unused_reserves(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            first = dfb.reserve()
            second = dfb.reserve()
            first.push()
            second.push()

        @ttl.datamovement()
        def reader():
            pass

        @ttl.datamovement()
        def writer():
            for column in range(2):
                with dfb.wait() as blk:
                    ttl.copy(blk, out[0, column]).wait()

    return unused_reserves


FACTORIES = {
    "unused-waits": make_unused_waits,
    "unused-reserves": make_unused_reserves,
}
operation = FACTORIES[MODE]()

tensors = [
    ttnn.from_torch(
        torch.zeros((32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(2)
]
operation(*tensors)
print("COMPILED")
