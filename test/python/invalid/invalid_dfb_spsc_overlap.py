# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Compile-only coverage for DFB SPSC rejection in frontend-generated IR.

The program creates one DFB consumed by both a compute thread and a data
movement thread over the full launch grid. The verifier must reject the shared
DFB because the consumer launch-node domains overlap.
"""

# CHECK: dataflow buffer cb_index={{[0-9]+}} has multiple consumer threads active on the same launched node
# CHECK: tt-metal CBs are single-producer single-consumer; allocate one DFB per consumer

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402


def _host_ttnn(tensor_shape):
    return ttnn.from_torch(
        torch.zeros(tensor_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@ttl.operation(grid=(2, 1))
def overlapping_dfb_consumers(input_tensor, output_tensor):
    shared_cb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)
    scratch_cb = ttl.make_dataflow_buffer_like(
        output_tensor, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute_consumer():
        with shared_cb.wait() as shared_blk, scratch_cb.reserve() as scratch_blk:
            scratch_blk.store(shared_blk)

    @ttl.datamovement()
    def data_movement_producer():
        with shared_cb.reserve() as shared_blk:
            ttl.copy(input_tensor[0, 0], shared_blk).wait()

    @ttl.datamovement()
    def data_movement_consumer():
        with shared_cb.wait() as shared_blk:
            ttl.copy(shared_blk, output_tensor[0, 0]).wait()


def main():
    input_tensor = _host_ttnn((32, 32))
    output_tensor = _host_ttnn((32, 32))
    overlapping_dfb_consumers(input_tensor, output_tensor)


if __name__ == "__main__":
    main()
