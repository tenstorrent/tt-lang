# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Compile-only coverage for the Python-visible physical DFB limit error.

The nested conditional keeps every logical DFB lifetime unbounded. Recursive
composition therefore requires 33 distinct physical DFB indices.
"""

# CHECK: DFB allocation needs 33 physical indices but hardware supports at most 32

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402


COMPOSITION_LEVELS = 5


def _host_ttnn():
    return ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _make_nested_conditional_copy():
    @ttl.operation()
    def conditional_copy(source: ttl.DFB, destination: ttl.DFB):
        node_x, _ = ttl.node(dims=2)
        if node_x == 0:
            source_block = source.wait()
            destination_block = destination.reserve()
            destination_block.store(source_block)

    nested_copy = conditional_copy
    for composition_level in range(COMPOSITION_LEVELS):
        inner_copy = nested_copy

        @ttl.operation()
        def doubled_copy(source: ttl.DFB, destination: ttl.DFB):
            intermediate_dfb = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            inner_copy(source, intermediate_dfb)
            inner_copy(intermediate_dfb, destination)

        nested_copy = doubled_copy

    return nested_copy


_nested_conditional_copy = _make_nested_conditional_copy()


@ttl.operation(grid=(1, 1))
def over_capacity(input_tensor, output_tensor):
    input_dfb, output_dfb = [
        ttl.make_dfb("bf16", shape=(1, 1), block_count=2) for endpoint_index in range(2)
    ]

    ttl.copy(input_tensor[0, 0], input_dfb.reserve()).wait()
    _nested_conditional_copy(input_dfb, output_dfb)
    ttl.copy(output_dfb.wait(), output_tensor[0, 0]).wait()


def main():
    over_capacity(_host_ttnn(), _host_ttnn())


if __name__ == "__main__":
    main()
