# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Reuse physical CBs across cyclic BF16/F32 phase boundaries."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_l1_sharded

TILE_COUNT = 16
ITERATIONS = 64
PROGRAM_SEMAPHORE_COUNT = 16


def make_reset_dataflow_buffers_typecast(tile_count, iterations):
    def runtime_resource_factory(*, tensors, core_ranges, first_free_semaphore_id):
        del tensors
        first = int(first_free_semaphore_id)
        if first != 0:
            raise ValueError("DFB reset must not consume local semaphore IDs")
        return ttl.ProgramRuntimeResources(
            semaphore_descriptors=[
                ttnn.SemaphoreDescriptor(
                    semaphore_id,
                    core_ranges=core_ranges,
                    initial_value=0,
                )
                for semaphore_id in range(PROGRAM_SEMAPHORE_COUNT)
            ]
        )

    @ttl.operation(grid=(1, 1), runtime_resource_factory=runtime_resource_factory)
    def operation(bf16_in, f32_out, f32_in, bf16_out):
        first_in = ttl.make_dataflow_buffer_like(bf16_in, shape=(1, 1), block_count=2)
        first_out = ttl.make_dataflow_buffer_like(f32_out, shape=(1, 1), block_count=2)
        second_in = ttl.make_dataflow_buffer_like(f32_in, shape=(1, 1), block_count=2)
        second_out = ttl.make_dataflow_buffer_like(
            bf16_out, shape=(1, 1), block_count=2
        )

        for _ in range(iterations):
            for tile_index in range(tile_count):
                first_read = first_in.reserve()
                ttl.copy(bf16_in[tile_index : tile_index + 1, 0:1], first_read).wait()
                first_write = first_out.reserve()
                first_write.store(ttl.math.typecast(first_in.wait(), torch.float32))
                ttl.copy(
                    first_out.wait(),
                    f32_out[tile_index : tile_index + 1, 0:1],
                ).wait()

            ttl.reset_dataflow_buffers()

            for tile_index in range(tile_count):
                second_read = second_in.reserve()
                ttl.copy(f32_in[tile_index : tile_index + 1, 0:1], second_read).wait()
                second_write = second_out.reserve()
                second_write.store(ttl.math.typecast(second_in.wait(), torch.bfloat16))
                ttl.copy(
                    second_out.wait(),
                    bf16_out[tile_index : tile_index + 1, 0:1],
                ).wait()

            ttl.reset_dataflow_buffers()

    return operation


reset_dataflow_buffers_typecast = make_reset_dataflow_buffers_typecast(
    TILE_COUNT, ITERATIONS
)


def test_reset_dataflow_buffers_typecast(device):
    shape = (TILE_COUNT * ttnn.TILE_SIZE, ttnn.TILE_SIZE)
    bf16_input = torch.randn(shape, dtype=torch.bfloat16)
    f32_input = torch.randn(shape, dtype=torch.float32)
    bf16_in = to_l1_sharded(bf16_input, device)
    f32_out = to_l1_sharded(torch.zeros(shape, dtype=torch.float32), device)
    f32_in = to_l1_sharded(f32_input, device)
    bf16_out = to_l1_sharded(torch.zeros(shape, dtype=torch.bfloat16), device)

    reset_dataflow_buffers_typecast(bf16_in, f32_out, f32_in, bf16_out)

    torch.testing.assert_close(
        ttnn.to_torch(f32_out), bf16_input.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        ttnn.to_torch(bf16_out), f32_input.bfloat16(), rtol=0, atol=0
    )
