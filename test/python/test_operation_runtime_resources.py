# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests for per-invocation operation runtime resources."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose


RUNTIME_RESOURCE_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "operation_runtime_resources.hpp"
)
CORE_RANGES = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}
)


def _make_runtime_resource_operation(dtype):
    runtime_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    output_values = {
        torch.bfloat16: (0x3F80, 0x4000),
        torch.float32: (0x3F800000, 0x40000000),
    }[dtype]
    defines = (ttl.KernelDefine("OUTPUT_BF16", "1"),) if dtype == torch.bfloat16 else ()

    def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
        assert len(tensors) == 1
        semaphore = ttnn.SemaphoreDescriptor(
            first_free_semaphore_id,
            core_ranges=core_ranges,
            initial_value=0,
        )
        return ttl.ProgramRuntimeResources(
            semaphore_descriptors=(semaphore,),
            kernel_resources=(
                ttl.KernelRuntimeResources(
                    kernel=runtime_kernel,
                    runtime_args=(
                        ttl.CoreRuntimeArgs(
                            ttnn.CoreCoord(0, 0),
                            (output_values[0], first_free_semaphore_id, 1),
                        ),
                        ttl.CoreRuntimeArgs(
                            ttnn.CoreCoord(1, 0),
                            (output_values[1], first_free_semaphore_id, 1),
                        ),
                    ),
                    defines=defines,
                ),
            ),
            lifetimes=(semaphore,),
        )

    @ttl.operation(grid=(2, 1), runtime_resource_factory=make_resources)
    def runtime_resource_operation(out):
        ttl.call_extern_func(
            RUNTIME_RESOURCE_HEADER,
            "write_operation_runtime_value",
            func_args=[ttl.raw_addr(out)],
            kernel=runtime_kernel,
        )

    return runtime_resource_operation


RUNTIME_RESOURCE_OPERATIONS = {
    torch.bfloat16: _make_runtime_resource_operation(torch.bfloat16),
    torch.float32: _make_runtime_resource_operation(torch.float32),
}


def _to_two_core_width_sharded(host_tensor, device):
    device_tensor = to_dram(host_tensor, device)
    shard_spec = ttnn.ShardSpec(
        CORE_RANGES,
        (32, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.to_memory_config(device_tensor, memory_config=memory_config)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "f32"],
)
def test_operation_runtime_resources_materialize_per_core_values(device, dtype):
    """One logical kernel receives distinct values and one caller semaphore."""
    host_output = torch.zeros((32, 64), dtype=dtype)
    output = _to_two_core_width_sharded(host_output, device)

    RUNTIME_RESOURCE_OPERATIONS[dtype](output)

    expected = torch.cat(
        (
            torch.full((32, 32), 1.0, dtype=dtype),
            torch.full((32, 32), 2.0, dtype=dtype),
        ),
        dim=1,
    )
    assert_allclose(ttnn.to_torch(output).float(), expected.float())
