# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests for per-invocation operation runtime resources."""

import os
import runpy

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
    default_output_values = {
        torch.bfloat16: (0x3F80, 0x4000),
        torch.float32: (0x3F800000, 0x40000000),
    }[dtype]
    defines = (ttl.KernelDefine("OUTPUT_BF16", "1"),) if dtype == torch.bfloat16 else ()

    def make_resource_factory(output_values):
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

        return make_resources

    default_resource_factory = make_resource_factory(default_output_values)

    @ttl.operation(grid=(2, 1), runtime_resource_factory=default_resource_factory)
    def runtime_resource_operation(out):
        core_x, _ = ttl.node(dims=2)
        # Separate branches force per-core clones that exercise resource partitioning.
        if core_x == 0:
            ttl.call_extern_func(
                RUNTIME_RESOURCE_HEADER,
                "write_operation_runtime_value",
                func_args=[ttl.raw_addr(out)],
                kernel=runtime_kernel,
            )
        else:
            ttl.call_extern_func(
                RUNTIME_RESOURCE_HEADER,
                "write_operation_runtime_value",
                func_args=[ttl.raw_addr(out)],
                kernel=runtime_kernel,
            )

    return runtime_resource_operation, make_resource_factory


RUNTIME_RESOURCE_OPERATIONS = {
    dtype: _make_runtime_resource_operation(dtype)[0]
    for dtype in (torch.bfloat16, torch.float32)
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
@pytest.mark.parametrize(
    "specialize_cores",
    [False, True],
    ids=["generic-cores", "specialized-cores"],
)
def test_operation_runtime_resources_materialize_per_core_values(
    device, dtype, specialize_cores
):
    """One logical kernel receives distinct values and one caller semaphore."""
    host_output = torch.zeros((32, 64), dtype=dtype)
    output = _to_two_core_width_sharded(host_output, device)

    options = (
        "--ttl-specialize-cores" if specialize_cores else "--no-ttl-specialize-cores"
    )
    RUNTIME_RESOURCE_OPERATIONS[dtype](output, options=options)

    expected = torch.cat(
        (
            torch.full((32, 32), 1.0, dtype=dtype),
            torch.full((32, 32), 2.0, dtype=dtype),
        ),
        dim=1,
    )
    assert_allclose(ttnn.to_torch(output).float(), expected.float())


def test_emitted_runtime_resource_runner_updates_invocation_values(
    device, monkeypatch, tmp_path
):
    """The emitted specialized runner accepts a factory on every invocation."""
    runner_path = tmp_path / "operation_runtime_resources_runner.py"
    monkeypatch.setenv("TTLANG_EMIT_RUNNER", str(runner_path))
    operation, make_resource_factory = _make_runtime_resource_operation(torch.bfloat16)
    compile_output = _to_two_core_width_sharded(
        torch.zeros((32, 64), dtype=torch.bfloat16),
        device,
    )
    operation(compile_output, options="--ttl-specialize-cores")

    emitted_runner = runpy.run_path(str(runner_path))
    test_cases = (
        ((0x4040, 0x4080), (3.0, 4.0)),
        ((0x40A0, 0x40C0), (5.0, 6.0)),
    )
    for output_values, expected_values in test_cases:
        output = _to_two_core_width_sharded(
            torch.zeros((32, 64), dtype=torch.bfloat16),
            device,
        )
        emitted_runner["run"](
            [output],
            runtime_resource_factory=make_resource_factory(output_values),
            device=device,
        )
        expected = torch.cat(
            (
                torch.full((32, 32), expected_values[0], dtype=torch.bfloat16),
                torch.full((32, 32), expected_values[1], dtype=torch.bfloat16),
            ),
            dim=1,
        )
        assert_allclose(ttnn.to_torch(output).float(), expected.float())
