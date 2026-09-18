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

    def make_resource_factory(output_values, *, alternate=False):
        def make_resources(*, tensors, core_ranges, first_free_semaphore_id):
            assert len(tensors) == 1
            semaphore = ttnn.SemaphoreDescriptor(
                first_free_semaphore_id,
                core_ranges=core_ranges,
                initial_value=0,
            )
            invocation_defines = defines
            if alternate:
                invocation_defines += (ttl.KernelDefine("OUTPUT_ALTERNATE", "1"),)
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
                        defines=invocation_defines,
                    ),
                ),
                lifetimes=(semaphore,),
            )

        return make_resources

    default_resource_factory = make_resource_factory(default_output_values)

    @ttl.operation(grid=(2, 1), runtime_resource_factory=default_resource_factory)
    def runtime_resource_operation(out):
        output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement(kernel=runtime_kernel)
        def fill_output():
            core_x, _ = ttl.node(dims=2)
            # Separate branches force per-core clones for resource partitioning.
            if core_x == 0:
                ttl.call_extern_func(
                    RUNTIME_RESOURCE_HEADER,
                    "write_operation_runtime_value",
                    func_args=[output_dfb],
                )
            else:
                ttl.call_extern_func(
                    RUNTIME_RESOURCE_HEADER,
                    "write_operation_runtime_value",
                    func_args=[output_dfb],
                )

        @ttl.datamovement()
        def write_output():
            core_x, _ = ttl.node(dims=2)
            output_block = output_dfb.wait()
            ttl.copy(output_block, out[0, core_x]).wait()
            output_block.pop()

    return runtime_resource_operation, make_resource_factory


RUNTIME_RESOURCE_OPERATIONS = {
    dtype: _make_runtime_resource_operation(dtype)[0]
    for dtype in (torch.bfloat16, torch.float32)
}


def _to_test_memory_config(host_tensor, device, memory_config):
    device_tensor = to_dram(host_tensor, device)
    if memory_config == "dram":
        return device_tensor
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
@pytest.mark.parametrize(
    "memory_config",
    ["dram", "sharded_l1"],
    ids=["dram", "sharded-l1"],
)
def test_operation_runtime_resources_materialize_per_core_values(
    device, dtype, specialize_cores, memory_config
):
    """One logical kernel receives distinct values and one caller semaphore."""
    host_output = torch.zeros((32, 64), dtype=dtype)
    output = _to_test_memory_config(host_output, device, memory_config)

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
    compile_output = _to_test_memory_config(
        torch.zeros((32, 64), dtype=torch.bfloat16),
        device,
        "sharded_l1",
    )
    operation(compile_output, options="--ttl-specialize-cores")

    emitted_runner = runpy.run_path(str(runner_path))
    test_cases = (
        ((0x4040, 0x4080), False, (3.0, 4.0)),
        ((0x40A0, 0x40C0), False, (5.0, 6.0)),
        ((0x4110, 0x4120), True, (7.0, 8.0)),
    )
    for output_values, alternate, expected_values in test_cases:
        output = _to_test_memory_config(
            torch.zeros((32, 64), dtype=torch.bfloat16),
            device,
            "sharded_l1",
        )
        emitted_runner["run"](
            [output],
            runtime_resource_factory=make_resource_factory(
                output_values,
                alternate=alternate,
            ),
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
