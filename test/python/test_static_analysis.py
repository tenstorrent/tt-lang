# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the compiler static-analysis entry point."""

import pytest
import torch

import ttl
import ttl.ttl_api as ttl_api
from ttl.diagnostics import TTLangCompileError
from ttl.dtype_utils import is_tensor_value
from ttl.static_analysis import StaticTensorSpec, build_operation_validator


def add_operation(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def add_compute():
        lhs_block = lhs_dfb.wait()
        rhs_block = rhs_dfb.wait()
        out_block = out_dfb.reserve()
        out_block.store(lhs_block + rhs_block)
        lhs_block.pop()
        rhs_block.pop()
        out_block.push()

    @ttl.datamovement()
    def read_inputs():
        lhs_block = lhs_dfb.reserve()
        lhs_copy = ttl.copy(lhs[0, 0], lhs_block)
        lhs_copy.wait()
        lhs_block.push()

        rhs_block = rhs_dfb.reserve()
        rhs_copy = ttl.copy(rhs[0, 0], rhs_block)
        rhs_copy.wait()
        rhs_block.push()

    @ttl.datamovement()
    def write_output():
        out_block = out_dfb.wait()
        out_copy = ttl.copy(out_block, out[0, 0])
        out_copy.wait()
        out_block.pop()


def unified_add_operation(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    lhs_block = lhs_dfb.reserve()
    ttl.copy(lhs[0:1, 0:1], lhs_block)
    rhs_block = rhs_dfb.reserve()
    ttl.copy(rhs[0:1, 0:1], rhs_block)

    result = out_dfb.reserve()
    lhs_ready = lhs_dfb.wait()
    rhs_ready = rhs_dfb.wait()
    result.store(lhs_ready + rhs_ready)

    output = out_dfb.wait()
    ttl.copy(output, out[0:1, 0:1])


def _spec(dtype=torch.bfloat16):
    return StaticTensorSpec((32, 32), dtype)


def test_static_tensor_recognition_does_not_accept_torch_runtime_tensors():
    assert is_tensor_value(_spec())
    assert not is_tensor_value(torch.empty((32, 32)))


def test_valid_operation_passes_without_runtime_compilation(monkeypatch):
    def reject_runtime_compilation(*_args, **_kwargs):
        raise AssertionError("runtime compilation must not run")

    monkeypatch.setattr(ttl_api, "_compile_ttnn_kernel", reject_runtime_compilation)
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="blackhole"
    )

    assert validator(_spec(), _spec(), _spec()) is None


def test_valid_unified_operation_uses_same_host_only_pipeline(monkeypatch):
    def reject_runtime_compilation(*_args, **_kwargs):
        raise AssertionError("runtime compilation must not run")

    monkeypatch.setattr(ttl_api, "_compile_ttnn_kernel", reject_runtime_compilation)
    validator = build_operation_validator(
        unified_add_operation, grid=(1, 1), target_arch="blackhole"
    )

    assert validator(_spec(), _spec(), _spec()) is None


def test_successful_analysis_is_cached_by_tensor_signature(monkeypatch):
    construction_calls = 0
    original_construct = ttl_api._construct_ttl_program

    def count_constructions(*args, **kwargs):
        nonlocal construction_calls
        construction_calls += 1
        return original_construct(*args, **kwargs)

    monkeypatch.setattr(ttl_api, "_construct_ttl_program", count_constructions)
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="blackhole"
    )

    validator(_spec(), _spec(), _spec())
    validator(_spec(), _spec(), _spec())
    assert construction_calls == 1


def test_analysis_cache_distinguishes_dtypes(monkeypatch):
    construction_calls = 0
    original_construct = ttl_api._construct_ttl_program

    def count_constructions(*args, **kwargs):
        nonlocal construction_calls
        construction_calls += 1
        return original_construct(*args, **kwargs)

    monkeypatch.setattr(ttl_api, "_construct_ttl_program", count_constructions)
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="wormhole_b0"
    )

    validator(_spec(), _spec(), _spec())
    with pytest.raises((TTLangCompileError, RuntimeError, TypeError, ValueError)):
        validator(_spec(torch.float32), _spec(), _spec())
    assert construction_calls == 2


def test_validation_pipeline_excludes_runtime_lowering():
    compiler_options = ttl_api.CompilerOptions()
    validation_passes = ttl_api._validation_pipeline_passes(
        fp32_dest_acc_en=None,
        dst_full_sync_en=None,
        compiler_options=compiler_options,
        l1_budget_override=0,
    )
    runtime_passes = ttl_api._runtime_lowering_pipeline_passes(compiler_options, 0)

    assert "ttl-validate-cb-budget" in validation_passes
    assert not any("convert-ttl-to-ttkernel" in item for item in validation_passes)
    assert any("convert-ttl-to-ttkernel" in item for item in runtime_passes)
    assert not any("ttl-validate-cb-budget" in item for item in runtime_passes)


def test_grid_size_one_dimensional_is_rejected():
    def invalid_grid_size(lhs, rhs, out):
        ttl.grid_size(dims=1)

        @ttl.compute()
        def compute():
            pass

    validator = build_operation_validator(
        invalid_grid_size, grid=(1, 1), target_arch="blackhole"
    )

    with pytest.raises(ValueError, match=r"grid_size\(\).*dims=1"):
        validator(_spec(), _spec(), _spec())


def test_unsupported_tuple_capture_is_rejected():
    capture = (1, 2)

    def invalid_capture(lhs, rhs, out):
        lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1))

        @ttl.compute()
        def compute():
            _ = capture[0]
            lhs_dfb.wait()

    validator = build_operation_validator(
        invalid_capture, grid=(1, 1), target_arch="blackhole"
    )

    with pytest.raises(TypeError, match="Unhandled capture.*tuple"):
        validator(_spec(), _spec(), _spec())


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((32,), torch.bfloat16),
        ((0, 32), torch.bfloat16),
        ((32, 32), torch.int64),
    ],
)
def test_invalid_tensor_descriptors_are_rejected(shape, dtype):
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="blackhole"
    )

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        validator(
            StaticTensorSpec(shape, dtype),
            StaticTensorSpec(shape, dtype),
            StaticTensorSpec(shape, dtype),
        )


def test_invalid_target_architecture_is_rejected():
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="imaginary"
    )

    with pytest.raises(ValueError, match="unsupported target architecture"):
        validator(_spec(), _spec(), _spec())
