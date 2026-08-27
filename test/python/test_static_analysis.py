# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the compiler static-analysis entry point."""

from unittest.mock import MagicMock

import pytest
import torch

import ttl
import ttl.ttl_api as ttl_api
from ttl.dataflow_buffer import make_tensor_backed_dfb
from ttl.diagnostics import TTLangCompileError
from ttl.dtype_utils import is_tensor_value
from ttl.layouts import (
    TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED,
    detect_memory_layout,
)
from ttl.static_analysis import (
    COMPILER_VALIDATION_API_VERSION,
    StaticTensorSpec,
    build_operation_validator,
)


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


def copy_operation_with_unused_tensor(unused, input_tensor, output_tensor):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read_input():
        block = input_dfb.reserve()
        copy = ttl.copy(input_tensor[0, 0], block)
        copy.wait()
        block.push()

    @ttl.datamovement()
    def write_output():
        block = input_dfb.wait()
        copy = ttl.copy(block, output_tensor[0, 0])
        copy.wait()
        block.pop()


def _spec(dtype=torch.bfloat16):
    return StaticTensorSpec((32, 32), dtype)


def test_compiler_validation_api_has_explicit_compatibility_version():
    assert COMPILER_VALIDATION_API_VERSION == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"shape": (0, 32)}, "shape dimensions must be positive"),
        (
            {"shape": (32, 32), "padded_shape": (32,)},
            "padded_shape must have the same rank",
        ),
        (
            {"shape": (32, 32), "padded_shape": (16, 32)},
            "padded_shape.*cannot be smaller",
        ),
        ({"shape": (32, 32), "layout": "TIEL"}, "layout must be one of"),
        (
            {"shape": (32, 32), "memory_space": "HOST"},
            "memory_space must be one of",
        ),
        (
            {"shape": (32, 32), "memory_layout": "MYSTERY"},
            "memory_layout must be one of",
        ),
        (
            {"shape": (32, 32), "layout": "ROW_MAJOR"},
            "ROW_MAJOR layout must not include tile_shape or tile_size_bytes",
        ),
        (
            {"shape": (32, 32), "tile_shape": (32,)},
            "requires a two-dimensional tile_shape",
        ),
        (
            {"shape": (32, 32), "tile_size_bytes": 0},
            "tile_size_bytes must be positive",
        ),
        (
            {"shape": (32, 32), "memory_layout": "HEIGHT_SHARDED"},
            "requires regular shard_shape metadata",
        ),
        (
            {"shape": (32, 32), "memory_layout": "ND_SHARDED"},
            "requires nd_shard_shape metadata",
        ),
        (
            {"shape": (32, 32), "shard_shape": (32, 32)},
            "INTERLEAVED cannot include regular shard metadata",
        ),
        (
            {"shape": (32, 32), "shard_orientation": "ROW_MAJOR"},
            "INTERLEAVED cannot include regular shard metadata",
        ),
        (
            {
                "shape": (32, 32),
                "memory_layout": "ND_SHARDED",
                "nd_shard_shape": (32, 32),
                "nd_shard_num_cores": 0,
            },
            "nd_shard_num_cores must be positive",
        ),
        (
            {
                "shape": (32, 32),
                "memory_layout": "HEIGHT_SHARDED",
                "shard_shape": (32, 32),
                "shard_orientation": "DIAGONAL",
            },
            "shard_orientation must be one of",
        ),
        (
            {
                "shape": (32, 32),
                "memory_layout": "ND_SHARDED",
                "nd_shard_shape": (32, 32),
                "nd_shard_distribution": "RANDOM",
            },
            "nd_shard_distribution must be one of",
        ),
        (
            {
                "shape": (32, 32),
                "mesh_shape": (1, 2),
                "mesh_dims": (0,),
            },
            "mesh_dims requires mesh_shape with the same rank",
        ),
        (
            {
                "shape": (32, 32),
                "memory_layout": "HEIGHT_SHARDED",
                "shard_shape": (32, 32),
                "shard_core_ranges": ((1, 0, 0, 0),),
            },
            "invalid core range",
        ),
    ],
)
def test_static_tensor_descriptor_rejects_inconsistent_metadata(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StaticTensorSpec(dtype=torch.bfloat16, **kwargs)


def test_static_tensor_recognition_does_not_accept_torch_runtime_tensors():
    assert is_tensor_value(_spec())
    assert not is_tensor_value(torch.empty((32, 32)))


def test_static_tensor_recognition_requires_an_explicit_marker():
    assert not is_tensor_value(MagicMock())


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


def test_static_tensor_cache_identity_includes_complete_metadata():
    base = StaticTensorSpec(
        (64, 64),
        torch.bfloat16,
        padded_shape=(64, 64),
        layout="TILE",
        memory_space="L1",
        memory_layout="HEIGHT_SHARDED",
        tile_shape=(32, 32),
        tile_size_bytes=2048,
        shard_shape=(32, 64),
        shard_grid=(2, 1),
        shard_orientation="ROW_MAJOR",
        shard_core_ranges=((0, 0, 1, 0),),
        mesh_shape=(1, 2),
        mesh_dims=(None, 0),
    )
    changed = StaticTensorSpec(
        (64, 64),
        torch.bfloat16,
        padded_shape=(64, 64),
        layout="TILE",
        memory_space="L1",
        memory_layout="WIDTH_SHARDED",
        tile_shape=(32, 32),
        tile_size_bytes=2048,
        shard_shape=(64, 32),
        shard_grid=(1, 2),
        shard_orientation="COL_MAJOR",
        shard_core_ranges=((0, 0, 0, 1),),
        mesh_shape=(2, 1),
        mesh_dims=(1, None),
    )

    common = {
        "resolved_grid": (1, 1),
        "fp32_dest_acc_en": None,
        "dst_full_sync_en": None,
        "math_fidelity": None,
        "target_arch": "blackhole",
    }
    assert ttl_api._make_cache_key((base,), **common) != ttl_api._make_cache_key(
        (changed,), **common
    )


def test_static_tensor_layout_and_shard_metadata_reach_compiler_helpers():
    spec = StaticTensorSpec(
        (64, 64),
        torch.bfloat16,
        memory_space="L1",
        memory_layout="HEIGHT_SHARDED",
        tile_size_bytes=2048,
        shard_shape=(32, 64),
        shard_grid=(2, 1),
    )

    assert detect_memory_layout(spec) == TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED
    assert ttl_api._detect_memory_space_from_tensor(spec, "DRAM") == "L1"
    assert spec.get_tile().tile_shape == (32, 32)
    assert spec.memory_config().shard_spec.shape == (32, 64)


def test_sharded_static_tensor_reaches_validation_pipeline():
    specs = tuple(
        StaticTensorSpec(
            (32, 32),
            torch.bfloat16,
            memory_space="L1",
            memory_layout="HEIGHT_SHARDED",
            tile_size_bytes=2048,
            shard_shape=(32, 32),
            shard_grid=(1, 1),
        )
        for _ in range(3)
    )
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="blackhole"
    )

    assert validator(*specs) is None


def test_unsupported_static_memory_layout_is_not_treated_as_interleaved():
    spec = StaticTensorSpec(
        (64, 64),
        torch.bfloat16,
        memory_layout="ND_SHARDED",
        nd_shard_shape=(32, 32),
        nd_shard_grid=(2, 2),
    )

    with pytest.raises(ValueError, match="Unsupported tensor memory layout"):
        detect_memory_layout(spec)


def test_tensor_backed_dfb_accepts_complete_static_tensor_metadata():
    spec = StaticTensorSpec(
        (64, 64),
        torch.bfloat16,
        memory_space="L1",
        memory_layout="HEIGHT_SHARDED",
        tile_size_bytes=2048,
        shard_shape=(64, 64),
        shard_grid=(1, 1),
    )

    dfb = make_tensor_backed_dfb(spec, (1, 1), byte_offset=0)

    assert dfb.tensor_backing is spec


def test_row_major_static_tensor_is_not_silently_validated_as_tiled():
    validator = build_operation_validator(
        add_operation, grid=(1, 1), target_arch="blackhole"
    )
    row_major = StaticTensorSpec(
        (32, 32),
        torch.bfloat16,
        layout="ROW_MAJOR",
        tile_shape=None,
    )

    with pytest.raises(ValueError, match="requires tilized tensors"):
        validator(row_major, row_major, row_major)


def test_unused_row_major_argument_is_still_rejected():
    validator = build_operation_validator(
        copy_operation_with_unused_tensor, grid=(1, 1), target_arch="blackhole"
    )
    row_major = StaticTensorSpec(
        (32, 32),
        torch.bfloat16,
        layout="ROW_MAJOR",
        tile_shape=None,
    )

    with pytest.raises(ValueError, match="requires tilized tensors"):
        validator(row_major, _spec(), _spec())


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


def test_grid_size_one_dimensional_is_accepted():
    def one_dimensional_grid_size(lhs, rhs, out):
        ttl.grid_size(dims=1)

        @ttl.compute()
        def compute():
            pass

    validator = build_operation_validator(
        one_dimensional_grid_size, grid=(1, 1), target_arch="blackhole"
    )

    assert validator(_spec(), _spec(), _spec()) is None


def test_captured_tuple_subscript_is_rejected():
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

    with pytest.raises(TTLangCompileError, match="only supports subscripting tensors"):
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
