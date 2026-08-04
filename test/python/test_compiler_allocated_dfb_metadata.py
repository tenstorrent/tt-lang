# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests compiler-allocated DFB tile metadata and runtime descriptors."""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.constants import DEFAULT_TILE_SIZE
from ttl.dataflow_buffer import CompilerAllocatedDFBConfig, DataflowBuffer, make_dfb
from ttl.dialects import ttl
from ttl.dtype_utils import format_name_to_ttnn_dtype, tile_bytes_from_dtype
from ttl.ir import Context, Module
from ttl.kernel_runner import build_cb_descriptors, emit_runner_source
from ttl.ttl_api import _extract_compiler_allocated_dfbs


MLIR_FORMAT_CASES = [
    ("bf16", "bfloat16"),
    ("f16", "float16"),
    ("f32", "float32"),
    ("si32", "int32"),
    ("u32", "uint32"),
    ("u16", "uint16"),
    ("u8", "uint8"),
    ("bfp_bf8", "bfloat8_b"),
    ("bfp_bf4", "bfloat4_b"),
]
TT_METAL_TILE_SIZES = [
    (32, 32),
    (16, 32),
    (32, 16),
    (16, 16),
    (8, 32),
    (4, 32),
    (2, 32),
    (1, 32),
    (8, 16),
    (4, 16),
    (2, 16),
    (1, 16),
]
RUNTIME_FORMATS = [case[1] for case in MLIR_FORMAT_CASES]


@pytest.mark.parametrize(
    "mlir_type,data_format,tile_hw",
    [
        (
            mlir_type,
            data_format,
            (8, 32),
        )
        for mlir_type, data_format in MLIR_FORMAT_CASES
    ],
    ids=[case[0] for case in MLIR_FORMAT_CASES],
)
def test_extract_compiler_allocated_subtile_metadata(mlir_type, data_format, tile_hw):
    tile_height, tile_width = tile_hw
    with Context() as context:
        ttl.ensure_dialects_registered(context)
        module = Module.parse(
            f"""
            module attributes {{
              ttl.compiler_allocated_dfbs = [{{
                block_count = 2 : i32,
                dfb_index = 3 : i32,
                element_type = !ttcore.tile<{tile_height}x{tile_width}, {mlir_type}>,
                num_tiles = 5 : i32
              }}]
            }} {{}}
            """
        )
        configs = _extract_compiler_allocated_dfbs(module)

    assert configs == [
        CompilerAllocatedDFBConfig(
            dfb_index=3,
            num_tiles=5,
            data_format=data_format,
            block_count=2,
            tile=tile_hw,
        )
    ]


@pytest.mark.parametrize(
    "data_format",
    RUNTIME_FORMATS,
)
@pytest.mark.parametrize(
    "tile_hw", TT_METAL_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
def test_compiler_allocated_subtile_descriptor(data_format, tile_hw):
    core_ranges = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
    )
    config = CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=3,
        data_format=data_format,
        block_count=2,
        tile=tile_hw,
    )

    descriptor = build_cb_descriptors([None], [config], core_ranges)[0]
    format_descriptor = descriptor.format_descriptors[0]
    page_size = _expected_page_size(data_format, tile_hw)
    tile_height, tile_width = tile_hw

    assert descriptor.total_size == 6 * page_size
    assert format_descriptor.page_size == page_size
    assert format_descriptor.tile.height == tile_height
    assert format_descriptor.tile.width == tile_width


def _expected_page_size(data_format, tile_hw):
    tile_elements = tile_hw[0] * tile_hw[1]
    if data_format in ("bfloat16", "float16", "uint16"):
        return tile_elements * 2
    if data_format in ("float32", "int32", "uint32"):
        return tile_elements * 4
    if data_format == "uint8":
        return tile_elements
    # Matches tt-metal's face-row exponent layout and L1 alignment.
    elements_per_exponent = 16
    l1_alignment_bytes = 16
    assert tile_elements % elements_per_exponent == 0
    exponent_count = tile_elements // elements_per_exponent
    exponent_bytes = (
        (exponent_count + l1_alignment_bytes - 1) // l1_alignment_bytes
    ) * l1_alignment_bytes
    if data_format == "bfloat8_b":
        return tile_elements + exponent_bytes
    if data_format == "bfloat4_b":
        return tile_elements // 2 + exponent_bytes
    raise AssertionError(f"Unhandled test data format: {data_format}")


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize(
    "tile_hw", TT_METAL_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
def test_bfp_storage_tile_size(dtype, tile_hw):
    data_format = "bfloat8_b" if dtype == ttnn.bfloat8_b else "bfloat4_b"
    page_size = tile_bytes_from_dtype(dtype, tile_hw)

    assert page_size == _expected_page_size(data_format, tile_hw)


@pytest.mark.parametrize(
    "dtype,tile_hw,expected_size",
    [
        (ttnn.uint8, (16, 32), 512),
        (ttnn.bfloat8_b, (32, 32), 1088),
        (ttnn.bfloat4_b, (32, 32), 576),
    ],
    ids=["uint8-subtile", "bfp8", "bfp4"],
)
def test_tile_size_for_remaining_ttnn_formats(dtype, tile_hw, expected_size):
    assert tile_bytes_from_dtype(dtype, tile_hw) == expected_size


def test_rank_three_user_dfb_capacity():
    core_ranges = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
    )
    config = DataflowBuffer(
        tensor=None,
        shape=(2, 2, 2),
        block_count=2,
        dtype=ttnn.bfloat16,
    )

    descriptor = build_cb_descriptors([None], [config], core_ranges)[0]

    assert descriptor.total_size == 16 * 2048


def test_make_dfb_uses_default_tile_size():
    config = make_dfb(ttnn.bfloat16, shape=(1, 1))

    assert config.tile == (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)


@pytest.mark.parametrize("mlir_type", ["bfp_f8", "bfp_f4", "bfp_f2", "bfp_bf2", "i1"])
def test_unsupported_compiler_allocated_dtype_is_diagnosed(mlir_type):
    with Context() as context:
        ttl.ensure_dialects_registered(context)
        module = Module.parse(
            f"""
            module attributes {{
              ttl.compiler_allocated_dfbs = [{{
                block_count = 1 : i32,
                dfb_index = 0 : i32,
                element_type = !ttcore.tile<32x32, {mlir_type}>,
                num_tiles = 1 : i32
              }}]
            }} {{}}
            """
        )
        with pytest.raises(ValueError, match="not supported by the ttnn runtime"):
            _extract_compiler_allocated_dfbs(module)


@pytest.mark.parametrize(
    "data_format,expected_dtype",
    [
        ("uint8", ttnn.uint8),
        ("bfloat8_b", ttnn.bfloat8_b),
        ("bfloat4_b", ttnn.bfloat4_b),
    ],
)
def test_remaining_compiler_formats_convert_to_ttnn(data_format, expected_dtype):
    assert format_name_to_ttnn_dtype(data_format) == expected_dtype


def test_emitted_runner_preserves_compiler_allocated_subtile():
    config = CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=3,
        data_format="bfloat16",
        block_count=2,
        tile=(16, 32),
    )

    source = emit_runner_source(
        kernel_specs=[],
        cb_configs=[config],
        grid_cols=1,
        grid_rows=1,
        num_tensors=0,
    )

    assert "((16, 32), ttnn.bfloat16, 1024, 6144)" in source
    assert "tile=ttnn.TileDescriptor(tile)" in source
