# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests compiler-allocated DFB tile metadata and runtime descriptors."""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
from ttl.dialects import ttl
from ttl.dtype_utils import tile_bytes_from_dtype
from ttl.ir import Context, Module
from ttl.kernel_runner import build_cb_descriptors, emit_runner_source
from ttl.ttl_api import _extract_compiler_allocated_dfbs


MLIR_FORMAT_CASES = [
    ("bf16", "bfloat16", 2),
    ("f16", "float16", 2),
    ("f32", "float32", 4),
    ("si32", "int32", 4),
    ("u32", "uint32", 4),
    ("u16", "uint16", 2),
]
COMPUTE_TILE_SIZES = [(16, 16), (16, 32), (32, 16), (32, 32)]


@pytest.mark.parametrize(
    "mlir_type,data_format",
    [(case[0], case[1]) for case in MLIR_FORMAT_CASES],
    ids=[case[0] for case in MLIR_FORMAT_CASES],
)
def test_extract_compiler_allocated_subtile_metadata(mlir_type, data_format):
    with Context() as context:
        ttl.ensure_dialects_registered(context)
        module = Module.parse(
            f"""
            module attributes {{
              ttl.compiler_allocated_dfbs = [{{
                block_count = 2 : i32,
                dfb_index = 3 : i32,
                element_type = !ttcore.tile<16x32, {mlir_type}>,
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
            tile=(16, 32),
        )
    ]


@pytest.mark.parametrize(
    "data_format,bytes_per_element",
    [(case[1], case[2]) for case in MLIR_FORMAT_CASES],
    ids=[case[0] for case in MLIR_FORMAT_CASES],
)
@pytest.mark.parametrize(
    "tile_hw", COMPUTE_TILE_SIZES, ids=lambda tile: f"{tile[0]}x{tile[1]}"
)
def test_compiler_allocated_subtile_descriptor(data_format, bytes_per_element, tile_hw):
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
    tile_height, tile_width = tile_hw
    page_size = tile_height * tile_width * bytes_per_element

    assert descriptor.total_size == 6 * page_size
    assert format_descriptor.page_size == page_size
    assert format_descriptor.tile.height == tile_height
    assert format_descriptor.tile.width == tile_width


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_bfp_subtile_size_is_rejected(dtype):
    with pytest.raises(ValueError, match="supports only 32x32 tiles"):
        tile_bytes_from_dtype(dtype, (16, 32))


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
