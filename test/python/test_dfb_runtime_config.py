# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving final physical DFB allocation metadata."""

import pytest

from ttl.dataflow_buffer import DFBStorageSegment, PhysicalDFBConfig
from ttl.dialects import ttcore  # noqa: F401
from ttl.ir import Context, Module
from ttl.ttl_api import _resolve_dfb_configs


def _entry(
    dfb_index,
    *,
    num_tiles=1,
    element_type="bf16",
    block_count=2,
    page_size=2048,
):
    """Build one textual physical-allocation metadata entry."""

    return (
        f"{{dfb_index = {dfb_index} : i32, num_tiles = {num_tiles} : i32, "
        f"element_type = {element_type}, block_count = {block_count} : i32, "
        f"page_size = {page_size} : i32}}"
    )


def _module(allocations=None):
    """Parse a module with optional physical-allocation metadata."""

    if allocations is None:
        return Module.parse("module {}")
    entries = ", ".join(allocations)
    return Module.parse(f"module attributes {{ttl.dfb_allocations = [{entries}]}} {{}}")


def test_complete_physical_allocations_are_sorted():
    with Context():
        module = _module(
            [
                _entry(
                    1,
                    num_tiles=4,
                    element_type="f32",
                    block_count=3,
                    page_size=4096,
                ),
                _entry(0, num_tiles=2),
            ]
        )

        assert _resolve_dfb_configs(module) == [
            PhysicalDFBConfig(0, 2, "bfloat16", 2, 2048, None),
            PhysicalDFBConfig(1, 4, "float32", 3, 4096, None),
        ]


def test_tensor_backing_segments_preserve_nodes_and_tensor_range():
    with Context():
        module = Module.parse(
            """module attributes {ttl.dfb_allocations = [{
              block_count = 1 : i32,
              dfb_index = 0 : i32,
              element_type = !ttcore.tile<32x32, bf16>,
              num_tiles = 1 : i32,
              page_size = 2048 : i32,
              storage_segments = [{
                tensor_backing = #ttl.tensor_backing<
                  tensor_index = 2, byte_offset = 2048, byte_size = 2048>,
                nodes = [[1, 0], [0, 0]]
              }]
            }]} {}"""
        )

        assert _resolve_dfb_configs(module) == [
            PhysicalDFBConfig(
                0,
                1,
                "bfloat16",
                1,
                2048,
                (32, 32),
                (
                    DFBStorageSegment(
                        nodes=((0, 0), (1, 0)),
                        tensor_index=2,
                        byte_offset=2048,
                        byte_size=2048,
                    ),
                ),
            )
        ]


@pytest.mark.parametrize(
    ("element_type", "data_format", "tile", "page_size"),
    [
        ("!ttcore.tile<32x32, bfp_bf4>", "bfloat4_b", (32, 32), 576),
        ("!ttcore.tile<32x32, bfp_bf8>", "bfloat8_b", (32, 32), 1088),
        ("!ttcore.tile<1x16, bf16>", "bfloat16", (1, 16), 32),
        ("!ttcore.tile<1x16, u8>", "uint8", (1, 16), 16),
        ("!ttcore.tile<32x32, u16>", "uint16", (32, 32), 2048),
        ("!ttcore.tile<32x32, u32>", "uint32", (32, 32), 4096),
        ("!ttcore.tile<32x32, si32>", "int32", (32, 32), 4096),
    ],
)
def test_complete_physical_allocations_preserve_tile_types(
    element_type, data_format, tile, page_size
):
    with Context():
        module = _module([_entry(0, element_type=element_type, page_size=page_size)])

        assert _resolve_dfb_configs(module) == [
            PhysicalDFBConfig(0, 1, data_format, 2, page_size, tile)
        ]


def test_empty_complete_physical_allocations_replace_frontend_configs():
    with Context():
        module = _module([])

        assert _resolve_dfb_configs(module) == []


def test_missing_complete_allocations_are_rejected():
    with Context():
        module = _module()

        with pytest.raises(ValueError, match="missing ttl.dfb_allocations"):
            _resolve_dfb_configs(module)


@pytest.mark.parametrize(
    ("allocations", "message"),
    [
        ([_entry(-1)], "dfb_index must be non-negative"),
        ([_entry(0, num_tiles=0)], "num_tiles must be positive"),
        ([_entry(0, block_count=0)], "block_count must be positive"),
        ([_entry(0, page_size=0)], "page_size must be positive"),
        ([_entry(0, element_type="i1")], "Unrecognized MLIR scalar element type"),
        ([_entry(0), _entry(0)], "duplicate dfb_index 0"),
        ([_entry(1)], "dense physical index range"),
        (
            [
                "{dfb_index = 0 : i32, num_tiles = 1 : i32, "
                "element_type = bf16, page_size = 2048 : i32}"
            ],
            "missing 'block_count'",
        ),
        (
            [
                "{dfb_index = 0 : i32, num_tiles = 1 : i32, "
                "element_type = bf16, block_count = 2 : i32}"
            ],
            "missing 'page_size'",
        ),
    ],
)
def test_invalid_complete_physical_allocations_are_rejected(allocations, message):
    with Context():
        module = _module(allocations)

        with pytest.raises(ValueError, match=message):
            _resolve_dfb_configs(module)
