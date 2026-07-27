# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving final physical DFB allocation metadata."""

import pytest

from ttl.dataflow_buffer import PhysicalDFBConfig
from ttl.ttl_api import _resolve_dfb_configs


class _FakeModule:
    def __init__(self, attributes):
        self.operation = type("Operation", (), {"attributes": attributes})()


def _entry(dfb_index, *, num_tiles=1, element_type="bf16", block_count=2):
    return {
        "dfb_index": dfb_index,
        "num_tiles": num_tiles,
        "element_type": element_type,
        "block_count": block_count,
    }


def test_complete_physical_allocations_replace_frontend_configs():
    frontend_configs = [object(), object(), object()]
    module = _FakeModule(
        {
            "ttl.dfb_allocations": [
                _entry(1, num_tiles=4, element_type="f32", block_count=3),
                _entry(0, num_tiles=2),
            ],
            "ttl.compiler_allocated_dfbs": [_entry(2)],
        }
    )

    assert _resolve_dfb_configs(module, frontend_configs) == [
        PhysicalDFBConfig(0, 2, "bfloat16", 2),
        PhysicalDFBConfig(1, 4, "float32", 3),
    ]


@pytest.mark.parametrize(
    ("element_type", "data_format", "tile"),
    [
        ("!ttcore.tile<32x32, bfp_bf4>", "bfloat4_b", (32, 32)),
        ("!ttcore.tile<32x32, bfp_bf8>", "bfloat8_b", (32, 32)),
        ("!ttcore.tile<1x16, bf16>", "bfloat16", (1, 16)),
        ("!ttcore.tile<1x16, u8>", "uint8", (1, 16)),
        ("!ttcore.tile<32x32, u16>", "uint16", (32, 32)),
        ("!ttcore.tile<32x32, u32>", "uint32", (32, 32)),
        ("!ttcore.tile<32x32, si32>", "int32", (32, 32)),
    ],
)
def test_complete_physical_allocations_preserve_tile_types(
    element_type, data_format, tile
):
    module = _FakeModule(
        {"ttl.dfb_allocations": [_entry(0, element_type=element_type)]}
    )

    assert _resolve_dfb_configs(module, []) == [
        PhysicalDFBConfig(0, 1, data_format, 2, tile)
    ]


def test_empty_complete_physical_allocations_replace_frontend_configs():
    module = _FakeModule({"ttl.dfb_allocations": []})

    assert _resolve_dfb_configs(module, [object()]) == []


def test_absent_complete_allocations_use_legacy_compiler_metadata():
    frontend_config = object()
    module = _FakeModule(
        {"ttl.compiler_allocated_dfbs": [_entry(1, element_type="f32")]}
    )

    assert _resolve_dfb_configs(module, [frontend_config]) == [
        frontend_config,
        PhysicalDFBConfig(1, 1, "float32", 2),
    ]


@pytest.mark.parametrize(
    ("allocations", "message"),
    [
        ([_entry(-1)], "dfb_index must be non-negative"),
        ([_entry(0, num_tiles=0)], "num_tiles must be positive"),
        ([_entry(0, block_count=0)], "block_count must be positive"),
        (
            [_entry(0, element_type="!ttcore.tile<1, bf16>")],
            "Invalid tile dimensions",
        ),
        ([_entry(0), _entry(0)], "duplicate dfb_index 0"),
        ([_entry(1)], "dense physical index range"),
        (
            [
                {
                    "dfb_index": 0,
                    "num_tiles": 1,
                    "element_type": "bf16",
                }
            ],
            "missing 'block_count'",
        ),
    ],
)
def test_invalid_complete_physical_allocations_are_rejected(allocations, message):
    module = _FakeModule({"ttl.dfb_allocations": allocations})

    with pytest.raises(ValueError, match=message):
        _resolve_dfb_configs(module, [])
