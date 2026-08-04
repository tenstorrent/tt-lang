# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests compiler-allocated DFB tile metadata and runtime descriptors."""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
from ttl.dialects import ttl
from ttl.ir import Context, Module
from ttl.kernel_runner import build_cb_descriptors, emit_runner_source
from ttl.ttl_api import _extract_compiler_allocated_dfbs


def test_extract_compiler_allocated_subtile_metadata():
    with Context() as context:
        ttl.ensure_dialects_registered(context)
        module = Module.parse(
            """
            module attributes {
              ttl.compiler_allocated_dfbs = [{
                block_count = 2 : i32,
                dfb_index = 3 : i32,
                element_type = !ttcore.tile<16x32, bf16>,
                num_tiles = 5 : i32
              }]
            } {}
            """
        )
        configs = _extract_compiler_allocated_dfbs(module)

    assert configs == [
        CompilerAllocatedDFBConfig(
            dfb_index=3,
            num_tiles=5,
            data_format="bfloat16",
            block_count=2,
            tile=(16, 32),
        )
    ]


def test_compiler_allocated_subtile_descriptor():
    core_ranges = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
    )
    config = CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=3,
        data_format="bfloat16",
        block_count=2,
        tile=(16, 32),
    )

    descriptor = build_cb_descriptors([None], [config], core_ranges)[0]
    format_descriptor = descriptor.format_descriptors[0]

    assert descriptor.total_size == 6 * 16 * 32 * 2
    assert format_descriptor.page_size == 16 * 32 * 2
    assert format_descriptor.tile.height == 16
    assert format_descriptor.tile.width == 32


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
