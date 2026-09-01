# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests frontend dtype-name conversion used by DFB construction."""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.dataflow_buffer import make_dfb
from ttl.dtype_utils import format_name_to_ttnn_dtype


@pytest.mark.parametrize(
    "data_format,expected_dtype",
    [
        ("bfp_bf8", ttnn.bfloat8_b),
        ("bfp_bf4", ttnn.bfloat4_b),
        ("bfp8", ttnn.bfloat8_b),
        ("bfp4", ttnn.bfloat4_b),
    ],
)
def test_bfp_format_names_convert_to_ttnn(data_format, expected_dtype):
    """Compiler names and frontend aliases resolve to TTNN BFP dtypes."""
    assert format_name_to_ttnn_dtype(data_format) == expected_dtype


@pytest.mark.parametrize(
    "data_format,expected_dtype",
    [
        ("bfp8", ttnn.bfloat8_b),
        ("bfp4", ttnn.bfloat4_b),
    ],
)
def test_make_dfb_accepts_frontend_bfp_aliases(data_format, expected_dtype):
    """Frontend BFP aliases construct DFBs with the matching TTNN dtype."""
    assert make_dfb(data_format, shape=(1, 1), block_count=1).dtype == expected_dtype
