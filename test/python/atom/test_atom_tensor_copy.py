# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Unified @ttl.operation tensor copy: read one tile from
a ttnn tensor into a DFB and write it back out. The compute thread is
empty, so the splitter emits an empty compute kernel alongside the two
data-movement threads."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.operation(grid=(1, 1))
def atom_tensor_copy(src, dst):
    cb = ttl.make_dataflow_buffer_like(src, shape=(1, 1), block_count=2)
    blk_in = cb.reserve()
    ttl.copy(src[0:1, 0:1], blk_in)
    blk_out = cb.wait()
    ttl.copy(blk_out, dst[0:1, 0:1])


def test_atom_tensor_copy(device):
    tile = ttnn.TILE_SIZE
    src_t = torch.randn(tile, tile, dtype=torch.bfloat16)

    src = to_l1(src_t, device)
    dst = to_l1(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_tensor_copy(src, dst)

    got = ttnn.to_torch(dst).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, src_t, rtol=1e-3, atol=1e-3)
