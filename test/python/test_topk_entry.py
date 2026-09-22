# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttl.math TopK entry points emit the tile operations."""

from contextlib import contextmanager

import pytest

import ttl
import ttl.dialects.ttl as ttl_dialect
from ttl.ir import Context, InsertionPoint, Location, Module


@contextmanager
def _topk_module():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context, Location.unknown():
        module = Module.parse(
            """
            module {
              func.func @topk() {
                return
              }
            }
            """
        )
        func = module.body.operations[0]
        with InsertionPoint(func.regions[0].blocks[0].operations[0]):
            yield module


def test_math_topk_entry_points_emit_tile_ops():
    with _topk_module() as module:
        ttl.math.topk_local_sort(0, 0, 4, 0, order="ascending", fused=True)
        ttl.math.topk_merge(0, 0, 32, order="descending", stable_sort=True)
        ttl.math.topk_rebuild(
            0,
            0,
            0,
            32,
            5,
            0,
            order="descending",
            rank_stamped=True,
            tag_bits=8,
        )
        text = str(module)

    assert "ttl.tile_topk_local_sort" in text
    assert "fused = true, order = #ttl.topk_order<ascending>" in text
    assert "ttl.tile_topk_merge" in text
    assert "order = #ttl.topk_order<descending>, stable_sort = true" in text
    assert "ttl.tile_topk_rebuild" in text
    assert "tag_bits = 8" in text


def test_math_topk_requires_order():
    with _topk_module():
        with pytest.raises(TypeError, match="order"):
            ttl.math.topk_merge(0, 0, 32)


def test_math_topk_rejects_unknown_order():
    with _topk_module():
        with pytest.raises(ValueError, match="TopK order must be one of"):
            ttl.math.topk_merge(0, 0, 32, order="largest")


def test_math_topk_does_not_expose_slab_helpers():
    """Slab helpers stay inside the compiler."""
    for helper in (
        "topk_fuse_tile",
        "topk_defuse_tile",
        "topk_stamp_local_positions",
        "topk_strip_rank_tags",
        "topk_canonicalize_negzero_values",
    ):
        assert not hasattr(ttl.math, helper)


def test_math_topk_emits_tensor_op():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context, Location.unknown():
        module = Module.parse(
            """
            module {
              func.func @topk(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                              %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
                return
              }
            }
            """
        )
        func = module.body.operations[0]
        values, indices = func.regions[0].blocks[0].arguments
        with InsertionPoint(func.regions[0].blocks[0].operations[0]):
            ttl.math.topk(values, 32, indices=indices, stable=True)
        text = str(module)

    assert "ttl.topk" in text
    assert "k = 32" in text
    assert "stable = true" in text
    assert "tensor<1x1x!ttcore.tile<32x32, bf16>>" in text
    assert "tensor<1x1x!ttcore.tile<32x32, u16>>" in text


def test_math_topk_requires_indices():
    with _topk_module():
        with pytest.raises(ValueError, match="indices tensor"):
            ttl.math.topk(None, 32)
