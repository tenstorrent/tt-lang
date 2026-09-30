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


@contextmanager
def _topk_operands(
    values="tensor<1x2x!ttcore.tile<32x32, bf16>>",
    indices="tensor<1x2x!ttcore.tile<32x32, u16>>",
):
    """Yield ``(values, indices, module)`` block arguments of the given types."""
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    with context, Location.unknown():
        module = Module.parse(
            f"""
            module {{
              func.func @topk(%values: {values}, %indices: {indices}) {{
                return
              }}
            }}
            """
        )
        func = module.body.operations[0]
        values_arg, indices_arg = func.regions[0].blocks[0].arguments
        with InsertionPoint(func.regions[0].blocks[0].operations[0]):
            yield values_arg, indices_arg, module


def test_math_topk_emits_tensor_op():
    with _topk_operands() as (values, indices, module):
        ttl.math.topk(values, 32, indices=indices, stable=True)
        text = str(module)

    assert "ttl.topk" in text
    assert "k = 32" in text
    assert "stable = true" in text
    assert "tensor<1x1x!ttcore.tile<32x32, bf16>>" in text
    assert "tensor<1x1x!ttcore.tile<32x32, u16>>" in text


def test_math_topk_small_k_returns_one_tile():
    with _topk_operands() as (values, indices, module):
        ttl.math.topk(values, 8, indices=indices)
        text = str(module)

    assert "k = 8" in text
    assert "tensor<1x1x!ttcore.tile<32x32, bf16>>" in text


# Every constraint the compiler enforces is reported at the call site with a
# message that names the argument.
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 12}, "k must be one of"),
        ({"k": 32, "dim": 0}, "only the last dimension"),
        ({"k": 32, "sorted": False}, "sorted=False is not supported"),
    ],
)
def test_math_topk_rejects_unsupported_arguments(kwargs, message):
    with _topk_operands() as (values, indices, _module):
        with pytest.raises(ValueError, match=message):
            ttl.math.topk(values, indices=indices, **kwargs)


def test_math_topk_rejects_non_u16_indices():
    with _topk_operands(indices="tensor<1x2x!ttcore.tile<32x32, bf16>>") as (
        values,
        indices,
        _module,
    ):
        with pytest.raises(ValueError, match="indices must be u16 tiles"):
            ttl.math.topk(values, 32, indices=indices)


def test_math_topk_rejects_indices_shape_mismatch():
    with _topk_operands(indices="tensor<1x4x!ttcore.tile<32x32, u16>>") as (
        values,
        indices,
        _module,
    ):
        with pytest.raises(ValueError, match="indices must have the shape of values"):
            ttl.math.topk(values, 32, indices=indices)


@pytest.mark.parametrize(
    "width_tiles, message",
    [
        (1, "power of two between 2 and 64 tiles"),
        (3, "power of two between 2 and 64 tiles"),
        (128, "power of two between 2 and 64 tiles"),
    ],
)
def test_math_topk_rejects_unsupported_width(width_tiles, message):
    with _topk_operands(
        values=f"tensor<1x{width_tiles}x!ttcore.tile<32x32, bf16>>",
        indices=f"tensor<1x{width_tiles}x!ttcore.tile<32x32, u16>>",
    ) as (values, indices, _module):
        with pytest.raises(ValueError, match=message):
            ttl.math.topk(values, 32, indices=indices)


def test_math_topk_rejects_scalar_operands():
    with _topk_operands(values="tensor<1x2xbf16>", indices="tensor<1x2xbf16>") as (
        values,
        indices,
        _module,
    ):
        with pytest.raises(ValueError, match="values must be a rank-2 block of tiles"):
            ttl.math.topk(values, 32, indices=indices)


def test_math_topk_requires_indices():
    with _topk_module():
        with pytest.raises(TypeError, match="indices"):
            ttl.math.topk(None, 32)
