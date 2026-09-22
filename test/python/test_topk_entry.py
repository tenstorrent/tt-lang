# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttl.math TopK entry points emit the tile operations."""

import ttl
import ttl.dialects.ttl as ttl_dialect
from ttl.ir import Context, InsertionPoint, Location, Module


def test_math_topk_entry_points_emit_tile_ops():
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
            ttl.math.topk_local_sort(0, 0, 4, 0, fused=True, largest=False)
            ttl.math.topk_merge(0, 0, 32, stable_sort=True, tie_order="ascending")
            ttl.math.topk_rebuild(
                0,
                0,
                0,
                32,
                5,
                0,
                rank_stamped=True,
                tag_bits=8,
            )
        text = str(module)

    assert "ttl.tile_topk_local_sort" in text
    assert "fused = true" in text
    assert "largest = false" in text
    assert "ttl.tile_topk_merge" in text
    assert "#ttl.topk_tie_order<ascending>" in text
    assert "ttl.tile_topk_rebuild" in text
    assert "tag_bits = 8" in text


def test_math_topk_does_not_expose_slab_helpers():
    """The compiler emits the slab helpers, so they have no entry point."""
    for helper in (
        "topk_fuse_tile",
        "topk_defuse_tile",
        "topk_stamp_local_positions",
        "topk_strip_rank_tags",
        "topk_canonicalize_negzero_values",
    ):
        assert not hasattr(ttl.math, helper)
