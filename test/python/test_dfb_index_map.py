# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side tests for post-inlining DFB index reuse metadata."""

from types import SimpleNamespace

import pytest

from ttl.dataflow_buffer import (
    CompilerAllocatedDFBConfig,
    _reset_cb_counter,
    get_cb_count,
    make_dfb,
)
from ttl.atom import _cb_configs_from_lifted
from ttl.ttl_api import _apply_dfb_index_map, _merge_dfb_configs


def _user_config(shape, block_count):
    return SimpleNamespace(shape=shape, block_count=block_count)


def _module_with_map(*entries):
    operation = SimpleNamespace(attributes={"ttl.dfb_index_map": list(entries)})
    return SimpleNamespace(operation=operation)


def test_user_remap_keeps_largest_physical_capacity():
    small = _user_config((1, 1), 2)
    large = _user_config((2, 4), 4)
    module = _module_with_map({"old_index": 2, "new_index": 0})

    remapped = _apply_dfb_index_map([small, None, large], module)

    assert remapped == [large]


def test_user_remap_preserves_dense_holes_for_physical_indices():
    config = _user_config((1, 1), 2)
    module = _module_with_map({"old_index": 2, "new_index": 1})

    remapped = _apply_dfb_index_map([None, None, config], module)

    assert remapped == [None, config]


def test_absent_user_remap_is_identity():
    configs = [_user_config((1, 1), 2)]
    module = SimpleNamespace(operation=SimpleNamespace(attributes={}))

    assert _apply_dfb_index_map(configs, module) is configs


def test_compiler_config_grows_reused_user_slot():
    user = _user_config((1, 1), 2)
    compiler = CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=8,
        data_format="bfloat16",
        block_count=4,
    )

    assert _merge_dfb_configs([user], [compiler]) == [compiler]


def test_larger_user_config_wins_over_compiler_member():
    user = _user_config((4, 4), 4)
    compiler = CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=1,
        data_format="bfloat16",
        block_count=2,
    )

    assert _merge_dfb_configs([user], [compiler]) == [user]


def test_explicit_reuse_key_shares_one_logical_index():
    _reset_cb_counter()

    first = make_dfb("bf16", (1, 8), block_count=2, reuse="workspace.q")
    independent = make_dfb("bf16", (1, 1), block_count=1)
    second = make_dfb("bf16", (1, 8), block_count=2, reuse="workspace.q")

    assert first._cb_index == second._cb_index == 0
    assert independent._cb_index == 1
    assert get_cb_count() == 2


def test_explicit_reuse_accepts_different_capacity():
    _reset_cb_counter()
    small = make_dfb(
        "bf16", (1, 4), block_count=2, reuse="workspace.q"
    )
    large = make_dfb(
        "bf16", (1, 8), block_count=4, reuse="workspace.q"
    )

    assert small._cb_index == large._cb_index == 0
    assert _cb_configs_from_lifted({
        "small": small,
        "large": large,
    }) == [large]


@pytest.mark.parametrize(
    "dtype,tile",
    [
        ("bf16", (8, 32)),
        ("bfp8", (32, 32)),
    ],
)
def test_explicit_reuse_rejects_incompatible_page_geometry(dtype, tile):
    _reset_cb_counter()
    make_dfb("bf16", (1, 8), block_count=2, reuse="workspace.q")

    with pytest.raises(ValueError, match="incompatible declarations"):
        make_dfb(
            dtype,
            (1, 8),
            block_count=2,
            tile=tile,
            reuse="workspace.q",
        )


@pytest.mark.parametrize("reuse", ["", 7])
def test_explicit_reuse_requires_nonempty_string(reuse):
    _reset_cb_counter()
    with pytest.raises(ValueError, match="non-empty string"):
        make_dfb("bf16", (1, 1), reuse=reuse)


def test_explicit_reuse_registry_is_per_compilation():
    _reset_cb_counter()
    make_dfb("bf16", (1, 8), reuse="workspace.q")

    _reset_cb_counter()
    fresh = make_dfb("bf16", (1, 4), reuse="workspace.q")

    assert fresh._cb_index == 0
    assert get_cb_count() == 1
