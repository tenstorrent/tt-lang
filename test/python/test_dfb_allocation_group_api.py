# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python validation for typed DFB allocation groups."""

from dataclasses import FrozenInstanceError

import pytest

from ttl import dataflow_buffer
from ttl.dfb_allocation_group import (
    _dfb_allocation_group_binding_scope,
    make_dfb_allocation_group,
)


class _FakeShardSpec:
    shape = (32, 32)


class _FakeMemoryConfig:
    buffer_type = "L1"
    memory_layout = "HEIGHT_SHARDED"
    shard_spec = _FakeShardSpec()


class _FakeTile:
    tile_shape = (32, 32)

    @staticmethod
    def get_tile_size(_dtype):
        return 2048


class _FakeTensor:
    dtype = "bfloat16"
    layout = "TILE"

    @staticmethod
    def memory_config():
        return _FakeMemoryConfig()

    @staticmethod
    def get_tile():
        return _FakeTile()


def test_all_dfb_factories_preserve_one_bound_allocation_group(monkeypatch):
    monkeypatch.setattr("ttl.dtype_utils.is_ttnn_tensor", lambda tensor: True)
    shared_allocation = make_dfb_allocation_group()
    tensor = _FakeTensor()

    with _dfb_allocation_group_binding_scope():
        explicit = dataflow_buffer.make_dfb(
            "bf16", shape=(1, 1), allocation_group=shared_allocation
        )
        tensor_like = dataflow_buffer.make_dataflow_buffer_like(
            tensor, shape=(1, 1), allocation_group=shared_allocation
        )
        tensor_backed = dataflow_buffer.make_tensor_backed_dfb(
            tensor, shape=(1, 1), allocation_group=shared_allocation
        )

    assert explicit.allocation_group is tensor_like.allocation_group
    assert explicit.allocation_group is tensor_backed.allocation_group
    assert explicit.allocation_group.declaration is shared_allocation
    assert explicit.allocation_group.ordinal == 0


def test_distinct_allocation_groups_receive_distinct_ordinals():
    first_group = make_dfb_allocation_group()
    second_group = make_dfb_allocation_group()

    with _dfb_allocation_group_binding_scope():
        first = dataflow_buffer.make_dfb(
            "bf16", shape=(1, 1), allocation_group=first_group
        )
        second = dataflow_buffer.make_dfb(
            "bf16", shape=(1, 1), allocation_group=second_group
        )

    assert first.allocation_group.ordinal == 0
    assert second.allocation_group.ordinal == 1


def test_allocation_group_is_immutable():
    allocation_group = make_dfb_allocation_group()

    with pytest.raises(FrozenInstanceError):
        allocation_group.ordinal = 1


def test_allocation_group_requires_an_operation_binding_scope():
    allocation_group = make_dfb_allocation_group()

    with pytest.raises(TypeError, match="requires an enclosing @ttl.operation"):
        dataflow_buffer.make_dfb(
            "bf16", shape=(1, 1), allocation_group=allocation_group
        )


def test_allocation_group_rejects_untyped_identity():
    with pytest.raises(TypeError, match="ttl.make_dfb_allocation_group"):
        dataflow_buffer.make_dfb("bf16", shape=(1, 1), allocation_group=object())
