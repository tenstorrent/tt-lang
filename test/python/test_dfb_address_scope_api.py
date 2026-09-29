# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python API validation for DFB address scopes."""

import ast
import copy

import pytest
import ttl

from ttl import dataflow_buffer
from ttl.atom import _lift_setup


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


@pytest.mark.parametrize(
    "address_scope",
    [None, "local", dataflow_buffer.DFBAddressScope.REMOTE_UNIFORM],
)
def test_all_dfb_factories_preserve_address_scope(monkeypatch, address_scope):
    monkeypatch.setattr("ttl.dtype_utils.is_ttnn_tensor", lambda tensor: True)
    tensor = _FakeTensor()

    explicit = dataflow_buffer.make_dfb(
        "bf16", shape=(1, 1), address_scope=address_scope
    )
    tensor_like = dataflow_buffer.make_dataflow_buffer_like(
        tensor, shape=(1, 1), address_scope=address_scope
    )
    tensor_backed = dataflow_buffer.make_tensor_backed_dfb(
        tensor, shape=(1, 1), address_scope=address_scope
    )

    expected = (
        None
        if address_scope is None
        else dataflow_buffer.DFBAddressScope(address_scope)
    )
    assert explicit.address_scope == expected
    assert tensor_like.address_scope == expected
    assert tensor_backed.address_scope == expected


def test_invalid_address_scope_is_rejected():
    with pytest.raises(
        ValueError,
        match="DFB address_scope must be DFBAddressScope.LOCAL",
    ):
        dataflow_buffer.make_dfb(
            "bf16", shape=(1, 1), address_scope="operation_uniform"
        )


@pytest.mark.parametrize(
    "scope", ["remote_uniform", ttl.DFBAddressScope.REMOTE_UNIFORM]
)
def test_operation_captures_address_scope(scope):
    @ttl.operation(grid=(1, 1))
    def operation(inp):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), address_scope=scope)
        block = dfb.wait()
        block.pop()

    assert operation._spec.compile_time_captures["scope"] == scope
    assert operation._spec.frozen_scope["scope"] == scope


def test_operation_accepts_inline_address_scope_enum():
    @ttl.operation(grid=(1, 1))
    def operation(inp):
        dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), address_scope=ttl.DFBAddressScope.REMOTE_UNIFORM
        )
        block = dfb.wait()
        block.pop()

    assert operation._spec.frozen_scope["ttl"] is ttl


@pytest.mark.parametrize(
    "scope", ["remote_uniform", ttl.DFBAddressScope.REMOTE_UNIFORM]
)
def test_composed_operation_lifts_captured_address_scope(scope):
    @ttl.operation()
    def helper():
        dfb = ttl.make_dfb("bf16", shape=(1, 1), address_scope=scope)
        block = dfb.wait()
        block.pop()

    @ttl.operation(grid=(1, 1))
    def operation():
        helper()

    spec = operation._spec
    ast.parse(spec.source)
    _, dfbs, _, _ = _lift_setup(
        copy.deepcopy(spec.fn_ast), dict(spec.frozen_scope), spec.operation_identity
    )
    assert len(dfbs) == 1
    assert next(iter(dfbs.values())).address_scope == ttl.DFBAddressScope.REMOTE_UNIFORM


def test_captured_address_scope_changes_operation_identity():
    def make_operation(scope):
        @ttl.operation(grid=(1, 1))
        def operation():
            dfb = ttl.make_dfb("bf16", shape=(1, 1), address_scope=scope)
            block = dfb.wait()
            block.pop()

        return operation

    local = make_operation(ttl.DFBAddressScope.LOCAL)
    remote = make_operation(ttl.DFBAddressScope.REMOTE_UNIFORM)
    assert local._spec.operation_identity != remote._spec.operation_identity
