# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python API validation for DFB address scopes."""

import pytest

from ttl import dataflow_buffer


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


@pytest.mark.parametrize("address_scope", [None, "local", "remote_uniform"])
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

    assert explicit.address_scope == address_scope
    assert tensor_like.address_scope == address_scope
    assert tensor_backed.address_scope == address_scope


def test_invalid_address_scope_is_rejected():
    with pytest.raises(
        ValueError,
        match="DFB address_scope must be 'local', 'remote_uniform', or None",
    ):
        dataflow_buffer.make_dfb(
            "bf16", shape=(1, 1), address_scope="operation_uniform"
        )
