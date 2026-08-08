# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only validation for tensor-backed DFB construction."""

import math

import pytest

from ttl import dataflow_buffer


class _FakeMemoryConfig:
    def __init__(
        self,
        buffer_type="L1",
        memory_layout="HEIGHT_SHARDED",
        shard_shape=(32, 512),
    ):
        self.buffer_type = buffer_type
        self.memory_layout = memory_layout
        self.shard_spec = _FakeShardSpec(shard_shape)


class _FakeShardSpec:
    def __init__(self, shape):
        self.shape = shape


class _FakeTile:
    def __init__(self, tile_shape=(32, 32)):
        self.tile_shape = tile_shape

    def get_tile_size(self, _dtype):
        return self.tile_shape[0] * self.tile_shape[1] * 2


class _FakeTensor:
    def __init__(
        self,
        *,
        dtype="bfloat16",
        layout="TILE",
        buffer_type="L1",
        memory_layout="HEIGHT_SHARDED",
        shard_shape=(32, 512),
        tile_shape=(32, 32),
    ):
        self.dtype = dtype
        self.layout = layout
        self._memory_config = _FakeMemoryConfig(buffer_type, memory_layout, shard_shape)
        self._tile = _FakeTile(tile_shape)

    def memory_config(self):
        return self._memory_config

    def get_tile(self):
        return self._tile


def test_make_tensor_backed_dfb_records_complete_capacity(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )
    dataflow_buffer._reset_cb_counter()

    tensor = _FakeTensor()
    dfb = dataflow_buffer.make_tensor_backed_dfb(
        tensor, shape=(1, 4), block_count=2, byte_offset=2048
    )

    assert dfb.tensor_backing is tensor
    assert dfb.shape == (1, 4)
    assert dfb.block_count == 2
    assert dfb.byte_offset == 2048
    assert dfb.byte_size == 16384


@pytest.mark.parametrize(
    "memory_layout", ["HEIGHT_SHARDED", "WIDTH_SHARDED"], ids=["height", "width"]
)
def test_make_tensor_backed_dfb_accepts_supported_sharded_layouts(
    monkeypatch, memory_layout
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    dfb = dataflow_buffer.make_tensor_backed_dfb(
        _FakeTensor(memory_layout=memory_layout), shape=(1, 4)
    )

    assert dfb.shape == (1, 4)


def test_make_tensor_backed_dfb_accepts_range_ending_at_shard_boundary(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    dfb = dataflow_buffer.make_tensor_backed_dfb(
        _FakeTensor(shard_shape=(32, 96)),
        shape=(1, 2),
        byte_offset=2048,
    )

    assert dfb.byte_offset == 2048
    assert dfb.byte_size == 4096


@pytest.mark.parametrize(
    ("tile_shape", "dfb_shape", "shard_shape", "expected_page_size"),
    [
        ((16, 32), (1, 3), (16, 96), 1024),
        ((32, 32), (1, 7), (32, 224), 2048),
    ],
    ids=["16x32", "32x32"],
)
def test_make_tensor_backed_dfb_accepts_compact_storage_view(
    monkeypatch, tile_shape, dfb_shape, shard_shape, expected_page_size
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    dfb = dataflow_buffer.make_tensor_backed_dfb(
        _FakeTensor(tile_shape=(1, 32), shard_shape=shard_shape),
        shape=dfb_shape,
        tile=tile_shape,
    )

    assert dfb.tile == tile_shape
    assert dfb.byte_size == math.prod(dfb_shape) * expected_page_size


@pytest.mark.parametrize(
    ("storage_tile", "view_tile"),
    [
        ((1, 32), (8, 32)),
        ((1, 32), (16, 16)),
        ((32, 32), (16, 32)),
    ],
)
def test_make_tensor_backed_dfb_rejects_unsupported_storage_view(
    monkeypatch, storage_tile, view_tile
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="tile views require"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(tile_shape=storage_tile),
            shape=(1, 1),
            tile=view_tile,
        )


@pytest.mark.parametrize("tile", [(16,), (16, 32, 32), (16.0, 32), "16x32"])
def test_make_tensor_backed_dfb_rejects_invalid_view_tile(monkeypatch, tile):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="exactly two integer dimensions"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(tile_shape=(1, 32)), shape=(1, 1), tile=tile
        )


def test_make_tensor_backed_dfb_aligns_offset_to_view_page(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="1024-byte DFB page size"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(tile_shape=(1, 32)),
            shape=(1, 1),
            byte_offset=64,
            tile=(16, 32),
        )


def test_make_tensor_backed_dfb_rejects_range_past_shard_boundary(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="exceeds logical per-shard size 4096"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(shard_shape=(32, 64)),
            shape=(1, 2),
            byte_offset=2048,
        )


@pytest.mark.parametrize(
    ("byte_offset", "error_type"),
    [
        (-1, ValueError),
        (1, ValueError),
        ((1 << 32) - 2048, ValueError),
        (1 << 32, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_make_tensor_backed_dfb_rejects_invalid_byte_ranges(
    monkeypatch, byte_offset, error_type
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(error_type):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(), shape=(1, 1), byte_offset=byte_offset
        )


@pytest.mark.parametrize("block_count", [0, 33, 1.5, True])
def test_make_tensor_backed_dfb_rejects_invalid_block_count(monkeypatch, block_count):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises((TypeError, ValueError)):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(), shape=(1, 1), block_count=block_count
        )


@pytest.mark.parametrize("shape", [(), (1, 0), (1, -1), (1, 1.5), (1, True)])
def test_make_tensor_backed_dfb_rejects_invalid_shape(monkeypatch, shape):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="positive integers"):
        dataflow_buffer.make_tensor_backed_dfb(_FakeTensor(), shape=shape)


def test_make_tensor_backed_dfb_rejects_uint32_capacity_overflow(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="uint32 descriptor fields"):
        dataflow_buffer.make_tensor_backed_dfb(_FakeTensor(), shape=(1, 1 << 21))


def test_make_tensor_backed_dfb_requires_ttnn_tensor(monkeypatch):
    monkeypatch.setattr("ttl.dtype_utils.is_ttnn_tensor", lambda _tensor: False)

    with pytest.raises(TypeError, match="requires a TTNN tensor"):
        dataflow_buffer.make_tensor_backed_dfb(object(), shape=(1, 1))


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (_FakeTensor(buffer_type="DRAM"), "must use L1 storage"),
        (_FakeTensor(memory_layout="INTERLEAVED"), "must be height- or width-sharded"),
        (_FakeTensor(layout="ROW_MAJOR"), "must use TILE layout"),
        (_FakeTensor(dtype="int32"), "supports BF16 and FP32"),
    ],
    ids=["dram", "interleaved", "row_major", "int32"],
)
def test_make_tensor_backed_dfb_rejects_unvalidated_tensor_contract(
    monkeypatch, tensor, message
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor",
        lambda candidate: isinstance(candidate, _FakeTensor),
    )

    with pytest.raises(ValueError, match=message):
        dataflow_buffer.make_tensor_backed_dfb(tensor, shape=(1, 1))
