# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only validation for tensor-backed DFB construction."""

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

    def get_tile_size(self, dtype):
        element_size = 4 if dtype == "float32" else 2
        return self.tile_shape[0] * self.tile_shape[1] * element_size


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
    ("dtype", "view_tile", "expected_page_size"),
    [
        ("bfloat16", (16, 32), 1024),
        ("bfloat16", (32, 32), 2048),
        ("float32", (16, 32), 2048),
        ("float32", (32, 32), 4096),
    ],
    ids=["bf16-half", "bf16-full", "fp32-half", "fp32-full"],
)
def test_make_tensor_backed_dfb_records_physical_row_page_view(
    monkeypatch, dtype, view_tile, expected_page_size
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )
    dataflow_buffer._reset_cb_counter()

    tensor = _FakeTensor(
        dtype=dtype,
        tile_shape=(1, 32),
        shard_shape=(1, 32 * 64),
    )
    dfb = dataflow_buffer.make_tensor_backed_dfb(
        tensor,
        shape=(1, 2),
        tile=view_tile,
    )

    assert dfb.tensor_backing is tensor
    assert dfb.tile == view_tile
    assert dfb.byte_size == 2 * expected_page_size


@pytest.mark.parametrize(
    ("storage_tile", "view_tile"),
    [
        ((32, 32), (16, 32)),
        ((1, 32), (8, 32)),
        ((1, 16), (16, 16)),
    ],
    ids=["full-to-half", "row-to-eight", "narrow-row"],
)
def test_make_tensor_backed_dfb_rejects_unsupported_physical_view(
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


@pytest.mark.parametrize("view_tile", [(), (32,), (32, 32, 1), (32, True)])
def test_make_tensor_backed_dfb_rejects_invalid_view_tile(monkeypatch, view_tile):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="integer dimensions"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(tile_shape=(1, 32)),
            shape=(1, 1),
            tile=view_tile,
        )


def test_make_tensor_backed_dfb_aligns_offset_to_physical_page(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="2048-byte DFB page size"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(tile_shape=(1, 32), shard_shape=(1, 2048)),
            shape=(1, 1),
            tile=(32, 32),
            byte_offset=64,
        )


@pytest.mark.parametrize(
    "memory_layout",
    ["HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED"],
    ids=["height", "width", "block"],
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


@pytest.mark.parametrize(
    "memory_layout_name",
    ["HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED"],
    ids=["height", "width", "block"],
)
def test_make_tensor_backed_dfb_accepts_ttnn_memory_layout_enums(
    monkeypatch, memory_layout_name
):
    ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )
    memory_layout = getattr(ttnn.TensorMemoryLayout, memory_layout_name)

    dfb = dataflow_buffer.make_tensor_backed_dfb(
        _FakeTensor(memory_layout=memory_layout), shape=(1, 4)
    )

    assert dfb.shape == (1, 4)


@pytest.mark.parametrize(
    "memory_layout",
    ["INTERLEAVED", "ND_SHARDED", "height_sharded", "NOT_HEIGHT_SHARDED"],
    ids=[
        "interleaved",
        "nd-sharded",
        "non-enum-spelling",
        "substring-collision",
    ],
)
def test_make_tensor_backed_dfb_rejects_unsupported_memory_layouts(
    monkeypatch, memory_layout
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(
        ValueError,
        match=rf"must be height-, width-, or block-sharded, got {memory_layout}$",
    ):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(memory_layout=memory_layout), shape=(1, 4)
        )


@pytest.mark.parametrize(
    "memory_layout_name",
    ["INTERLEAVED", "ND_SHARDED"],
    ids=["interleaved", "nd-sharded"],
)
def test_make_tensor_backed_dfb_rejects_unsupported_ttnn_memory_layout_enums(
    monkeypatch, memory_layout_name
):
    ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )
    memory_layout = getattr(ttnn.TensorMemoryLayout, memory_layout_name)

    with pytest.raises(
        ValueError,
        match=rf"must be height-, width-, or block-sharded, got {memory_layout}$",
    ):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(memory_layout=memory_layout), shape=(1, 4)
        )


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
        (_FakeTensor(layout="ROW_MAJOR"), "must use TILE layout"),
        (_FakeTensor(dtype="int32"), "supports BF16 and FP32"),
    ],
    ids=["dram", "row_major", "int32"],
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
