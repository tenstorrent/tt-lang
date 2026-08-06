# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only validation for tensor-backed DFB construction."""

import pytest

from ttl import dataflow_buffer


class _FakeMemoryConfig:
    def __init__(self, buffer_type="L1", memory_layout="HEIGHT_SHARDED"):
        self.buffer_type = buffer_type
        self.memory_layout = memory_layout


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
        tile_shape=(32, 32),
    ):
        self.dtype = dtype
        self.layout = layout
        self._memory_config = _FakeMemoryConfig(buffer_type, memory_layout)
        self._tile = _FakeTile(tile_shape)

    def memory_config(self):
        return self._memory_config

    def get_tile(self):
        return self._tile


@pytest.mark.parametrize(
    ("dtype", "byte_offset", "expected_byte_size"),
    [
        ("bfloat16", 2048, 16384),
        ("float32", 4096, 32768),
        ("uint16", 2048, 16384),
    ],
    ids=["bf16", "fp32", "uint16"],
)
def test_make_tensor_backed_dfb_records_complete_capacity(
    monkeypatch, dtype, byte_offset, expected_byte_size
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )
    dataflow_buffer._reset_cb_counter()

    tensor = _FakeTensor(dtype=dtype)
    dfb = dataflow_buffer.make_tensor_backed_dfb(
        tensor, shape=(1, 4), block_count=2, byte_offset=byte_offset
    )

    assert dfb.tensor_backing is tensor
    assert dfb.shape == (1, 4)
    assert dfb.block_count == 2
    assert dfb.byte_offset == byte_offset
    assert dfb.byte_size == expected_byte_size


@pytest.mark.parametrize(
    ("view_tile", "expected_page_size"),
    [((16, 32), 1024), ((32, 32), 2048)],
    ids=["half", "full"],
)
def test_make_tensor_backed_dfb_records_row_page_view(
    monkeypatch, view_tile, expected_page_size
):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )
    dataflow_buffer._reset_cb_counter()

    tensor = _FakeTensor(tile_shape=(1, 32))
    dfb = dataflow_buffer.make_tensor_backed_dfb(
        tensor,
        shape=(1, 2),
        tile=view_tile,
    )

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
def test_make_tensor_backed_dfb_rejects_unsupported_tile_view(
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


def test_make_tensor_backed_dfb_aligns_offset_to_view_page(monkeypatch):
    monkeypatch.setattr(
        "ttl.dtype_utils.is_ttnn_tensor", lambda tensor: isinstance(tensor, _FakeTensor)
    )

    with pytest.raises(ValueError, match="1024-byte DFB page size"):
        dataflow_buffer.make_tensor_backed_dfb(
            _FakeTensor(tile_shape=(1, 32)),
            shape=(1, 1),
            tile=(16, 32),
            byte_offset=64,
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

    with pytest.raises(ValueError, match="uint32 descriptor ABI"):
        dataflow_buffer.make_tensor_backed_dfb(_FakeTensor(), shape=(1, 1 << 21))


def test_make_tensor_backed_dfb_requires_ttnn_tensor(monkeypatch):
    monkeypatch.setattr("ttl.dtype_utils.is_ttnn_tensor", lambda _tensor: False)

    with pytest.raises(TypeError, match="requires a TTNN tensor"):
        dataflow_buffer.make_tensor_backed_dfb(object(), shape=(1, 1))


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (_FakeTensor(buffer_type="DRAM"), "requires an L1 tensor"),
        (
            _FakeTensor(memory_layout="WIDTH_SHARDED"),
            "requires height-sharded memory",
        ),
        (_FakeTensor(layout="ROW_MAJOR"), "requires TILE layout"),
        (_FakeTensor(dtype="int32"), "supports BF16, FP32, and UINT16"),
    ],
    ids=["dram", "width_sharded", "row_major", "int32"],
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
