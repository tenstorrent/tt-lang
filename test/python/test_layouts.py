# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact TTNN tensor configuration conversion."""

from types import SimpleNamespace

import pytest

from ttl import layouts


class _FakeTTNN:
    TILE_LAYOUT = object()
    ROW_MAJOR_LAYOUT = object()

    class DataType:
        FLOAT32 = object()
        BFLOAT16 = object()
        BFLOAT8_B = object()
        BFLOAT4_B = object()
        INT32 = object()
        UINT32 = object()
        UINT16 = object()
        UINT8 = object()

    class StorageType:
        HOST = object()
        DEVICE = object()

    class BufferType:
        DRAM = object()
        L1 = object()
        L1_SMALL = object()
        TRACE = object()

    class TensorMemoryLayout:
        INTERLEAVED = object()
        SINGLE_BANK = object()
        HEIGHT_SHARDED = object()
        WIDTH_SHARDED = object()
        BLOCK_SHARDED = object()
        ND_SHARDED = object()


class _FakeTensor:
    def __init__(
        self,
        *,
        dtype=_FakeTTNN.DataType.BFLOAT16,
        tensor_layout=_FakeTTNN.TILE_LAYOUT,
        storage_type=_FakeTTNN.StorageType.DEVICE,
        buffer_type=_FakeTTNN.BufferType.DRAM,
        memory_layout=_FakeTTNN.TensorMemoryLayout.INTERLEAVED,
    ):
        self.dtype = dtype
        self.layout = tensor_layout
        self._storage_type = storage_type
        self._memory_config = SimpleNamespace(
            buffer_type=buffer_type,
            memory_layout=memory_layout,
        )

    def storage_type(self):
        return self._storage_type

    def memory_config(self):
        return self._memory_config


@pytest.fixture(autouse=True)
def _use_fake_ttnn(monkeypatch):
    monkeypatch.setattr(layouts, "_get_ttnn", lambda: _FakeTTNN)


def test_supported_dtype_layout_table_is_exact():
    supported = set(layouts.get_supported_ttnn_dtype_layouts(_FakeTTNN))
    tiled_dtypes = {
        _FakeTTNN.DataType.FLOAT32,
        _FakeTTNN.DataType.BFLOAT16,
        _FakeTTNN.DataType.BFLOAT8_B,
        _FakeTTNN.DataType.BFLOAT4_B,
        _FakeTTNN.DataType.INT32,
        _FakeTTNN.DataType.UINT32,
        _FakeTTNN.DataType.UINT16,
        _FakeTTNN.DataType.UINT8,
    }
    row_major_dtypes = tiled_dtypes - {
        _FakeTTNN.DataType.BFLOAT8_B,
        _FakeTTNN.DataType.BFLOAT4_B,
    }
    expected = {(dtype, _FakeTTNN.TILE_LAYOUT) for dtype in tiled_dtypes} | {
        (dtype, _FakeTTNN.ROW_MAJOR_LAYOUT) for dtype in row_major_dtypes
    }

    assert supported == expected


@pytest.mark.parametrize(
    ("ttnn_layout", "ttl_layout"),
    [
        (
            _FakeTTNN.TensorMemoryLayout.INTERLEAVED,
            layouts.TENSOR_MEMORY_LAYOUT_INTERLEAVED,
        ),
        (
            _FakeTTNN.TensorMemoryLayout.HEIGHT_SHARDED,
            layouts.TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED,
        ),
        (
            _FakeTTNN.TensorMemoryLayout.WIDTH_SHARDED,
            layouts.TENSOR_MEMORY_LAYOUT_WIDTH_SHARDED,
        ),
        (
            _FakeTTNN.TensorMemoryLayout.BLOCK_SHARDED,
            layouts.TENSOR_MEMORY_LAYOUT_BLOCK_SHARDED,
        ),
        (
            _FakeTTNN.TensorMemoryLayout.ND_SHARDED,
            layouts.TENSOR_MEMORY_LAYOUT_ND_SHARDED,
        ),
    ],
)
def test_detect_memory_layout_maps_exact_supported_values(ttnn_layout, ttl_layout):
    tensor = _FakeTensor(memory_layout=ttnn_layout)

    assert layouts.detect_memory_layout(tensor) == ttl_layout


@pytest.mark.parametrize(
    "memory_layout",
    [_FakeTTNN.TensorMemoryLayout.SINGLE_BANK, object()],
)
def test_detect_memory_layout_rejects_unsupported_values(memory_layout):
    tensor = _FakeTensor(memory_layout=memory_layout)

    with pytest.raises(ValueError, match="Unsupported TTNN tensor memory layout"):
        layouts.detect_memory_layout(tensor)


@pytest.mark.parametrize(
    ("ttnn_buffer_type", "ttl_buffer_type"),
    [
        (_FakeTTNN.BufferType.DRAM, layouts.BUFFER_TYPE_DRAM),
        (_FakeTTNN.BufferType.L1, layouts.BUFFER_TYPE_L1),
        (_FakeTTNN.BufferType.L1_SMALL, layouts.BUFFER_TYPE_L1_SMALL),
    ],
)
def test_detect_buffer_type_maps_exact_device_values(ttnn_buffer_type, ttl_buffer_type):
    tensor = _FakeTensor(buffer_type=ttnn_buffer_type)

    assert layouts.detect_buffer_type(tensor) == ttl_buffer_type


def test_detect_buffer_type_identifies_host_storage_before_memory_config():
    tensor = _FakeTensor(
        storage_type=_FakeTTNN.StorageType.HOST,
        buffer_type=_FakeTTNN.BufferType.TRACE,
    )

    assert layouts.detect_buffer_type(tensor) == layouts.BUFFER_TYPE_SYSTEM_MEMORY


@pytest.mark.parametrize(
    ("storage_type", "buffer_type", "message"),
    [
        (
            _FakeTTNN.StorageType.DEVICE,
            _FakeTTNN.BufferType.TRACE,
            "trace buffers cannot be tensor arguments",
        ),
        (
            _FakeTTNN.StorageType.DEVICE,
            object(),
            "Unsupported TTNN buffer type",
        ),
        (object(), _FakeTTNN.BufferType.DRAM, "Unsupported TTNN tensor storage type"),
    ],
)
def test_detect_buffer_type_rejects_unsupported_values(
    storage_type, buffer_type, message
):
    tensor = _FakeTensor(storage_type=storage_type, buffer_type=buffer_type)

    with pytest.raises(ValueError, match=message):
        layouts.detect_buffer_type(tensor)


@pytest.mark.parametrize(
    ("dtype", "tensor_layout", "expected"),
    [
        (
            _FakeTTNN.DataType.BFLOAT8_B,
            _FakeTTNN.TILE_LAYOUT,
            layouts.TENSOR_LAYOUT_TILE,
        ),
        (
            _FakeTTNN.DataType.UINT16,
            _FakeTTNN.ROW_MAJOR_LAYOUT,
            layouts.TENSOR_LAYOUT_ROW_MAJOR,
        ),
    ],
)
def test_detect_tensor_layout_accepts_supported_combinations(
    dtype, tensor_layout, expected
):
    tensor = _FakeTensor(dtype=dtype, tensor_layout=tensor_layout)

    assert layouts.detect_tensor_layout(tensor) == expected


def test_detect_tensor_layout_rejects_packed_row_major_dtype():
    tensor = _FakeTensor(
        dtype=_FakeTTNN.DataType.BFLOAT8_B,
        tensor_layout=_FakeTTNN.ROW_MAJOR_LAYOUT,
    )

    with pytest.raises(ValueError, match="Unsupported TTNN dtype/layout combination"):
        layouts.detect_tensor_layout(tensor)


def test_detect_tensor_layout_rejects_unknown_layout():
    tensor = _FakeTensor(tensor_layout=object())

    with pytest.raises(ValueError, match="Unsupported TTNN tensor layout"):
        layouts.detect_tensor_layout(tensor)
