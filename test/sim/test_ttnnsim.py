# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest
import torch

from sim import ttnn, TTNN_AVAILABLE

# Marker for tests that require ttnn golden functions
requires_ttnn = pytest.mark.skipif(
    not TTNN_AVAILABLE, reason="ttnn not available (required for golden function tests)"
)


def test_constants_and_dtypes():
    assert isinstance(ttnn.TILE_SIZE, int)
    assert ttnn.TILE_SIZE > 0
    assert hasattr(ttnn, "TILE_LAYOUT")
    assert ttnn.bfloat16 == torch.bfloat16
    assert ttnn.float32 == torch.float32


def test_device_open_close():
    dev = ttnn.open_device(0)
    assert repr(dev).startswith("Device(id=")
    # closing should be a no-op
    assert ttnn.close_device(dev) is None


def test_device_compute_with_storage_grid_size():
    """Test that Device.compute_with_storage_grid_size() returns 8x8 grid."""
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()

    assert isinstance(grid, ttnn.CoreCoord)
    assert grid.x == 8, f"Expected grid.x=8, got {grid.x}"
    assert grid.y == 8, f"Expected grid.y=8, got {grid.y}"

    ttnn.close_device(device)


def test_tensor_rand_and_empty_and_to_torch():
    shape = (4, 8)
    t1 = ttnn.rand(shape, dtype=ttnn.float32)
    assert isinstance(t1, ttnn.Tensor)
    assert t1.shape == shape
    assert t1.dtype == torch.float32

    t2 = ttnn.empty(shape, dtype=ttnn.bfloat16)
    assert isinstance(t2, ttnn.Tensor)
    assert t2.shape == shape
    assert t2.dtype == torch.bfloat16

    # to_torch accepts both wrapper and raw torch tensors
    tt = ttnn.to_torch(t1)
    assert isinstance(tt, torch.Tensor)
    tt2 = ttnn.to_torch(torch.zeros(2, 2))
    assert isinstance(tt2, torch.Tensor)


def test_tensor_get_set_item_and_repr():
    # __repr__ contains shape (any tensor)
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert "shape=(3, 4)" in repr(ttnn.Tensor(a))

    # Tile-coordinate get/set require a tile-aligned tensor.
    raw = torch.zeros(64, 64, dtype=torch.float32)
    tw = ttnn.Tensor(raw)

    # set with ttnn.Tensor: tile (0, 0) → element rows 0:32, cols 0:32
    tw[0, 0] = ttnn.Tensor(torch.full((32, 32), 9.0, dtype=torch.float32))
    assert torch.all(tw.to_torch()[0:32, 0:32] == 9.0)

    # set with ttnn.Tensor: tile (0, 1) → element rows 0:32, cols 32:64
    tw[0, 1] = ttnn.Tensor(torch.full((32, 32), 7.0, dtype=torch.float32))
    assert torch.all(tw.to_torch()[0:32, 32:64] == 7.0)

    # bare-integer key (non-tuple) is wrapped as (1,), which is a 1-element key
    # on a 2-D tensor — rejected with ValueError (key length != tensor rank).
    with pytest.raises(ValueError, match="does not match tensor rank"):
        _ = tw[1]


def test_to_torch_type_errors():
    class Foo:
        pass

    bogus: Any = Foo()
    with pytest.raises(TypeError):
        ttnn.to_torch(bogus)


# ---- Tile-based indexing tests ----


def test_tensor_tile_based_getitem():
    """Test tile-based indexing with __getitem__."""
    # Create a 2x2 tile tensor (64x64 elements)
    t = ttnn.rand((64, 64), dtype=ttnn.float32)

    # Get a single tile
    tile = t[0:1, 0:1]
    assert isinstance(tile, ttnn.Tensor)
    assert tile.shape == (32, 32)

    # Get a row of tiles
    row = t[0:1, 0:2]
    assert row.shape == (32, 64)

    # Get a column of tiles
    col = t[0:2, 0:1]
    assert col.shape == (64, 32)

    # Get all tiles
    all_tiles = t[0:2, 0:2]
    assert all_tiles.shape == (64, 64)


def test_tensor_tile_based_setitem():
    """Test tile-based indexing with __setitem__."""
    # Create a 2x2 tile tensor (64x64 elements)
    t = ttnn.rand((64, 64), dtype=ttnn.float32)

    # Set a single tile with ttnn.Tensor
    tile_data = ttnn.Tensor(torch.ones(32, 32))
    t[0:1, 0:1] = tile_data

    # Verify the tile was set
    retrieved = t[0:1, 0:1]
    assert torch.allclose(retrieved.to_torch(), torch.ones(32, 32))

    # Set a tile with ttnn.Tensor
    t[1:2, 1:2] = ttnn.Tensor(torch.ones(32, 32) * 2.0)
    retrieved2 = t[1:2, 1:2]
    assert torch.allclose(retrieved2.to_torch(), torch.ones(32, 32) * 2.0)


def test_tensor_0d_raises():
    """Test that constructing a 0-d (scalar) Tensor raises ValueError."""
    with pytest.raises(ValueError, match="at least 1 dimension"):
        ttnn.Tensor(torch.tensor(5.0))


def test_tensor_tile_indexing_invalid_shape():
    """Test that tile indexing fails for key length mismatches."""
    # Passing slice(None, 1) (stop-only, no start) to a 1-D tensor reaches
    # _validate_tile_slice, which requires an explicit start value and raises.
    t1d = ttnn.Tensor(torch.randn(64))
    with pytest.raises(ValueError, match="must have explicit start value"):
        _ = t1d[slice(None, 1)]  # missing start -> our validation catches it

    # 2-element key on a 4-D tensor: rank mismatch must be caught explicitly
    # rather than silently treating only the last two dims.
    t4d = ttnn.Tensor(torch.randn(2, 2, 64, 64))
    with pytest.raises(ValueError, match="does not match tensor rank"):
        _ = t4d[0:1, 0:1]


def test_tensor_tile_indexing_invalid_tile_alignment():
    """Test that tile indexing fails for non-tile-aligned tensors."""
    # Create a tensor that's not a multiple of tile size
    t = ttnn.Tensor(torch.randn(60, 60))
    with pytest.raises(ValueError, match="not a multiple of tile dimension"):
        _ = t[0:1, 0:1]


# ---- Binary operations tests ----


def test_tensor_binary_add():
    """Test element-wise addition."""
    a = ttnn.Tensor(torch.ones(4, 4))
    b = ttnn.Tensor(torch.ones(4, 4) * 2.0)

    # Tensor + Tensor
    c = a + b
    assert isinstance(c, ttnn.Tensor)
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 3.0)

    # Tensor + scalar
    d = a + 3.0
    assert torch.allclose(d.to_torch(), torch.ones(4, 4) * 4.0)

    # Tensor + int scalar
    e = a + 5
    assert torch.allclose(e.to_torch(), torch.ones(4, 4) * 6.0)


def test_tensor_binary_sub():
    """Test element-wise subtraction."""
    a = ttnn.Tensor(torch.ones(4, 4) * 5.0)
    b = ttnn.Tensor(torch.ones(4, 4) * 2.0)

    c = a - b
    assert isinstance(c, ttnn.Tensor)
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 3.0)


def test_tensor_binary_mul():
    """Test element-wise multiplication."""
    a = ttnn.Tensor(torch.ones(4, 4) * 3.0)
    b = ttnn.Tensor(torch.ones(4, 4) * 2.0)

    c = a * b
    assert isinstance(c, ttnn.Tensor)
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 6.0)


def test_tensor_binary_div():
    """Test element-wise division."""
    a = ttnn.Tensor(torch.ones(4, 4) * 6.0)
    b = ttnn.Tensor(torch.ones(4, 4) * 2.0)

    # True division
    c = a / b
    assert isinstance(c, ttnn.Tensor)
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 3.0)

    # Floor division
    d = a // b
    assert torch.allclose(d.to_torch(), torch.ones(4, 4) * 3.0)


def test_tensor_binary_mod_pow():
    """Test modulo and power operations."""
    a = ttnn.Tensor(torch.ones(4, 4) * 7.0)
    b = ttnn.Tensor(torch.ones(4, 4) * 3.0)

    # Modulo
    c = a % b
    assert isinstance(c, ttnn.Tensor)
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 1.0)

    # Power
    d = ttnn.Tensor(torch.ones(4, 4) * 2.0)
    e = d**b
    assert torch.allclose(e.to_torch(), torch.ones(4, 4) * 8.0)


def test_tensor_matmul():
    """Test matrix multiplication."""
    a = ttnn.Tensor(torch.ones(4, 3) * 2.0)
    b = ttnn.Tensor(torch.ones(3, 5) * 3.0)

    c = a @ b
    assert isinstance(c, ttnn.Tensor)
    assert c.shape == (4, 5)
    # 2.0 * 3.0 * 3 (sum across dimension) = 18.0
    assert torch.allclose(c.to_torch(), torch.ones(4, 5) * 18.0)


def test_tensor_reverse_operations():
    """Test reverse binary operations (when left operand is not a Tensor)."""
    a = ttnn.Tensor(torch.ones(4, 4) * 2.0)

    # Reverse add
    b = 5.0 + a
    assert isinstance(b, ttnn.Tensor)
    assert torch.allclose(b.to_torch(), torch.ones(4, 4) * 7.0)

    # Reverse sub
    c = 10.0 - a
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 8.0)

    # Reverse mul
    d = 3.0 * a
    assert torch.allclose(d.to_torch(), torch.ones(4, 4) * 6.0)

    # Reverse div
    e = 10.0 / a
    assert torch.allclose(e.to_torch(), torch.ones(4, 4) * 5.0)


def test_tensor_binary_ops_reject_torch_tensor():
    """Test that binary operations reject torch.Tensor operands."""
    a = ttnn.Tensor(torch.ones(4, 4))
    b = torch.ones(4, 4) * 2.0

    # Should reject torch.Tensor
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = a + b  # type: ignore[operator]

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = a - b  # type: ignore[operator]

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = a * b  # type: ignore[operator]

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = a / b  # type: ignore[operator]

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = a @ b  # type: ignore[operator]


# ---- multiply function tests ----


@requires_ttnn
def test_multiply_basic():
    """Test basic element-wise multiplication."""
    a = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16))
    b = ttnn.from_torch(torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16))

    c = ttnn.multiply(a, b)

    assert isinstance(c, ttnn.Tensor)
    assert c.shape == (2, 2)

    expected = torch.tensor([[5.0, 12.0], [21.0, 32.0]], dtype=torch.bfloat16)
    assert torch.allclose(c.to_torch(), expected, rtol=1e-2)


@requires_ttnn
def test_multiply_same_shape():
    """Test multiply with same-shaped tensors."""
    a = ttnn.from_torch(torch.ones(4, 4, dtype=torch.float32) * 3.0)
    b = ttnn.from_torch(torch.ones(4, 4, dtype=torch.float32) * 7.0)

    c = ttnn.multiply(a, b)

    assert c.shape == (4, 4)
    assert torch.allclose(c.to_torch(), torch.ones(4, 4) * 21.0)


@requires_ttnn
def test_multiply_tile_sized_tensors():
    """Test multiply with tile-sized tensors (32x32)."""
    a = ttnn.rand((32, 32), dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16) * 2.0)

    c = ttnn.multiply(a, b)

    assert c.shape == (32, 32)
    # Result should be a * 2.0
    expected = a.to_torch() * 2.0
    assert torch.allclose(c.to_torch(), expected, rtol=1e-2)


@requires_ttnn
def test_multiply_zeros():
    """Test multiply with zeros."""
    a = ttnn.from_torch(torch.randn(4, 4, dtype=torch.float32))
    b = ttnn.from_torch(torch.zeros(4, 4, dtype=torch.float32))

    c = ttnn.multiply(a, b)

    assert torch.allclose(c.to_torch(), torch.zeros(4, 4))


@requires_ttnn
def test_multiply_ones():
    """Test multiply with ones (identity)."""
    a = ttnn.from_torch(torch.randn(4, 4, dtype=torch.float32))
    b = ttnn.from_torch(torch.ones(4, 4, dtype=torch.float32))

    c = ttnn.multiply(a, b)

    assert torch.allclose(c.to_torch(), a.to_torch())


@requires_ttnn
def test_multiply_negative_values():
    """Test multiply with negative values."""
    a = ttnn.from_torch(torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32))
    b = ttnn.from_torch(torch.tensor([[2.0, -3.0], [4.0, -5.0]], dtype=torch.float32))

    c = ttnn.multiply(a, b)

    expected = torch.tensor([[-2.0, -6.0], [-12.0, -20.0]], dtype=torch.float32)
    assert torch.allclose(c.to_torch(), expected)


@requires_ttnn
def test_multiply_large_tensors():
    """Test multiply with larger tensors."""
    a = ttnn.rand((64, 64), dtype=ttnn.bfloat16)
    b = ttnn.rand((64, 64), dtype=ttnn.bfloat16)

    c = ttnn.multiply(a, b)

    assert c.shape == (64, 64)
    # Verify computation is correct
    expected = a.to_torch() * b.to_torch()
    assert torch.allclose(c.to_torch(), expected, rtol=1e-2)


# ---- Core coordinate classes tests ----


def test_core_coord():
    """Test CoreCoord creation and operations."""
    c1 = ttnn.CoreCoord(3, 5)
    assert c1.x == 3
    assert c1.y == 5

    # Test repr
    assert "CoreCoord(x=3, y=5)" == repr(c1)

    # Test equality
    c2 = ttnn.CoreCoord(3, 5)
    c3 = ttnn.CoreCoord(3, 6)
    assert c1 == c2
    assert c1 != c3

    # Test inequality with non-CoreCoord
    assert c1 != "not a coord"


def test_core_range():
    """Test CoreRange creation and operations."""
    start = ttnn.CoreCoord(0, 0)
    end = ttnn.CoreCoord(2, 3)
    r = ttnn.CoreRange(start, end)

    assert r.start == start
    assert r.end == end

    # Test repr
    repr_str = repr(r)
    assert "CoreRange" in repr_str
    assert "start" in repr_str

    # Test num_cores (3 x 4 grid = 12 cores)
    assert r.num_cores() == 12


def test_core_range_single_node():
    """Test CoreRange with a single core."""
    c = ttnn.CoreCoord(5, 7)
    r = ttnn.CoreRange(c, c)
    assert r.num_cores() == 1


def test_core_range_set():
    """Test CoreRangeSet creation and operations."""
    r1 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
    r2 = ttnn.CoreRange(ttnn.CoreCoord(3, 3), ttnn.CoreCoord(4, 4))

    rs = ttnn.CoreRangeSet([r1, r2])

    # Test ranges accessor
    ranges = rs.ranges()
    assert len(ranges) == 2
    assert ranges[0] == r1
    assert ranges[1] == r2

    # Test num_cores (4 + 4 = 8)
    assert rs.num_cores() == 8

    # Test repr
    assert "CoreRangeSet" in repr(rs)


def test_core_range_set_empty():
    """Test empty CoreRangeSet."""
    rs = ttnn.CoreRangeSet([])
    assert rs.num_cores() == 0
    assert len(rs.ranges()) == 0


# ---- split_work_to_cores tests ----


def test_split_work_evenly_divisible():
    """Test split_work_to_cores with evenly divisible work."""
    grid = ttnn.CoreCoord(4, 4)  # 16 cores
    units = 64  # 64 / 16 = 4 units per core

    num_cores, _all_cores, group1, group2, units1, units2 = ttnn.split_work_to_cores(
        grid, units
    )

    assert num_cores == 16
    assert _all_cores.num_cores() == 16
    assert group1.num_cores() == 16
    assert group2.num_cores() == 0  # No second group needed
    assert units1 == 4
    assert units2 == 0


def test_split_work_with_remainder():
    """Test split_work_to_cores with remainder."""
    grid = ttnn.CoreCoord(4, 4)  # 16 cores
    units = 65  # 65 / 16 = 4 remainder 1

    num_cores, _all_cores, group1, group2, units1, units2 = ttnn.split_work_to_cores(
        grid, units
    )

    assert num_cores == 16
    assert group1.num_cores() == 1  # 1 core gets extra unit
    assert group2.num_cores() == 15  # 15 cores get base units
    assert units1 == 5  # 4 + 1
    assert units2 == 4


def test_split_work_fewer_units_than_cores():
    """Test split_work_to_cores when there are fewer units than cores."""
    grid = ttnn.CoreCoord(8, 8)  # 64 cores
    units = 10  # Only 10 units

    num_cores, _all_cores, group1, group2, units1, units2 = ttnn.split_work_to_cores(
        grid, units
    )

    assert num_cores == 10  # Only use 10 cores
    assert group1.num_cores() == 10
    assert group2.num_cores() == 0
    assert units1 == 1  # Each core gets 1 unit
    assert units2 == 0


def test_split_work_zero_units():
    """Test split_work_to_cores with zero units."""
    grid = ttnn.CoreCoord(4, 4)
    units = 0

    num_cores, _all_cores, group1, group2, units1, units2 = ttnn.split_work_to_cores(
        grid, units
    )

    assert num_cores == 0
    assert _all_cores.num_cores() == 0
    assert group1.num_cores() == 0
    assert group2.num_cores() == 0
    assert units1 == 0
    assert units2 == 0


def test_split_work_row_wise():
    """Test split_work_to_cores with row_wise=True."""
    grid = ttnn.CoreCoord(2, 2)  # 4 cores
    units = 5  # 5 / 4 = 1 remainder 1

    num_cores, _all_cores, group1, group2, units1, units2 = ttnn.split_work_to_cores(
        grid, units, row_wise=True
    )

    assert num_cores == 4
    assert group1.num_cores() == 1
    assert group2.num_cores() == 3
    assert units1 == 2
    assert units2 == 1


def test_split_work_core_range_set_input():
    """Test split_work_to_cores with CoreRangeSet input."""
    r1 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))  # 4 cores
    r2 = ttnn.CoreRange(ttnn.CoreCoord(3, 3), ttnn.CoreCoord(3, 4))  # 2 cores
    crs = ttnn.CoreRangeSet([r1, r2])  # Total 6 cores

    units = 20  # 20 / 6 = 3 remainder 2

    num_cores, _all_cores, group1, group2, units1, units2 = ttnn.split_work_to_cores(
        crs, units
    )

    assert num_cores == 6
    assert group1.num_cores() == 2  # 2 cores get extra unit
    assert group2.num_cores() == 4  # 4 cores get base units
    assert units1 == 4  # 3 + 1
    assert units2 == 3


# ---- Helper functions tests ----


@requires_ttnn
def test_isclose():
    """Test isclose function."""
    a = ttnn.Tensor(torch.tensor([1.0, 2.0, 3.0]))
    b = ttnn.Tensor(torch.tensor([1.0001, 2.0001, 3.0001]))

    # Default tolerances should say they're close
    result = ttnn.isclose(a, b, rtol=1e-3, atol=1e-3)
    assert isinstance(result, ttnn.Tensor)
    assert result.to_torch().all().item()

    # Tighter tolerances should say they're not close
    result2 = ttnn.isclose(a, b, rtol=1e-6, atol=1e-6)
    assert not result2.to_torch().all().item()


@requires_ttnn
def test_isclose_with_nan():
    """Test isclose with NaN values."""
    a = ttnn.Tensor(torch.tensor([1.0, float("nan"), 3.0]))
    b = ttnn.Tensor(torch.tensor([1.0, float("nan"), 3.0]))

    # Without equal_nan, NaNs are not equal
    result1 = ttnn.isclose(a, b, equal_nan=False)
    torch_result = result1.to_torch()
    assert torch_result[0].item() is True
    assert torch_result[1].item() is False  # NaN != NaN
    assert torch_result[2].item() is True

    # With equal_nan, NaNs are equal
    result2 = ttnn.isclose(a, b, equal_nan=True)
    assert result2.to_torch().all().item()


@requires_ttnn
def test_repeat():
    """Test repeat function."""
    a = ttnn.Tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # 2x2

    # Repeat 2x in each dimension (4D vector required)
    b = ttnn.repeat(a, (1, 1, 2, 2))
    assert isinstance(b, ttnn.Tensor)
    assert b.shape == (1, 1, 4, 4)

    # Check pattern
    expected = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 1.0, 2.0],
                    [3.0, 4.0, 3.0, 4.0],
                    [1.0, 2.0, 1.0, 2.0],
                    [3.0, 4.0, 3.0, 4.0],
                ]
            ]
        ]
    )
    assert torch.allclose(b.to_torch(), expected)


@requires_ttnn
def test_repeat_single_dimension():
    """Test repeat with repetition in only one dimension."""
    a = ttnn.Tensor(torch.tensor([[1.0, 2.0]]))  # 1x2

    # Repeat 3x in rows, 1x in columns (4D vector required)
    b = ttnn.repeat(a, (1, 1, 3, 1))
    assert b.shape == (1, 1, 3, 2)

    expected = torch.tensor([[[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]])
    assert torch.allclose(b.to_torch(), expected)


# ---- from_torch tests ----


def test_from_torch_basic_conversion():
    """Test basic tensor conversion from torch to ttnn."""
    t = torch.full((64, 64), 3.0, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t)

    assert tensor.shape == (64, 64)
    assert tensor.dtype == torch.bfloat16
    assert torch.allclose(ttnn.to_torch(tensor), t)


def test_from_torch_dtype_conversion():
    """Test dtype conversion during from_torch."""
    t = torch.randn((32, 32), dtype=torch.float32)
    tensor = ttnn.from_torch(t, dtype=ttnn.bfloat16)

    assert tensor.dtype == torch.bfloat16
    assert tensor.shape == (32, 32)


def test_from_torch_dtype_no_conversion():
    """Test that dtype is preserved when not specified."""
    t = torch.zeros((64, 64), dtype=torch.float32)
    tensor = ttnn.from_torch(t)

    assert tensor.dtype == torch.float32
    assert torch.equal(ttnn.to_torch(tensor), t)


def test_from_torch_various_shapes():
    """Test from_torch with various tensor shapes."""
    shapes = [(32, 32), (64, 64), (128, 128), (256, 256)]

    for shape in shapes:
        t = torch.ones(shape, dtype=torch.bfloat16)
        tensor = ttnn.from_torch(t)
        assert tensor.shape == shape


def test_from_torch_layout_parameter_accepted():
    """Test that layout parameter is accepted (no-op in simulator)."""
    t = torch.randn((64, 64), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT)

    assert tensor.shape == (64, 64)
    assert torch.allclose(ttnn.to_torch(tensor), t)


def test_from_torch_device_parameter_accepted():
    """Test that device parameter is accepted (no-op in simulator)."""
    device = ttnn.open_device(device_id=0)
    t = torch.randn((64, 64), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t, device=device)

    assert tensor.shape == (64, 64)
    ttnn.close_device(device)


def test_from_torch_memory_config_parameter_accepted():
    """Test that memory_config parameter is accepted (no-op in simulator)."""
    t = torch.randn((64, 64), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert tensor.shape == (64, 64)


def test_from_torch_all_parameters():
    """Test from_torch with all parameters specified."""
    device = ttnn.open_device(device_id=0)
    t = torch.full((128, 128), 5.0, dtype=torch.float32)

    tensor = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    assert tensor.shape == (128, 128)
    assert tensor.dtype == torch.bfloat16
    ttnn.close_device(device)


def test_from_torch_roundtrip_conversion():
    """Test that from_torch -> to_torch preserves data."""
    original = torch.randn((64, 64), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(original)
    result = ttnn.to_torch(tensor)

    assert torch.equal(original, result)


def test_from_torch_values_preserved():
    """Test that tensor values are correctly preserved."""
    values = [0.0, 1.0, -1.0, 3.14159, -2.71828]

    for val in values:
        t = torch.full((32, 32), val, dtype=torch.bfloat16)
        tensor = ttnn.from_torch(t)
        result = ttnn.to_torch(tensor)

        assert torch.allclose(result, t, rtol=1e-3)


def test_from_torch_non_contiguous_tensor():
    """Test from_torch with non-contiguous tensor."""
    t = torch.randn((128, 128), dtype=torch.bfloat16)
    t_transposed = t.t()  # Non-contiguous

    tensor = ttnn.from_torch(t_transposed)
    assert tensor.shape == (128, 128)


def test_from_torch_slice_conversion():
    """Test from_torch with tensor slice."""
    t = torch.randn((128, 128), dtype=torch.bfloat16)
    t_slice = t[32:96, 32:96]

    tensor = ttnn.from_torch(t_slice)
    assert tensor.shape == (64, 64)
    assert torch.equal(ttnn.to_torch(tensor), t_slice)


def test_from_torch_dtype_conversion_preserves_values():
    """Test that dtype conversion preserves values within precision limits."""
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    tensor = ttnn.from_torch(t, dtype=ttnn.bfloat16)

    result = ttnn.to_torch(tensor).to(torch.float32)
    assert torch.allclose(result, t, rtol=1e-2)  # bfloat16 has lower precision


def test_from_torch_large_tensor():
    """Test from_torch with larger tensor."""
    t = torch.randn((512, 512), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(t)

    assert tensor.shape == (512, 512)
    assert torch.equal(ttnn.to_torch(tensor), t)


# ---- Golden function wrapper tests ----


@requires_ttnn
def test_golden_function_wrappers_arithmetic():
    """Test dynamically generated golden function wrappers for arithmetic operations."""
    a = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    b = ttnn.from_torch(torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32))

    # Test add
    c = ttnn.add(a, b)
    assert isinstance(c, ttnn.Tensor)
    expected = torch.tensor([[6.0, 8.0], [10.0, 12.0]], dtype=torch.float32)
    assert torch.allclose(c.to_torch(), expected)

    # Test subtract
    d = ttnn.subtract(b, a)
    assert isinstance(d, ttnn.Tensor)
    expected = torch.tensor([[4.0, 4.0], [4.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(d.to_torch(), expected)

    # Test multiply
    e = ttnn.multiply(a, b)
    assert isinstance(e, ttnn.Tensor)
    expected = torch.tensor([[5.0, 12.0], [21.0, 32.0]], dtype=torch.float32)
    assert torch.allclose(e.to_torch(), expected)


@requires_ttnn
def test_golden_function_wrappers_comparisons():
    """Test dynamically generated golden function wrappers for comparison operations."""
    a = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    b = ttnn.from_torch(torch.tensor([[2.0, 2.0], [2.0, 5.0]], dtype=torch.float32))

    # Test eq
    result = ttnn.eq(a, b)
    assert isinstance(result, ttnn.Tensor)
    expected = torch.tensor([[False, True], [False, False]])
    assert torch.equal(result.to_torch(), expected)

    # Test ne
    result = ttnn.ne(a, b)
    expected = torch.tensor([[True, False], [True, True]])
    assert torch.equal(result.to_torch(), expected)

    # Test gt
    result = ttnn.gt(a, b)
    expected = torch.tensor([[False, False], [True, False]])
    assert torch.equal(result.to_torch(), expected)

    # Test lt
    result = ttnn.lt(a, b)
    expected = torch.tensor([[True, False], [False, True]])
    assert torch.equal(result.to_torch(), expected)


@requires_ttnn
def test_golden_function_wrappers_unary():
    """Test dynamically generated golden function wrappers for unary operations."""
    a = ttnn.from_torch(torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float32))

    # Test sqrt
    result = ttnn.sqrt(a)
    assert isinstance(result, ttnn.Tensor)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(result.to_torch(), expected)

    # Test abs (test with negative values)
    b = ttnn.from_torch(torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32))
    result = ttnn.abs(b)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(result.to_torch(), expected)

    # Test exp
    c = ttnn.from_torch(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    result = ttnn.exp(c)
    expected = torch.exp(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(result.to_torch(), expected)


@requires_ttnn
def test_golden_function_wrappers_trigonometric():
    """Test dynamically generated golden function wrappers for trigonometric operations."""
    import math

    a = ttnn.from_torch(
        torch.tensor(
            [[0.0, math.pi / 2], [math.pi, 3 * math.pi / 2]], dtype=torch.float32
        )
    )

    # Test sin
    result = ttnn.sin(a)
    assert isinstance(result, ttnn.Tensor)
    expected = torch.sin(a.to_torch())
    assert torch.allclose(result.to_torch(), expected, atol=1e-6)

    # Test cos
    result = ttnn.cos(a)
    expected = torch.cos(a.to_torch())
    assert torch.allclose(result.to_torch(), expected, atol=1e-6)

    # Test tan
    b = ttnn.from_torch(torch.tensor([[0.0, math.pi / 4]], dtype=torch.float32))
    result = ttnn.tan(b)
    expected = torch.tan(b.to_torch())
    assert torch.allclose(result.to_torch(), expected, atol=1e-6)


@requires_ttnn
def test_golden_function_wrappers_activation():
    """Test dynamically generated golden function wrappers for activation functions."""
    a = ttnn.from_torch(
        torch.tensor([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    )

    # Test relu
    result = ttnn.relu(a)
    assert isinstance(result, ttnn.Tensor)
    expected = torch.tensor([[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(result.to_torch(), expected)

    # Test sigmoid
    result = ttnn.sigmoid(a)
    expected = torch.sigmoid(a.to_torch())
    assert torch.allclose(result.to_torch(), expected)

    # Test gelu
    result = ttnn.gelu(a)
    expected = torch.nn.functional.gelu(a.to_torch())
    assert torch.allclose(result.to_torch(), expected, atol=1e-5)


@requires_ttnn
def test_golden_function_wrappers_logical():
    """Test dynamically generated golden function wrappers for logical operations."""
    a = ttnn.from_torch(torch.tensor([[True, True], [False, False]]))
    b = ttnn.from_torch(torch.tensor([[True, False], [True, False]]))

    # Test logical_and
    result = ttnn.logical_and(a, b)
    assert isinstance(result, ttnn.Tensor)
    expected = torch.tensor([[True, False], [False, False]])
    assert torch.equal(result.to_torch(), expected)

    # Test logical_or
    result = ttnn.logical_or(a, b)
    expected = torch.tensor([[True, True], [True, False]])
    assert torch.equal(result.to_torch(), expected)


class TestTensorTileIndexing:
    """Tests for Tensor tile-coordinate __getitem__ and __setitem__."""

    # --- alignment validation ---

    def test_invalid_size_raises(self) -> None:
        """Tensors not aligned to tile dimensions raise ValueError on any tile access."""
        t = ttnn.Tensor(torch.zeros(30, 30))
        with pytest.raises(ValueError, match="not a multiple of tile dimension"):
            _ = t[slice(0, 1), slice(0, 1)]
        with pytest.raises(ValueError, match="not a multiple of tile dimension"):
            _ = t[0, 0]

    def test_1d_valid(self) -> None:
        """1-D tile-aligned tensors support 1-element tile-coordinate access."""
        t = ttnn.Tensor(torch.arange(64, dtype=torch.float32))
        # Single-element key selects first 32-element tile
        tile0 = t[slice(0, 1)]
        assert tile0.shape == (32,)
        assert torch.allclose(tile0.to_torch(), torch.arange(32, dtype=torch.float32))
        # Second tile
        tile1 = t[slice(1, 2)]
        assert tile1.shape == (32,)
        assert torch.allclose(
            tile1.to_torch(), torch.arange(32, 64, dtype=torch.float32)
        )

    # --- slice format validation ---

    def test_slice_none_start_raises(self) -> None:
        t = ttnn.Tensor(torch.zeros(64, 64))
        with pytest.raises(ValueError, match="must have explicit start value"):
            _ = t[slice(None, 1), slice(0, 1)]

    def test_slice_none_stop_raises(self) -> None:
        t = ttnn.Tensor(torch.zeros(64, 64))
        with pytest.raises(ValueError, match="must have explicit stop value"):
            _ = t[slice(0, None), slice(0, 1)]

    def test_slice_with_step_raises(self) -> None:
        t = ttnn.Tensor(torch.zeros(64, 64))
        with pytest.raises(ValueError, match="must not have a step value"):
            _ = t[slice(0, 1, 1), slice(0, 1)]

    # --- single-tile integer indexing ---

    def test_integer_pair_reads_single_tile(self) -> None:
        raw = torch.zeros(64, 64)
        raw[0:32, 32:64] = 1.0  # tile (0, 1)
        t = ttnn.Tensor(raw)
        tile = t[0, 1]
        assert tile.shape == (32, 32)
        assert torch.all(tile.to_torch() == 1.0)

    def test_integer_pair_writes_single_tile(self) -> None:
        raw = torch.zeros(64, 64)
        t = ttnn.Tensor(raw)
        t[1, 0] = ttnn.Tensor(torch.full((32, 32), 7.0))
        assert torch.all(raw[32:64, 0:32] == 7.0)
        assert torch.all(raw[0:32, :] == 0.0)  # other tiles unchanged

    # --- slice indexing ---

    def test_slice_reads_tile_region(self) -> None:
        raw = torch.zeros(128, 128)
        raw[0:32, :] = 1.0  # first tile row
        t = ttnn.Tensor(raw)
        row = t[slice(0, 1), slice(0, 4)]
        assert row.shape == (32, 128)
        assert torch.all(row.to_torch() == 1.0)

    def test_slice_writes_tile_region(self) -> None:
        raw = torch.zeros(64, 64)
        t = ttnn.Tensor(raw)
        t[slice(0, 1), slice(0, 2)] = ttnn.Tensor(torch.full((32, 64), 3.0))
        assert torch.all(raw[0:32, 0:64] == 3.0)
        assert torch.all(raw[32:64, :] == 0.0)

    # --- integer index preserves 2D shape ---

    def test_int_row_with_slice_col_preserves_2d(self) -> None:
        raw = torch.randn(128, 64)
        t = ttnn.Tensor(raw)
        result = t[0, slice(0, 2)]
        assert result.shape == (32, 64)
        assert torch.allclose(result.to_torch(), raw[0:32, 0:64])

    def test_int_col_with_slice_row_preserves_2d(self) -> None:
        raw = torch.randn(128, 64)
        t = ttnn.Tensor(raw)
        result = t[slice(0, 2), 0]
        assert result.shape == (64, 32)
        assert torch.allclose(result.to_torch(), raw[0:64, 0:32])

    # --- N-D keys with mixed int/slice on last two dims ---

    def test_nd_mixed_key_reads_tile_region(self) -> None:
        """Batch dim (int) + slice tile-row + int tile-col is valid tile indexing."""
        raw = torch.zeros(2, 128, 64)
        raw[1, 0:32, 32:64] = 5.0  # batch=1, tile-row=0, tile-col=1
        t = ttnn.Tensor(raw)
        # (batch=1, tile-row slice 0:1, tile-col 1) → element [1, 0:32, 32:64]
        result = t[1, slice(0, 1), 1]
        assert result.shape == (32, 32)
        assert torch.all(result.to_torch() == 5.0)

    def test_nd_mixed_key_writes_tile_region(self) -> None:
        """Batch dim (int) + int tile-row + slice tile-col writes correctly."""
        raw = torch.zeros(3, 64, 128)
        t = ttnn.Tensor(raw)
        # (batch=2, tile-row 1, tile-col slice 0:2) → element [2, 32:64, 0:64]
        t[2, 1, slice(0, 2)] = ttnn.Tensor(torch.full((32, 64), 9.0))
        assert torch.all(raw[2, 32:64, 0:64] == 9.0)
        assert torch.all(raw[0] == 0.0)  # other batches unchanged

    # --- degenerate (size-1) dimensions ---

    def test_degenerate_dim_allowed(self) -> None:
        raw = torch.randn(32, 1)
        t = ttnn.Tensor(raw)
        tile = t[0, 0]
        assert tile.shape == (32, 1)
        assert torch.allclose(tile.to_torch(), raw)
