# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch

from sim import ttnn, TTNN_AVAILABLE
from sim.sharding import (
    count_local_remote_l1_dram,
    count_local_remote_l1_dram_for_getitem,
    shard_origin_from_key,
)
from sim.ttnnsim import (
    CoreGrid,
    MemoryConfig,
    NdShardSpec,
    ShardDistributionStrategy,
    ShardOrientation,
    ShardSpec,
    ShardStrategy,
    ShardingStrategy,
    TensorMemoryLayout,
    TensorSpec,
)


def requires_ttnn(func: Callable[..., Any]) -> Callable[..., Any]:
    wrapped = pytest.mark.skipif(
        not TTNN_AVAILABLE,
        reason="ttnn not available (required for golden function tests)",
    )(func)
    return pytest.mark.requires_ttnn(wrapped)


def test_constants_and_dtypes():
    assert isinstance(ttnn.TILE_SIZE, int)
    assert ttnn.TILE_SIZE > 0
    assert hasattr(ttnn, "TILE_LAYOUT")
    assert torch.tensor([], dtype=ttnn.bfloat16).element_size() == 2
    assert ttnn.float32 == torch.float32
    assert hasattr(ttnn, "bfloat8_b")
    assert ttnn.bfloat8_b == ttnn.bfloat8_b
    assert ttnn.bfloat8_b != ttnn.bfloat16
    assert ttnn.bfloat8_b != torch.float32
    assert ttnn.bfloat8_b.element_size == 1
    t_bf8 = ttnn.rand((32, 32), dtype=ttnn.bfloat8_b)
    assert t_bf8.dtype == ttnn.bfloat8_b
    assert t_bf8.underlying_dtype == torch.float32


def test_bfloat8_b_capacity_bytes_statistics():
    """capacity_bytes for bfloat8_b accounts for the BFP8B shared-exponent overhead.

    BFP8B encodes n elements as n mantissa bytes plus one exponent byte per
    group of 16 elements: size_in_bytes(n) = n + ceil(n / 16).

    For a buffer with BLOCK_COUNT blocks of one 32x32 tile each:
      total_elements = BLOCK_COUNT * 32 * 32 = 4096
      float32:   4096 * 4                 = 16384 bytes
      bfloat8_b: 4096 + ceil(4096 / 16)  = 4352 bytes  (4096 mantissa + 256 exponent)
    """
    import math as _math
    from python.sim.dfb import DataflowBuffer

    BLOCK_COUNT = 4
    TILE_SHAPE = (1, 1)
    TOTAL_ELEMENTS = BLOCK_COUNT * 32 * 32  # 4096

    f32_tensor = ttnn.rand((32, 32), dtype=ttnn.float32)
    bf8_tensor = ttnn.rand((32, 32), dtype=ttnn.bfloat8_b)

    assert f32_tensor.element_size == 4
    assert bf8_tensor.element_size == 1  # mantissa only; exponent overhead is per-group

    f32_dfb = DataflowBuffer(
        likeness_tensor=f32_tensor, shape=TILE_SHAPE, block_count=BLOCK_COUNT
    )
    bf8_dfb = DataflowBuffer(
        likeness_tensor=bf8_tensor, shape=TILE_SHAPE, block_count=BLOCK_COUNT
    )

    expected_f32 = TOTAL_ELEMENTS * 4  # 16384
    expected_bf8 = TOTAL_ELEMENTS + _math.ceil(TOTAL_ELEMENTS / 16)  # 4352

    assert f32_dfb.capacity_bytes == expected_f32
    assert bf8_dfb.capacity_bytes == expected_bf8

    # Also verify size_in_bytes is accessible directly on the tensor
    assert bf8_tensor.size_in_bytes(TOTAL_ELEMENTS) == expected_bf8
    assert f32_tensor.size_in_bytes(TOTAL_ELEMENTS) == expected_f32

    # Partial groups: 15 elements still require 1 exponent byte (ceiling division).
    # Floor division would wrongly return 15 + 0 = 15.
    assert ttnn.bfloat8_b.size_in_bytes(15) == 15 + 1
    assert ttnn.bfloat8_b.size_in_bytes(16) == 16 + 1
    assert ttnn.bfloat8_b.size_in_bytes(17) == 17 + 2


def test_bfloat8_b_promoted_to_float32_by_default():
    """bfloat8_b tensors use float32 backing when float32 promotion is active."""
    t_rand = ttnn.rand((32, 32), dtype=ttnn.bfloat8_b)
    t_empty = ttnn.empty((32, 32), dtype=ttnn.bfloat8_b)
    t_from = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat8_b)

    for t in (t_rand, t_empty, t_from):
        assert t.dtype == ttnn.bfloat8_b, "declared dtype must remain bfloat8_b"
        assert (
            t.underlying_dtype == torch.float32
        ), "backing dtype must be float32 when promotion is active"


def test_bfloat8_b_no_promotion_uses_bfloat16_backing():
    """bfloat8_b tensors use bfloat16 backing when float32 promotion is disabled."""
    ttnn.set_disable_float32_promotion(True)
    try:
        t_rand = ttnn.rand((32, 32), dtype=ttnn.bfloat8_b)
        t_empty = ttnn.empty((32, 32), dtype=ttnn.bfloat8_b)
        t_from = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat8_b)

        for t in (t_rand, t_empty, t_from):
            assert t.dtype == ttnn.bfloat8_b, "declared dtype must remain bfloat8_b"
            assert (
                t.underlying_dtype == ttnn.bfloat16
            ), "backing dtype must be bfloat16 when promotion is disabled"
    finally:
        ttnn.set_disable_float32_promotion(False)


def test_bfloat8_b_promotion_restored_after_reenable():
    """Re-enabling float32 promotion restores float32 backing for bfloat8_b."""
    ttnn.set_disable_float32_promotion(True)
    ttnn.set_disable_float32_promotion(False)

    t = ttnn.rand((32, 32), dtype=ttnn.bfloat8_b)
    assert t.underlying_dtype == torch.float32


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
    shape = (32, 64)
    t1 = ttnn.rand(shape, dtype=ttnn.float32)
    assert isinstance(t1, ttnn.Tensor)
    assert t1.shape == shape
    assert t1.dtype == torch.float32

    t2 = ttnn.empty(shape, dtype=ttnn.bfloat16)
    assert isinstance(t2, ttnn.Tensor)
    assert t2.shape == shape
    assert t2.dtype == ttnn.bfloat16

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
#
# Layout convention for the arithmetic / golden-wrapper tests below:
#
#   - These tests exist to exercise the ``ttnn`` shim's elementwise math, not
#     tile layout.  Their inputs are small ad-hoc tensors (e.g. ``(2, 2)`` or
#     ``(4, 4)``) chosen for readability of the expected values, not for tile
#     alignment.
#   - Under ``TILE_LAYOUT`` the shim auto-pads such inputs to ``(32, 32)``
#     (per ``_pad_to_tile_alignment`` in ``ttnnsim.py``), which would muddy
#     the exact-shape assertions used here.  We therefore pass
#     ``layout=ttnn.ROW_MAJOR_LAYOUT`` explicitly so the original shape is
#     round-trip-preserved.
#   - Coverage for the tile-layout shim behaviour (auto-pad + arithmetic on
#     padded inputs) lives in the ``test_tile_layout_shim_*`` tests further
#     down, alongside the ``_pad_to_tile_alignment`` tests.


@requires_ttnn
def test_multiply_basic():
    """Test basic element-wise multiplication.

    Uses ROW_MAJOR_LAYOUT so the (2, 2) shape is round-trip-preserved; the
    purpose here is to exercise the shim's multiply, not tile semantics.
    """
    a = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.from_torch(
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    c = ttnn.multiply(a, b)

    assert isinstance(c, ttnn.Tensor)
    assert c.shape == (2, 2)

    expected = torch.tensor([[5.0, 12.0], [21.0, 32.0]], dtype=torch.bfloat16)
    assert torch.allclose(c.to_torch(), expected, rtol=1e-2)


@requires_ttnn
def test_multiply_same_shape():
    """Test multiply with same-shaped tensors."""
    a = ttnn.from_torch(
        torch.ones(4, 4, dtype=torch.float32) * 3.0, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    b = ttnn.from_torch(
        torch.ones(4, 4, dtype=torch.float32) * 7.0, layout=ttnn.ROW_MAJOR_LAYOUT
    )

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
    a = ttnn.from_torch(
        torch.randn(4, 4, dtype=torch.float32), layout=ttnn.ROW_MAJOR_LAYOUT
    )
    b = ttnn.from_torch(
        torch.zeros(4, 4, dtype=torch.float32), layout=ttnn.ROW_MAJOR_LAYOUT
    )

    c = ttnn.multiply(a, b)

    assert torch.allclose(c.to_torch(), torch.zeros(4, 4))


@requires_ttnn
def test_multiply_ones():
    """Test multiply with ones (identity)."""
    a = ttnn.from_torch(
        torch.randn(4, 4, dtype=torch.float32), layout=ttnn.ROW_MAJOR_LAYOUT
    )
    b = ttnn.from_torch(
        torch.ones(4, 4, dtype=torch.float32), layout=ttnn.ROW_MAJOR_LAYOUT
    )

    c = ttnn.multiply(a, b)

    assert torch.allclose(c.to_torch(), a.to_torch())


@requires_ttnn
def test_multiply_negative_values():
    """Test multiply with negative values."""
    a = ttnn.from_torch(
        torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.from_torch(
        torch.tensor([[2.0, -3.0], [4.0, -5.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

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

    # Test repr (positional, tt-metal style)
    assert repr(c1) == "CoreCoord(3, 5)"

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
    assert "CoreCoord(0, 0)" in repr_str
    assert "CoreCoord(2, 3)" in repr_str

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

    assert tensor.dtype == ttnn.bfloat16
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


def test_from_torch_tile_layout_pads_non_tile_aligned_shapes():
    """Regression test for issue #601: a TILE_LAYOUT tensor whose last two
    dims are not multiples of TILE_SHAPE is auto-padded with zeros so every
    tile is exactly 32x32 as the spec requires.  Per spec, ``(N, 1)`` column
    vectors live in column 0 of each tile, ``(1, M)`` row vectors in row 0,
    ``(1, 1)`` scalars at position ``(0, 0)``; broadcast / reduce then
    overwrite the padding (steps 1 and 2 respectively) when needed.
    """
    col = ttnn.from_torch(torch.arange(32, dtype=torch.float32).reshape(32, 1))
    assert col.shape == (32, 32)
    assert torch.equal(col.to_torch()[:, 0], torch.arange(32, dtype=torch.float32))
    assert torch.all(col.to_torch()[:, 1:] == 0)

    row = ttnn.from_torch(torch.arange(32, dtype=torch.float32).reshape(1, 32))
    assert row.shape == (32, 32)
    assert torch.equal(row.to_torch()[0, :], torch.arange(32, dtype=torch.float32))
    assert torch.all(row.to_torch()[1:, :] == 0)

    scalar = ttnn.from_torch(torch.tensor([[3.5]], dtype=torch.float32))
    assert scalar.shape == (32, 32)
    assert scalar.to_torch()[0, 0].item() == 3.5
    assert torch.all(scalar.to_torch()[1:, :] == 0)
    assert torch.all(scalar.to_torch()[:, 1:] == 0)

    odd = ttnn.from_torch(torch.ones((4, 4), dtype=torch.float32))
    assert odd.shape == (32, 32)
    assert torch.all(odd.to_torch()[:4, :4] == 1.0)
    assert torch.all(odd.to_torch()[4:, :] == 0)
    assert torch.all(odd.to_torch()[:, 4:] == 0)

    # 1-D tensors still cannot be tile-laid out (no second tile dim to pad).
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        ttnn.from_torch(torch.ones((32,)), layout=ttnn.TILE_LAYOUT)

    # rand / empty take a shape directly; they pad transparently too.
    assert ttnn.rand((32, 1)).shape == (32, 32)
    assert ttnn.empty((4, 4)).shape == (32, 32)

    # Row-major preserves the original shape exactly.
    assert ttnn.from_torch(torch.ones((32, 1)), layout=ttnn.ROW_MAJOR_LAYOUT).shape == (
        32,
        1,
    )
    assert ttnn.from_torch(torch.ones((4, 4)), layout=ttnn.ROW_MAJOR_LAYOUT).shape == (
        4,
        4,
    )


# ---- TILE_LAYOUT shim behaviour ----
#
# These tests pin down the auto-pad contract on the multi-dimensional and
# batched inputs that the basic ``test_from_torch_tile_layout_pads_*`` test
# above does not exercise, and verify that elementwise arithmetic in the shim
# operates correctly on the padded tiles (i.e. the padding zeros do not
# corrupt the logical values in the top-left corner).  The shim's auto-pad
# implementation (``_pad_to_tile_alignment``) uses
# ``torch.nn.functional.pad`` and is documented to touch only the last two
# dims; these tests make that contract explicit.


def test_pad_to_tile_alignment_3d_last_two_unaligned():
    """3-D input with non-aligned last two dims pads each slice independently.

    ``(2, 5, 7)`` -> ``(2, 32, 32)``: the original 5x7 data lives in the
    top-left of each batch slice; the remaining 32x32 - 5x7 cells per slice
    are zero.
    """
    src = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    t = ttnn.from_torch(src)
    assert t.shape == (2, 32, 32)
    out = t.to_torch()
    assert torch.equal(out[:, :5, :7], src)
    assert torch.all(out[:, 5:, :] == 0)
    assert torch.all(out[:, :, 7:] == 0)


def test_pad_to_tile_alignment_3d_column_vector_per_slice():
    """3-D ``(B, N, 1)`` pads to ``(B, 32, 32)`` with data in column 0.

    Mirrors the 2-D column-vector convention on each batch slice.
    """
    src = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3, 1)
    t = ttnn.from_torch(src)
    assert t.shape == (2, 32, 32)
    out = t.to_torch()
    assert torch.equal(out[:, :3, 0:1], src)
    assert torch.all(out[:, :, 1:] == 0)
    assert torch.all(out[:, 3:, :] == 0)


def test_pad_to_tile_alignment_3d_row_vector_per_slice():
    """3-D ``(B, 1, M)`` pads to ``(B, 32, 32)`` with data in row 0."""
    src = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 1, 3)
    t = ttnn.from_torch(src)
    assert t.shape == (2, 32, 32)
    out = t.to_torch()
    assert torch.equal(out[:, 0:1, :3], src)
    assert torch.all(out[:, 1:, :] == 0)
    assert torch.all(out[:, :, 3:] == 0)


def test_pad_to_tile_alignment_4d_pads_only_last_two_dims():
    """4-D ``(B0, B1, H, W)`` pads only the last two dims.

    Pins down the documented contract of ``_pad_to_tile_alignment`` /
    ``torch.nn.functional.pad`` - leading batch dims are preserved untouched.
    """
    src = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    t = ttnn.from_torch(src)
    assert t.shape == (2, 3, 32, 32)
    out = t.to_torch()
    assert torch.equal(out[:, :, :5, :7], src)
    assert torch.all(out[:, :, 5:, :] == 0)
    assert torch.all(out[:, :, :, 7:] == 0)


def test_pad_to_tile_alignment_already_aligned_is_identity():
    """Already tile-aligned tensors round-trip unchanged.

    Covers 2-D and 3-D inputs whose last two dims are already multiples of
    ``TILE_SHAPE``; ``_pad_to_tile_alignment`` should be a no-op in this
    case.
    """
    src_2d = torch.randn((64, 96), dtype=torch.float32)
    t_2d = ttnn.from_torch(src_2d)
    assert t_2d.shape == (64, 96)
    assert torch.equal(t_2d.to_torch(), src_2d)

    src_3d = torch.randn((2, 32, 32), dtype=torch.float32)
    t_3d = ttnn.from_torch(src_3d)
    assert t_3d.shape == (2, 32, 32)
    assert torch.equal(t_3d.to_torch(), src_3d)


@requires_ttnn
def test_tile_layout_shim_multiply_column_vectors():
    """Multiplying two ``(32, 1)`` column vectors under ``TILE_LAYOUT``.

    Both inputs auto-pad to ``(32, 32)`` with data in column 0 and zeros
    elsewhere.  Elementwise multiply preserves that placement: column 0 of
    the output carries the elementwise products, and columns 1..31 stay
    zero (zero times anything is zero).
    """
    a_src = torch.arange(1, 33, dtype=torch.float32).reshape(32, 1)
    b_src = torch.arange(33, 65, dtype=torch.float32).reshape(32, 1)
    a = ttnn.from_torch(a_src)
    b = ttnn.from_torch(b_src)
    assert a.shape == b.shape == (32, 32)

    c = ttnn.multiply(a, b)
    assert c.shape == (32, 32)
    out = c.to_torch()
    assert torch.equal(out[:, 0:1], a_src * b_src)
    assert torch.all(out[:, 1:] == 0)


@requires_ttnn
def test_tile_layout_shim_add_row_vectors():
    """Adding two ``(1, 32)`` row vectors under ``TILE_LAYOUT``.

    Both inputs auto-pad to ``(32, 32)`` with data in row 0 and zeros
    elsewhere.  Elementwise add preserves placement: row 0 holds the sum,
    rows 1..31 stay zero (zero plus zero is zero).
    """
    a_src = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
    b_src = torch.arange(101, 133, dtype=torch.float32).reshape(1, 32)
    a = ttnn.from_torch(a_src)
    b = ttnn.from_torch(b_src)
    assert a.shape == b.shape == (32, 32)

    c = ttnn.add(a, b)
    assert c.shape == (32, 32)
    out = c.to_torch()
    assert torch.equal(out[0:1, :], a_src + b_src)
    assert torch.all(out[1:, :] == 0)


@requires_ttnn
def test_tile_layout_shim_multiply_corner_block():
    """Elementwise multiply of two ``(4, 4)`` inputs under ``TILE_LAYOUT``.

    Both inputs auto-pad to ``(32, 32)`` with the 4x4 source data in the
    top-left corner and zeros elsewhere.  Elementwise multiply therefore
    produces a tile whose top-left 4x4 block is the elementwise product
    and the rest is zero.
    """
    a_src = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=torch.float32,
    )
    b_src = torch.full((4, 4), 2.0, dtype=torch.float32)
    a = ttnn.from_torch(a_src)
    b = ttnn.from_torch(b_src)
    assert a.shape == b.shape == (32, 32)

    c = ttnn.multiply(a, b)
    out = c.to_torch()
    assert torch.equal(out[:4, :4], a_src * b_src)
    assert torch.all(out[4:, :] == 0)
    assert torch.all(out[:, 4:] == 0)


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
    assert tensor.dtype == ttnn.bfloat16
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
    tensor = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

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
    a = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.from_torch(
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

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
    a = ttnn.from_torch(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.from_torch(
        torch.tensor([[2.0, 2.0], [2.0, 5.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

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
    a = ttnn.from_torch(
        torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Test sqrt
    result = ttnn.sqrt(a)
    assert isinstance(result, ttnn.Tensor)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(result.to_torch(), expected)

    # Test abs (test with negative values)
    b = ttnn.from_torch(
        torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    result = ttnn.abs(b)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert torch.allclose(result.to_torch(), expected)

    # Test exp
    c = ttnn.from_torch(
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
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
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
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
    b = ttnn.from_torch(
        torch.tensor([[0.0, math.pi / 4]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    result = ttnn.tan(b)
    expected = torch.tan(b.to_torch())
    assert torch.allclose(result.to_torch(), expected, atol=1e-6)


@requires_ttnn
def test_golden_function_wrappers_activation():
    """Test dynamically generated golden function wrappers for activation functions."""
    a = ttnn.from_torch(
        torch.tensor([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
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
    a = ttnn.from_torch(
        torch.tensor([[True, True], [False, False]]),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    b = ttnn.from_torch(
        torch.tensor([[True, False], [True, False]]),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

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
        """Batch dim (int) + slice tile-row + int tile-col is valid tile indexing.

        Integer batch indices are normalized to unit slices so the batch
        dimension is preserved in the output (shape includes a leading 1).
        """
        raw = torch.zeros(2, 128, 64)
        raw[1, 0:32, 32:64] = 5.0  # batch=1, tile-row=0, tile-col=1
        t = ttnn.Tensor(raw)
        # (batch=1, tile-row slice 0:1, tile-col 1) → element [1:2, 0:32, 32:64]
        # batch integer index is normalized to a unit slice, preserving the dimension.
        result = t[1, slice(0, 1), 1]
        assert result.shape == (1, 32, 32)
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


class TestShardingTypes:
    """Tests for ShardingStrategy, ShardSpec, NdShardSpec, and MemoryConfig data types.

    ``shard_shape`` tuples below are **element** extents (tt-metal style), not
    tile-grid dimensions.
    """

    def test_sharding_strategy_values(self) -> None:
        """All sharding strategies are defined."""
        assert ShardingStrategy.INTERLEAVED
        assert ShardingStrategy.HEIGHT_SHARDED
        assert ShardingStrategy.WIDTH_SHARDED
        assert ShardingStrategy.BLOCK_SHARDED
        assert ShardingStrategy.ND_SHARDED

    def test_shard_spec_creation(self) -> None:
        """ShardSpec stores shard_grid and per-shard element shape."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(2, 8))
        assert spec.shard_grid == (4,)
        assert spec.shard_shape == (2, 8)

    def test_nd_shard_spec_creation(self) -> None:
        """NdShardSpec stores shard_shape, optional shard_grid, and distribution."""
        spec = NdShardSpec(
            shard_shape=(2, 2),
            shard_grid=(2, 4),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        assert spec.shard_grid == (2, 4)
        assert spec.shard_shape == (2, 2)
        assert spec.distribution == ShardDistributionStrategy.GRID_2D

    def test_nd_shard_spec_default_distribution(self) -> None:
        """NdShardSpec defaults to ROUND_ROBIN_1D (matches tt-metal ``NdShardSpec`` binding)."""
        spec = NdShardSpec(shard_shape=(1, 1), shard_grid=(4, 4))
        assert spec.distribution == ShardDistributionStrategy.ROUND_ROBIN_1D

    def test_memory_config_interleaved(self) -> None:
        """MemoryConfig without shard_spec defaults to INTERLEAVED."""
        mc = MemoryConfig(strategy=ShardingStrategy.INTERLEAVED)
        assert mc.strategy == ShardingStrategy.INTERLEAVED
        assert mc.shard_spec is None

    def test_memory_config_sharded(self) -> None:
        """MemoryConfig accepts a ShardSpec for sharded strategies."""
        spec = ShardSpec(shard_grid=(2, 4), shard_shape=(2, 2))
        mc = MemoryConfig(strategy=ShardingStrategy.BLOCK_SHARDED, shard_spec=spec)
        assert mc.strategy == ShardingStrategy.BLOCK_SHARDED
        assert mc.shard_spec is spec

    def test_memory_config_nd_sharded(self) -> None:
        """MemoryConfig accepts an NdShardSpec for ND_SHARDED strategy."""
        spec = NdShardSpec(
            shard_shape=(2, 2),
            shard_grid=(2, 4),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        assert mc.strategy == ShardingStrategy.ND_SHARDED
        assert mc.nd_shard_spec is spec
        assert mc.shard_spec is None

    def test_shard_strategy_values(self) -> None:
        """ShardStrategy exposes HEIGHT, WIDTH, and BLOCK."""
        assert ShardStrategy.HEIGHT
        assert ShardStrategy.WIDTH
        assert ShardStrategy.BLOCK

    def test_shard_orientation_values(self) -> None:
        """ShardOrientation exposes ROW_MAJOR and COL_MAJOR."""
        assert ShardOrientation.ROW_MAJOR
        assert ShardOrientation.COL_MAJOR

    def test_shard_spec_stores_orientation(self) -> None:
        """ShardSpec stores orientation and defaults to ROW_MAJOR."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(2, 8))
        assert spec.orientation == ShardOrientation.ROW_MAJOR
        spec_col = ShardSpec(
            shard_grid=(4,),
            shard_shape=(2, 8),
            orientation=ShardOrientation.COL_MAJOR,
        )
        assert spec_col.orientation == ShardOrientation.COL_MAJOR

    def test_core_grid_creation(self) -> None:
        """CoreGrid stores y, x, and exposes num_cores."""
        grid = CoreGrid(y=4, x=8)
        assert grid.y == 4
        assert grid.x == 8
        assert grid.num_cores == 32

    def test_predefined_constants(self) -> None:
        """DRAM_MEMORY_CONFIG and L1_MEMORY_CONFIG are MemoryConfig instances."""
        assert isinstance(ttnn.DRAM_MEMORY_CONFIG, MemoryConfig)
        assert isinstance(ttnn.L1_MEMORY_CONFIG, MemoryConfig)
        assert ttnn.DRAM_MEMORY_CONFIG.strategy == ShardingStrategy.INTERLEAVED
        assert ttnn.L1_MEMORY_CONFIG.strategy == ShardingStrategy.INTERLEAVED

    def test_tensor_spec_nd_sharded_matches_tech_report_inputs(self) -> None:
        """TensorSpec.nd_sharded(shard_shape, core_ranges) sets ND shard_shape."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3)),
            }
        )
        spec = TensorSpec(
            shape=(2, 4, 256, 512),
            dtype=torch.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).nd_sharded((1, 1, 64, 128), core_ranges)
        assert spec.memory_layout == TensorMemoryLayout.ND_SHARDED
        assert spec.memory_config.nd_shard_spec is not None
        assert spec.memory_config.nd_shard_spec.shard_shape == (1, 1, 64, 128)

    def test_tensor_spec_nd_sharded_requires_divisible_dims(self) -> None:
        """from_torch raises when shard_shape does not divide tensor shape."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3)),
            }
        )
        spec = TensorSpec(
            shape=(2, 4, 256, 512),
            dtype=torch.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).nd_sharded((1, 1, 63, 128), core_ranges)
        with pytest.raises(ValueError, match="not divisible"):
            ttnn.from_torch(
                torch.randn(2, 4, 256, 512),
                spec=spec,
                device=ttnn.open_device(0),
            )


class TestTensorMemoryConfig:
    """Tests for Tensor.memory_config attribute and related behaviour."""

    def test_tensor_default_memory_config_is_dram(self) -> None:
        """A plain Tensor defaults to DRAM_MEMORY_CONFIG."""
        t = ttnn.Tensor(torch.zeros(64, 64))
        assert t.memory_config is ttnn.DRAM_MEMORY_CONFIG

    def test_tensor_with_memory_config(self) -> None:
        """Tensor stores the MemoryConfig passed at construction."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(2, 4))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(256, 128), memory_config=mc)
        assert t.memory_config is mc

    def test_from_torch_propagates_memory_config(self) -> None:
        """from_torch attaches the given MemoryConfig to the returned Tensor."""
        spec = ShardSpec(shard_grid=(2,), shard_shape=(1, 4))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        t = ttnn.from_torch(torch.zeros(64, 128), memory_config=mc)
        assert t.memory_config is mc

    def test_getitem_propagates_memory_config(self) -> None:
        """Slicing a sharded Tensor propagates memory_config to the result."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(2, 4))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(256, 128), memory_config=mc)
        sliced = t[0:2, 0:4]
        assert sliced.memory_config is mc

    def test_nd_sharded_propagated_through_getitem(self) -> None:
        """Slicing an ND_SHARDED Tensor propagates memory_config."""
        spec = NdShardSpec(
            shard_shape=(64, 64),
            shard_grid=(2, 4),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(128, 256), memory_config=mc)
        sliced = t[0:2, 0:2]
        assert sliced.memory_config is mc


class TestCreateShardedMemoryConfig:
    """Tests for create_sharded_memory_config factory function."""

    def test_height_sharded(self) -> None:
        """HEIGHT strategy: each core owns a horizontal slice."""
        # 4 cores, 128x64 tensor (4x2 tiles), shard = (1, 2) tiles per core
        mc = ttnn.create_sharded_memory_config(
            shape=(128, 64),
            core_grid=CoreGrid(y=2, x=2),
            strategy=ShardStrategy.HEIGHT,
        )
        assert mc.strategy == ShardingStrategy.HEIGHT_SHARDED
        assert mc.shard_spec is not None
        assert mc.shard_spec.shard_grid == (4,)
        assert mc.shard_spec.shard_shape == (32, 64)
        assert mc.shard_spec.orientation == ShardOrientation.ROW_MAJOR

    def test_width_sharded(self) -> None:
        """WIDTH strategy: each core owns a vertical slice."""
        # 4 cores, 64x128 elements (2x4 tiles); shard_shape (64, 32) elements per core
        mc = ttnn.create_sharded_memory_config(
            shape=(64, 128),
            core_grid=CoreGrid(y=2, x=2),
            strategy=ShardStrategy.WIDTH,
        )
        assert mc.strategy == ShardingStrategy.WIDTH_SHARDED
        assert mc.shard_spec is not None
        assert mc.shard_spec.shard_grid == (4,)
        assert mc.shard_spec.shard_shape == (64, 32)

    def test_block_sharded(self) -> None:
        """BLOCK strategy: 2-D core grid, each core owns a rectangular block."""
        # 2x4 core grid, 128x256 elements (4x8 tiles); shard_shape (64, 64) elements per core
        mc = ttnn.create_sharded_memory_config(
            shape=(128, 256),
            core_grid=CoreGrid(y=2, x=4),
            strategy=ShardStrategy.BLOCK,
        )
        assert mc.strategy == ShardingStrategy.BLOCK_SHARDED
        assert mc.shard_spec is not None
        assert mc.shard_spec.shard_grid == (2, 4)
        assert mc.shard_spec.shard_shape == (64, 64)

    def test_use_height_and_width_as_shard_shape(self) -> None:
        """When use_height_and_width_as_shard_shape=True, shape is the shard shape."""
        mc = ttnn.create_sharded_memory_config(
            shape=(64, 32),
            core_grid=CoreGrid(y=2, x=4),
            strategy=ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        assert mc.strategy == ShardingStrategy.BLOCK_SHARDED
        assert mc.shard_spec is not None
        assert mc.shard_spec.shard_shape == (64, 32)

    def test_orientation_stored(self) -> None:
        """Orientation is stored in the resulting ShardSpec."""
        mc = ttnn.create_sharded_memory_config(
            shape=(128, 64),
            core_grid=CoreGrid(y=2, x=2),
            strategy=ShardStrategy.HEIGHT,
            orientation=ShardOrientation.COL_MAJOR,
        )
        assert mc.shard_spec is not None
        assert mc.shard_spec.orientation == ShardOrientation.COL_MAJOR

    def test_batch_dimensions_compressed_to_2d(self) -> None:
        """Higher-rank tensors are compressed to 2D before shard computation."""
        # (2, 128, 64) -> flat 2D (256, 64) = (8, 2) tiles; 4 cores HEIGHT -> shard_shape (64, 64) elements
        mc = ttnn.create_sharded_memory_config(
            shape=(2, 128, 64),
            core_grid=CoreGrid(y=2, x=2),
            strategy=ShardStrategy.HEIGHT,
        )
        assert mc.shard_spec is not None
        assert mc.shard_spec.shard_shape == (64, 64)


class TestTensorSpecTtnnApi:
    """tt-metal style TensorSpec / CoreRangeSet (tensor sharding tech report)."""

    def test_core_range_set_accepts_set_of_ranges(self) -> None:
        r = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))
        crs = ttnn.CoreRangeSet({r})
        assert crs.num_cores() == 4
        assert crs.ranges() == [r]

    def test_width_sharded_tensor_spec_shard_shape(self) -> None:
        """Width sharding: 512 / 4 = 128 columns per shard; height 64 full."""
        core_ranges = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}
        )
        spec = TensorSpec(
            shape=(1, 64, 512),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).width_sharded(core_ranges)
        assert spec.memory_layout == TensorMemoryLayout.WIDTH_SHARDED
        assert spec.memory_config is not None
        assert spec.memory_config.shard_spec is not None
        assert spec.memory_config.shard_spec.shard_shape == (64, 128)

    def test_from_torch_with_tensor_spec(self) -> None:
        core_ranges = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}
        )
        spec = TensorSpec(
            shape=(1, 64, 512),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).width_sharded(core_ranges)
        torch_tensor = torch.randn(tuple(spec.shape))
        device = ttnn.open_device(0)
        tt_tensor = ttnn.from_torch(torch_tensor, spec=spec, device=device)
        assert tt_tensor.shape == (1, 64, 512)
        assert ttnn.is_sharded(tt_tensor)

    def test_from_torch_rejects_shape_mismatch_with_spec(self) -> None:
        spec = TensorSpec(shape=(2, 64, 512), dtype=torch.float32).width_sharded(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}
            )
        )
        with pytest.raises(ValueError, match="does not match spec.shape"):
            ttnn.from_torch(torch.zeros(1, 64, 512), spec=spec)


class TestTensorShardingTechReportExamples:
    """Examples aligned with the tt-metal tensor sharding tech report.

    https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_sharding/tensor_sharding.md

    Locality uses :mod:`sim.sharding` (element coordinates): a view is **local**
    on a core when its elements lie in that core's shard; otherwise access is
    **remote** on that core.
    """

    @staticmethod
    def _device():
        return ttnn.open_device(0)

    def test_height_sharding_tensor_spec(self) -> None:
        """2D Height Sharding: ``TensorSpec`` + ``height_sharded`` (8 cores, 2x4 grid)."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3)),
            }
        )
        tensor_spec = ttnn.TensorSpec(
            shape=(2, 128, 256),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).height_sharded(core_ranges)
        assert tensor_spec.memory_config.shard_spec is not None
        sp = tensor_spec.memory_config.shard_spec
        assert sp.shard_grid == (8,)
        assert sp.shard_shape == (32, 256)
        torch_tensor = torch.randn(tuple(tensor_spec.shape))
        tt_tensor = ttnn.from_torch(
            torch_tensor, spec=tensor_spec, device=self._device()
        )
        assert tt_tensor.shape == (2, 128, 256)
        assert ttnn.is_sharded(tt_tensor)
        loc0, rem0, _ = count_local_remote_l1_dram(tt_tensor, 0)
        loc7, rem7, _ = count_local_remote_l1_dram(tt_tensor, 7)
        # HEIGHT_SHARDED counts along the last two element dimensions only (batch
        # stacked in the logical height used for shard_shape, not double-counted).
        plane_el = tt_tensor.shape[-2] * tt_tensor.shape[-1]
        assert loc0 + rem0 == plane_el and loc7 + rem7 == plane_el
        shard_hw = sp.shard_shape[-2] * sp.shard_shape[-1]
        assert loc0 == shard_hw
        assert loc7 == 0
        assert rem7 == plane_el
        k_core0_rows = (slice(0, 1), slice(0, 1), slice(0, 8))
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_core0_rows, 0) == (
            shard_hw,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_core0_rows, 1) == (
            0,
            shard_hw,
            0,
        )
        k_core1_rows = (slice(0, 1), slice(1, 2), slice(0, 8))
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_core1_rows, 1) == (
            shard_hw,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_core1_rows, 0) == (
            0,
            shard_hw,
            0,
        )
        assert shard_origin_from_key(tt_tensor, k_core0_rows) == (0, 0, 0)
        assert shard_origin_from_key(tt_tensor, k_core1_rows) == (0, 32, 0)

    def test_advanced_height_sharding_memory_config(self) -> None:
        """Advanced API: custom height sharding via ``MemoryConfig`` + ``ShardSpec``."""
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(
                    target_num_cores=8,
                    grid_size=[8, 7],
                    row_wise=True,
                ),
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        assert memory_config.shard_spec is not None
        assert memory_config.shard_spec.shard_grid == (8,)
        assert memory_config.shard_spec.shard_shape == (64, 512)
        torch_tensor = torch.randn(512, 512)
        height_sharded_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.float32,
            device=self._device(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
        assert height_sharded_tensor.shape == (512, 512)
        assert ttnn.is_sharded(height_sharded_tensor)
        sp = memory_config.shard_spec
        assert sp is not None
        plane_el = height_sharded_tensor.shape[-2] * height_sharded_tensor.shape[-1]
        loc0, rem0, _ = count_local_remote_l1_dram(height_sharded_tensor, 0)
        loc7, rem7, _ = count_local_remote_l1_dram(height_sharded_tensor, 7)
        assert loc0 + rem0 == plane_el and loc7 + rem7 == plane_el
        shard_hw = sp.shard_shape[-2] * sp.shard_shape[-1]
        assert loc0 == shard_hw and loc7 == shard_hw
        assert rem0 == plane_el - shard_hw and rem7 == plane_el - shard_hw
        k0 = (slice(0, 2), slice(0, 16))
        assert count_local_remote_l1_dram_for_getitem(height_sharded_tensor, k0, 0) == (
            sp.shard_shape[-2] * sp.shard_shape[-1],
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(height_sharded_tensor, k0, 1) == (
            0,
            sp.shard_shape[-2] * sp.shard_shape[-1],
            0,
        )
        assert shard_origin_from_key(height_sharded_tensor, k0) == (0, 0)

    def test_width_sharding_tensor_spec(self) -> None:
        """2D Width Sharding: ``TensorSpec`` + ``width_sharded`` (4 cores, 1x4 grid)."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
            }
        )
        tensor_spec = ttnn.TensorSpec(
            shape=(1, 64, 512),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).width_sharded(core_ranges)
        sp = tensor_spec.memory_config.shard_spec
        assert sp is not None
        assert sp.shard_grid == (4,)
        assert sp.shard_shape == (64, 128)
        torch_tensor = torch.randn(tuple(tensor_spec.shape))
        tt_tensor = ttnn.from_torch(
            torch_tensor, spec=tensor_spec, device=self._device()
        )
        assert tt_tensor.shape == (1, 64, 512)
        assert ttnn.is_sharded(tt_tensor)
        plane_el = tt_tensor.shape[-2] * tt_tensor.shape[-1]
        loc0, rem0, _ = count_local_remote_l1_dram(tt_tensor, 0)
        loc3, rem3, _ = count_local_remote_l1_dram(tt_tensor, 3)
        assert loc0 + rem0 == plane_el and loc3 + rem3 == plane_el
        sw = sp.shard_shape[-1]
        assert loc0 == tt_tensor.shape[-2] * sw
        assert loc3 == loc0
        k_w0 = (slice(0, 1), slice(0, 2), slice(0, 4))
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_w0, 0) == (
            plane_el // sp.shard_grid[0],
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_w0, 3) == (
            0,
            plane_el // sp.shard_grid[0],
            0,
        )
        k_w3 = (slice(0, 1), slice(0, 2), slice(12, 16))
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_w3, 3) == (
            plane_el // sp.shard_grid[0],
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k_w3, 0) == (
            0,
            plane_el // sp.shard_grid[0],
            0,
        )
        assert shard_origin_from_key(tt_tensor, k_w0) == (0, 0, 0)
        assert shard_origin_from_key(tt_tensor, k_w3) == (0, 0, 384)

    def test_advanced_width_sharding_memory_config(self) -> None:
        """Advanced API: width sharding via ``MemoryConfig`` + ``ShardSpec`` (keyword grid)."""
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1)),
                    }
                ),
                shard_shape=[128, 64],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        assert memory_config.shard_spec is not None
        assert memory_config.shard_spec.shard_grid == (8,)
        assert memory_config.shard_spec.shard_shape == (128, 64)
        torch_tensor = torch.randn(128, 512)
        width_sharded_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.float32,
            device=self._device(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
        assert width_sharded_tensor.shape == (128, 512)
        assert ttnn.is_sharded(width_sharded_tensor)
        sp = memory_config.shard_spec
        assert sp is not None
        plane_el = width_sharded_tensor.shape[-2] * width_sharded_tensor.shape[-1]
        loc0, rem0, _ = count_local_remote_l1_dram(width_sharded_tensor, 0)
        assert loc0 + rem0 == plane_el
        assert loc0 == width_sharded_tensor.shape[-2] * sp.shard_shape[-1]
        k_w0 = (slice(0, 4), slice(0, 2))
        assert count_local_remote_l1_dram_for_getitem(
            width_sharded_tensor, k_w0, 0
        ) == (
            loc0,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            width_sharded_tensor, k_w0, 1
        ) == (
            0,
            loc0,
            0,
        )
        assert shard_origin_from_key(width_sharded_tensor, k_w0) == (0, 0)

    def test_block_sharding_tensor_spec(self) -> None:
        """Block sharding: ``TensorSpec`` + ``block_sharded`` (16 cores, 4x4 grid)."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),
            }
        )
        tensor_spec = ttnn.TensorSpec(
            shape=(1, 256, 256),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).block_sharded(core_ranges)
        sp = tensor_spec.memory_config.shard_spec
        assert sp is not None
        assert sp.shard_grid == (4, 4)
        assert sp.shard_shape == (64, 64)
        torch_tensor = torch.randn(tuple(tensor_spec.shape))
        tt_tensor = ttnn.from_torch(
            torch_tensor, spec=tensor_spec, device=self._device()
        )
        assert tt_tensor.shape == (1, 256, 256)
        assert ttnn.is_sharded(tt_tensor)
        plane_el = tt_tensor.shape[-2] * tt_tensor.shape[-1]
        loc0, rem0, _ = count_local_remote_l1_dram(tt_tensor, 0)
        loc15, rem15, _ = count_local_remote_l1_dram(tt_tensor, 15)
        assert loc0 + rem0 == plane_el and loc15 + rem15 == plane_el
        sh, sw = sp.shard_shape[-2], sp.shard_shape[-1]
        assert loc0 == sh * sw
        assert loc15 == sh * sw
        k00 = (slice(0, 1), slice(0, 2), slice(0, 2))
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k00, 0) == (
            sh * sw,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k00, 5) == (
            0,
            sh * sw,
            0,
        )
        k11 = (slice(0, 1), slice(2, 4), slice(2, 4))
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k11, 5) == (
            sh * sw,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(tt_tensor, k11, 0) == (
            0,
            sh * sw,
            0,
        )
        assert shard_origin_from_key(tt_tensor, k00) == (0, 0, 0)
        assert shard_origin_from_key(tt_tensor, k11) == (0, 64, 64)

    def test_advanced_block_sharding_memory_config(self) -> None:
        """Advanced API: block sharding via ``MemoryConfig`` + ``ShardSpec``."""
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),
                    }
                ),
                shard_shape=[64, 64],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        assert memory_config.shard_spec is not None
        assert memory_config.shard_spec.shard_grid == (4, 4)
        assert memory_config.shard_spec.shard_shape == (64, 64)
        torch_tensor = torch.randn(192, 192)
        block_sharded_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.float32,
            device=self._device(),
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )
        assert block_sharded_tensor.shape == (192, 192)
        assert ttnn.is_sharded(block_sharded_tensor)
        sp = memory_config.shard_spec
        assert sp is not None
        plane_el = block_sharded_tensor.shape[-2] * block_sharded_tensor.shape[-1]
        sh, sw = sp.shard_shape[-2], sp.shard_shape[-1]
        loc0, rem0, _ = count_local_remote_l1_dram(block_sharded_tensor, 0)
        loc5, rem5, _ = count_local_remote_l1_dram(block_sharded_tensor, 5)
        loc15, rem15, _ = count_local_remote_l1_dram(block_sharded_tensor, 15)
        assert loc0 + rem0 == plane_el and loc5 + rem5 == plane_el
        assert loc15 + rem15 == plane_el
        assert loc0 == sh * sw and loc5 == sh * sw
        assert loc15 == 0 and rem15 == plane_el
        k00 = (slice(0, 2), slice(0, 2))
        k55 = (slice(2, 4), slice(2, 4))
        assert count_local_remote_l1_dram_for_getitem(block_sharded_tensor, k00, 0) == (
            sh * sw,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(block_sharded_tensor, k00, 5) == (
            0,
            sh * sw,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(block_sharded_tensor, k55, 5) == (
            sh * sw,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(block_sharded_tensor, k55, 0) == (
            0,
            sh * sw,
            0,
        )
        assert shard_origin_from_key(block_sharded_tensor, k00) == (0, 0)
        assert shard_origin_from_key(block_sharded_tensor, k55) == (64, 64)

    def test_nd_sharding_tensor_spec_batch_seq_and_features(self) -> None:
        """ND sharding examples: ``sharded_across_dims`` for batch+seq and features."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3)),
            }
        )
        nd_spec_batch_seq = ttnn.TensorSpec(
            shape=(4, 512, 768),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).sharded_across_dims([0, 1], core_ranges)
        assert nd_spec_batch_seq.memory_config.nd_shard_spec is not None
        nd0 = nd_spec_batch_seq.memory_config.nd_shard_spec
        assert nd0.shard_grid == (4, 2, 1)
        assert nd0.shard_shape == (1, 256, 768)
        torch_tensor = torch.randn(tuple(nd_spec_batch_seq.shape))
        batch_seq_sharded = ttnn.from_torch(
            torch_tensor, spec=nd_spec_batch_seq, device=self._device()
        )
        assert batch_seq_sharded.shape == (4, 512, 768)
        assert ttnn.is_sharded(batch_seq_sharded)
        total_bs = math.prod(batch_seq_sharded.shape)
        loc0_bs, rem0_bs, _ = count_local_remote_l1_dram(batch_seq_sharded, 0)
        loc7_bs, rem7_bs, _ = count_local_remote_l1_dram(batch_seq_sharded, 7)
        assert loc0_bs + rem0_bs == total_bs and loc7_bs + rem7_bs == total_bs
        assert loc0_bs == math.prod(nd0.shard_shape)
        assert loc7_bs == math.prod(nd0.shard_shape)
        k_bs0 = (0, slice(0, 8), slice(0, 24))
        k_bs7 = (3, slice(8, 16), slice(0, 24))
        assert count_local_remote_l1_dram_for_getitem(batch_seq_sharded, k_bs0, 0) == (
            math.prod(nd0.shard_shape),
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(batch_seq_sharded, k_bs0, 1) == (
            0,
            math.prod(nd0.shard_shape),
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(batch_seq_sharded, k_bs7, 7) == (
            math.prod(nd0.shard_shape),
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(batch_seq_sharded, k_bs7, 0) == (
            0,
            math.prod(nd0.shard_shape),
            0,
        )
        assert shard_origin_from_key(batch_seq_sharded, k_bs0) == (0, 0, 0)
        assert shard_origin_from_key(batch_seq_sharded, k_bs7) == (3, 256, 0)

        nd_spec_features = ttnn.TensorSpec(
            shape=(2, 256, 1024),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).sharded_across_dims([2], core_ranges)
        nd1 = nd_spec_features.memory_config.nd_shard_spec
        assert nd1 is not None
        assert nd1.shard_grid == (1, 1, 8)
        assert nd1.shard_shape == (2, 256, 128)
        torch_tensor_b = torch.randn(tuple(nd_spec_features.shape))
        feature_sharded = ttnn.from_torch(
            torch_tensor_b, spec=nd_spec_features, device=self._device()
        )
        assert feature_sharded.shape == (2, 256, 1024)
        assert ttnn.is_sharded(feature_sharded)
        total_f = math.prod(feature_sharded.shape)
        loc0_f, rem0_f, _ = count_local_remote_l1_dram(feature_sharded, 0)
        loc7_f, rem7_f, _ = count_local_remote_l1_dram(feature_sharded, 7)
        assert loc0_f + rem0_f == total_f and loc7_f + rem7_f == total_f
        assert loc0_f == math.prod(nd1.shard_shape)
        assert loc7_f == math.prod(nd1.shard_shape)
        k_f0 = (slice(0, 2), slice(0, 8), slice(0, 4))
        k_f7 = (slice(0, 2), slice(0, 8), slice(28, 32))
        assert count_local_remote_l1_dram_for_getitem(feature_sharded, k_f0, 0) == (
            math.prod(nd1.shard_shape),
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(feature_sharded, k_f0, 7) == (
            0,
            math.prod(nd1.shard_shape),
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(feature_sharded, k_f7, 7) == (
            math.prod(nd1.shard_shape),
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(feature_sharded, k_f7, 0) == (
            0,
            math.prod(nd1.shard_shape),
            0,
        )
        assert shard_origin_from_key(feature_sharded, k_f0) == (0, 0, 0)
        assert shard_origin_from_key(feature_sharded, k_f7) == (0, 0, 896)

    def test_advanced_nd_shard_spec_memory_config(self) -> None:
        """Example 3: Advanced ND sharding with custom shard specification (tech report)."""
        core_ranges = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3)),
            }
        )
        nd_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            NdShardSpec(
                shard_shape=[1, 1, 64, 128],
                core_ranges=core_ranges,
            ),
        )
        torch_tensor = torch.randn(2, 4, 256, 512)
        device = self._device()
        advanced_nd_sharded = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.float32,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=nd_memory_config,
        )
        assert advanced_nd_sharded.shape == (2, 4, 256, 512)
        assert ttnn.is_sharded(advanced_nd_sharded)
        nd = advanced_nd_sharded.memory_config.nd_shard_spec
        assert nd is not None
        assert nd.distribution == ShardDistributionStrategy.GRID_2D
        total_nd = math.prod(advanced_nd_sharded.shape)
        loc0_nd, rem0_nd, _ = count_local_remote_l1_dram(advanced_nd_sharded, 0)
        loc1_nd, rem1_nd, _ = count_local_remote_l1_dram(advanced_nd_sharded, 1)
        assert loc0_nd + rem0_nd == total_nd and loc1_nd + rem1_nd == total_nd
        assert loc0_nd == math.prod(nd.shard_shape)
        k_nd0 = (0, 0, slice(0, 2), slice(0, 4))
        k_nd1 = (0, 0, slice(0, 2), slice(4, 8))
        assert count_local_remote_l1_dram_for_getitem(
            advanced_nd_sharded, k_nd0, 0
        ) == (
            math.prod(nd.shard_shape),
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            advanced_nd_sharded, k_nd0, 1
        ) == (
            0,
            math.prod(nd.shard_shape),
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            advanced_nd_sharded, k_nd1, 1
        ) == (
            math.prod(nd.shard_shape),
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            advanced_nd_sharded, k_nd1, 0
        ) == (
            0,
            math.prod(nd.shard_shape),
            0,
        )
        assert shard_origin_from_key(advanced_nd_sharded, k_nd0) == (0, 0, 0, 0)
        assert shard_origin_from_key(advanced_nd_sharded, k_nd1) == (0, 0, 0, 128)


class TestShardingHelpers:
    """Tests for is_sharded, get_memory_config, and to_memory_config."""

    def test_is_sharded_interleaved_returns_false(self) -> None:
        """Interleaved tensors are not sharded."""
        t = ttnn.from_torch(torch.zeros(64, 64))
        assert not ttnn.is_sharded(t)

    def test_is_sharded_height_sharded_returns_true(self) -> None:
        """Height-sharded tensors are considered sharded."""
        mc = MemoryConfig(
            strategy=ShardingStrategy.HEIGHT_SHARDED,
            shard_spec=ShardSpec(shard_grid=(4,), shard_shape=(32, 64)),
        )
        t = ttnn.from_torch(torch.zeros(128, 64), memory_config=mc)
        assert ttnn.is_sharded(t)

    def test_is_sharded_block_sharded_returns_true(self) -> None:
        """Block-sharded tensors are considered sharded."""
        mc = MemoryConfig(
            strategy=ShardingStrategy.BLOCK_SHARDED,
            shard_spec=ShardSpec(shard_grid=(2, 2), shard_shape=(32, 32)),
        )
        t = ttnn.from_torch(torch.zeros(64, 64), memory_config=mc)
        assert ttnn.is_sharded(t)

    def test_get_memory_config_returns_attached_config(self) -> None:
        """get_memory_config returns the MemoryConfig stored on the tensor."""
        mc = MemoryConfig(
            strategy=ShardingStrategy.HEIGHT_SHARDED,
            shard_spec=ShardSpec(shard_grid=(4,), shard_shape=(32, 64)),
        )
        t = ttnn.from_torch(torch.zeros(128, 64), memory_config=mc)
        assert ttnn.get_memory_config(t) is mc

    def test_get_memory_config_default_is_dram(self) -> None:
        """get_memory_config on a plain tensor returns DRAM_MEMORY_CONFIG."""
        t = ttnn.from_torch(torch.zeros(64, 64))
        assert ttnn.get_memory_config(t) == ttnn.DRAM_MEMORY_CONFIG

    def test_to_memory_config_updates_config(self) -> None:
        """to_memory_config returns a tensor with the new MemoryConfig."""
        raw = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
        src = ttnn.from_torch(raw)
        mc = MemoryConfig(
            strategy=ShardingStrategy.HEIGHT_SHARDED,
            shard_spec=ShardSpec(shard_grid=(4,), shard_shape=(32, 64)),
        )
        dst = ttnn.to_memory_config(src, mc)
        assert ttnn.get_memory_config(dst) == mc

    def test_to_memory_config_preserves_data(self) -> None:
        """to_memory_config does not alter tensor values."""
        raw = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
        src = ttnn.from_torch(raw)
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED)
        dst = ttnn.to_memory_config(src, mc)
        assert torch.equal(dst.to_torch(), raw)

    def test_to_memory_config_does_not_mutate_source(self) -> None:
        """to_memory_config leaves the original tensor's MemoryConfig unchanged."""
        t = ttnn.from_torch(torch.zeros(64, 64))
        original_mc = ttnn.get_memory_config(t)
        ttnn.to_memory_config(t, MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED))
        assert ttnn.get_memory_config(t) is original_mc

    def test_to_memory_config_preserves_layout(self) -> None:
        """to_memory_config propagates the source tensor's layout."""
        raw = torch.zeros(5, 9)
        src = ttnn.from_torch(raw, layout=ttnn.ROW_MAJOR_LAYOUT)
        dst = ttnn.to_memory_config(src, ttnn.DRAM_MEMORY_CONFIG)
        assert dst.layout == ttnn.ROW_MAJOR_LAYOUT


class TestRowMajorLayout:
    """Tests for ROW_MAJOR_LAYOUT Tensor behaviour (Steps 1 and 2)."""

    # --- constants and construction ---

    def test_row_major_constant_accessible(self) -> None:
        """ROW_MAJOR_LAYOUT is exported from ttnnsim and is distinct from TILE_LAYOUT."""
        assert hasattr(ttnn, "ROW_MAJOR_LAYOUT")
        assert hasattr(ttnn, "TILE_LAYOUT")
        assert ttnn.ROW_MAJOR_LAYOUT != ttnn.TILE_LAYOUT

    def test_tensor_default_layout_is_tile(self) -> None:
        """Tensors constructed without explicit layout default to TILE_LAYOUT."""
        t = ttnn.Tensor(torch.zeros(32, 32))
        assert t.layout == ttnn.TILE_LAYOUT

    def test_tensor_row_major_layout_property(self) -> None:
        """Tensor.layout reports ROW_MAJOR_LAYOUT when constructed with it."""
        t = ttnn.Tensor(torch.zeros(7, 13), ttnn.ROW_MAJOR_LAYOUT)
        assert t.layout == ttnn.ROW_MAJOR_LAYOUT

    # --- non-tile-aligned shapes accepted ---

    def test_non_tile_aligned_shape_accepted(self) -> None:
        """Row-major Tensors with non-tile-aligned dimensions do not raise."""
        t = ttnn.Tensor(torch.zeros(7, 13), ttnn.ROW_MAJOR_LAYOUT)
        assert t.shape == (7, 13)

    def test_tile_alignment_not_checked_on_getitem(self) -> None:
        """Indexing a row-major Tensor with a non-tile-aligned shape does not raise."""
        raw = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        t = ttnn.Tensor(raw, ttnn.ROW_MAJOR_LAYOUT)
        result = t[0, 0]  # element (0, 0) as a (1, 1) slice
        assert result.shape == (1, 1)
        assert result.to_torch().item() == 0.0

    # --- element-space indexing (no tile scaling) ---

    def test_integer_index_becomes_unit_slice(self) -> None:
        """Integer index n maps to element slice n:n+1, not n*32:(n+1)*32."""
        raw = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        t = ttnn.Tensor(raw, ttnn.ROW_MAJOR_LAYOUT)
        result = t[1, 2]
        assert result.shape == (1, 1)
        assert result.to_torch().item() == raw[1, 2].item()

    def test_slice_index_passes_through_unchanged(self) -> None:
        """Slice indices are passed through without any TILE_SHAPE scaling."""
        raw = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        t = ttnn.Tensor(raw, ttnn.ROW_MAJOR_LAYOUT)
        result = t[slice(1, 3), slice(0, 4)]
        assert result.shape == (2, 4)
        assert torch.equal(result.to_torch(), raw[1:3, 0:4])

    def test_1d_integer_index(self) -> None:
        """1-D row-major: integer index n selects element n:n+1."""
        raw = torch.arange(8, dtype=torch.float32)
        t = ttnn.Tensor(raw, ttnn.ROW_MAJOR_LAYOUT)
        result = t[3]
        assert result.shape == (1,)
        assert result.to_torch().item() == 3.0

    def test_nd_indexing(self) -> None:
        """Row-major indexing works for 3-D tensors without tile scaling."""
        raw = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        t = ttnn.Tensor(raw, ttnn.ROW_MAJOR_LAYOUT)
        result = t[1, 2, slice(0, 4)]
        assert result.shape == (1, 1, 4)
        assert torch.equal(result.to_torch(), raw[1:2, 2:3, 0:4])

    def test_setitem_row_major(self) -> None:
        """__setitem__ writes at element-space coordinates for row-major."""
        raw = torch.zeros(4, 4, dtype=torch.float32)
        t = ttnn.Tensor(raw, ttnn.ROW_MAJOR_LAYOUT)
        t[2, 3] = ttnn.Tensor(torch.full((1, 1), 99.0))
        assert raw[2, 3].item() == 99.0
        assert raw[0, 0].item() == 0.0

    # --- repr ---

    def test_repr_shows_row_major_layout(self) -> None:
        """repr includes layout=ROW_MAJOR for row-major tensors."""
        t = ttnn.Tensor(torch.zeros(3, 4), ttnn.ROW_MAJOR_LAYOUT)
        r = repr(t)
        assert "ROW_MAJOR" in r

    def test_repr_omits_layout_for_tile(self) -> None:
        """repr does not include a layout field for the default TILE_LAYOUT."""
        t = ttnn.Tensor(torch.zeros(32, 32))
        assert "layout" not in repr(t)

    # --- creation helpers propagate layout ---

    def test_rand_propagates_row_major(self) -> None:
        t = ttnn.rand((5, 7), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        assert t.layout == ttnn.ROW_MAJOR_LAYOUT
        assert t.shape == (5, 7)

    def test_empty_propagates_row_major(self) -> None:
        t = ttnn.empty((3, 11), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        assert t.layout == ttnn.ROW_MAJOR_LAYOUT

    def test_from_torch_propagates_row_major(self) -> None:
        raw = torch.randn(5, 9)
        t = ttnn.from_torch(raw, layout=ttnn.ROW_MAJOR_LAYOUT)
        assert t.layout == ttnn.ROW_MAJOR_LAYOUT
        assert t.shape == (5, 9)

    # --- layout propagates through arithmetic ---

    def test_arithmetic_preserves_row_major(self) -> None:
        """Binary and unary ops on row-major Tensors return row-major Tensors."""
        a = ttnn.Tensor(torch.ones(3, 4), ttnn.ROW_MAJOR_LAYOUT)
        b = ttnn.Tensor(torch.ones(3, 4), ttnn.ROW_MAJOR_LAYOUT)

        assert (a + b).layout == ttnn.ROW_MAJOR_LAYOUT
        assert (a - b).layout == ttnn.ROW_MAJOR_LAYOUT
        assert (a * b).layout == ttnn.ROW_MAJOR_LAYOUT
        assert (a / b).layout == ttnn.ROW_MAJOR_LAYOUT
        assert (a**2).layout == ttnn.ROW_MAJOR_LAYOUT
        assert (-a).layout == ttnn.ROW_MAJOR_LAYOUT
        assert abs(a).layout == ttnn.ROW_MAJOR_LAYOUT

    def test_scalar_arithmetic_preserves_row_major(self) -> None:
        """Scalar operands preserve the layout of the Tensor side."""
        a = ttnn.Tensor(torch.ones(3, 4), ttnn.ROW_MAJOR_LAYOUT)
        assert (a + 1.0).layout == ttnn.ROW_MAJOR_LAYOUT
        assert (2.0 * a).layout == ttnn.ROW_MAJOR_LAYOUT

    # --- tile_count_from_tensor ---

    def test_tile_count_row_major_returns_scalar_count(self) -> None:
        """tile_count_from_tensor returns total element count for row-major."""
        t = ttnn.Tensor(torch.zeros(3, 4), ttnn.ROW_MAJOR_LAYOUT)
        assert ttnn.tile_count_from_tensor(t) == 12

    def test_tile_count_row_major_1d(self) -> None:
        t = ttnn.Tensor(torch.zeros(7), ttnn.ROW_MAJOR_LAYOUT)
        assert ttnn.tile_count_from_tensor(t) == 7

    def test_tile_count_row_major_nd(self) -> None:
        t = ttnn.Tensor(torch.zeros(2, 3, 5), ttnn.ROW_MAJOR_LAYOUT)
        assert ttnn.tile_count_from_tensor(t) == 30

    def test_tile_count_tiled_unaffected(self) -> None:
        """Tile count for tiled tensors is unchanged (regression guard)."""
        t = ttnn.Tensor(torch.zeros(64, 64))  # 2x2 tiles
        assert ttnn.tile_count_from_tensor(t) == 4


class TestAllReduce:
    """Tests for :func:`~sim.ttnnsim.all_reduce`.

    Partition structure is communicated via the tensor's ``mesh_shard_info``
    attribute, which is set by :func:`from_torch` when a
    :class:`~ttnnsim.ShardTensorToMesh` mapper is provided.  This is kept
    separate from the intra-device sharding strategies stored in
    :class:`~ttnnsim.MemoryConfig`.
    """

    def _mesh(self, n: int) -> ttnn.MeshDevice:
        return ttnn.open_mesh_device(ttnn.MeshShape(1, n))

    def test_shard_to_mesh_sets_mesh_shard_info(self) -> None:
        """from_torch with ShardTensorToMesh records dim and device count in mesh_shard_info."""
        mesh = self._mesh(4)
        t = ttnn.from_torch(
            torch.zeros(8, 6),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        assert t.mesh_shard_info is not None
        assert t.mesh_shard_info.dim == 0
        assert t.mesh_shard_info.num_devices == 4
        assert t.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_shard_to_mesh_records_width_dim(self) -> None:
        """ShardTensorToMesh along the last dim records dim=1 in mesh_shard_info."""
        mesh = self._mesh(3)
        t = ttnn.from_torch(
            torch.zeros(4, 9),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        assert t.mesh_shard_info is not None
        assert t.mesh_shard_info.dim == 1
        assert t.mesh_shard_info.num_devices == 3
        assert t.memory_config == ttnn.DRAM_MEMORY_CONFIG

    def test_all_reduce_via_mesh_sums_shards(self) -> None:
        """all_reduce over a ShardTensorToMesh tensor sums the shards."""
        mesh = self._mesh(4)
        # Build a tensor where each shard-row block holds a different value.
        data = torch.zeros(8, 4)
        data[0:2, :] = 1.0
        data[2:4, :] = 2.0
        data[4:6, :] = 3.0
        data[6:8, :] = 4.0
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_reduce(t)
        expected_shard = torch.full((2, 4), 10.0)
        for i in range(4):
            assert torch.allclose(
                result.to_torch()[i * 2 : (i + 1) * 2], expected_shard
            )

    def test_all_reduce_single_device_identity(self) -> None:
        """With a single-device mesh, all_reduce is an identity."""
        mesh = self._mesh(1)
        data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_reduce(t)
        assert torch.allclose(result.to_torch(), data)

    def test_all_reduce_preserves_layout(self) -> None:
        """Output layout matches input layout."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        assert ttnn.all_reduce(t).layout == ttnn.ROW_MAJOR_LAYOUT

    def test_all_reduce_dtype_conversion(self) -> None:
        """Output is cast when dtype is given."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4, dtype=torch.float32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_reduce(t, dtype=torch.float16)
        assert result.to_torch().dtype == torch.float16

    def test_all_reduce_memory_config_override(self) -> None:
        """Explicit memory_config is applied to the output."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        custom_mc = MemoryConfig(strategy=ShardingStrategy.INTERLEAVED)
        result = ttnn.all_reduce(t, memory_config=custom_mc)
        assert result.memory_config == custom_mc

    def test_all_reduce_kwargs_accepted(self) -> None:
        """Extra keyword arguments are accepted without error."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.all_reduce(t, cluster_axis=0, mesh_device=mesh)

    # ---- Error on unsharded tensor ----

    def test_all_reduce_requires_shard_metadata(self) -> None:
        """all_reduce raises ValueError when the tensor has no mesh sharding metadata."""
        t = ttnn.Tensor(torch.ones(8, 4))
        with pytest.raises(
            ValueError, match="Mesh device is required for all_reduce operation"
        ):
            ttnn.all_reduce(t)

    def test_shard_tensor_not_divisible_still_sets_mesh_shard_info(self) -> None:
        """from_torch with ShardTensorToMesh records mesh_shard_info even when indivisible."""
        mesh = self._mesh(3)
        t = ttnn.from_torch(
            torch.zeros(8, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        assert t.mesh_shard_info is not None
        assert t.mesh_shard_info.num_devices == 3
        assert t.mesh_shard_info.dim == 0

    def test_all_reduce_3d_partitioned_along_middle_dim(self) -> None:
        """all_reduce on a 3-D tensor partitioned along dim 1 reduces along that axis."""
        mesh = self._mesh(2)
        # Shape (B, H*n, W) — partitioned along dim 1.
        data = torch.zeros(3, 4, 5)
        data[:, 0:2, :] = 1.0  # first device's shard
        data[:, 2:4, :] = 3.0  # second device's shard
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        assert t.mesh_shard_info is not None
        assert t.mesh_shard_info.dim == 1
        result = ttnn.all_reduce(t)
        expected_shard = torch.full((3, 2, 5), 4.0)
        assert torch.allclose(result.to_torch()[:, 0:2, :], expected_shard)
        assert torch.allclose(result.to_torch()[:, 2:4, :], expected_shard)


class TestAllGather:
    """Tests for :func:`~sim.ttnnsim.all_gather`.

    The gather operation concatenates all per-device shards along ``dim``.
    Every device ends up with the same result.  The simulator represents
    n identical copies by stacking them along ``msi.dim``.
    """

    def _mesh(self, n: int) -> ttnn.MeshDevice:
        return ttnn.open_mesh_device(ttnn.MeshShape(1, n))

    def test_all_gather_same_dim_as_shard_dim(self) -> None:
        """all_gather along shard_dim concatenates all shards; output is n times the input."""
        mesh = self._mesh(4)
        data = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_gather(t, dim=0)
        # Each shard is [2, 4]; gathered per device = [8, 4] = data itself.
        # Output = 4 copies stacked along dim 0 = [32, 4].
        assert result.to_torch().shape == (32, 4)
        # Every [8, 4] block should equal the original data.
        for i in range(4):
            assert torch.allclose(result.to_torch()[i * 8 : (i + 1) * 8], data)

    def test_all_gather_different_dim_from_shard_dim(self) -> None:
        """all_gather along a non-shard dim grows that dim by num_devices."""
        mesh = self._mesh(4)
        # 4 devices, each with a [2, 6] shard; sharded along dim 0.
        data = torch.arange(48, dtype=torch.float32).reshape(8, 6)
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_gather(t, dim=1)
        # Each shard is [2, 6]; gathered along dim 1 = [2, 24].
        # Output = 4 copies stacked along dim 0 = [8, 24].
        assert result.to_torch().shape == (8, 24)
        # Device i's shard is data[i*2:(i+1)*2, :]; gathered along dim 1
        # = cat([shard_0, shard_1, shard_2, shard_3], dim=1) = [2, 24].
        expected_gathered_shard = torch.cat(
            [data[i * 2 : (i + 1) * 2, :] for i in range(4)], dim=1
        )
        for i in range(4):
            assert torch.allclose(
                result.to_torch()[i * 2 : (i + 1) * 2, :], expected_gathered_shard
            )

    def test_all_gather_single_device_identity(self) -> None:
        """With a single-device mesh, all_gather is an identity."""
        mesh = self._mesh(1)
        data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_gather(t, dim=0)
        assert torch.allclose(result.to_torch(), data)

    def test_all_gather_preserves_layout(self) -> None:
        """Output layout matches input layout."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        assert ttnn.all_gather(t, dim=0).layout == ttnn.ROW_MAJOR_LAYOUT

    def test_all_gather_memory_config_override(self) -> None:
        """Explicit memory_config is applied to the output."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        custom_mc = MemoryConfig(strategy=ShardingStrategy.INTERLEAVED)
        result = ttnn.all_gather(t, dim=0, memory_config=custom_mc)
        assert result.memory_config == custom_mc

    def test_all_gather_preserves_mesh_shard_info(self) -> None:
        """Output mesh_shard_info keeps the same dim and num_devices."""
        mesh = self._mesh(4)
        t = ttnn.from_torch(
            torch.ones(8, 6),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        result = ttnn.all_gather(t, dim=0)
        assert result.mesh_shard_info is not None
        assert result.mesh_shard_info.dim == 0
        assert result.mesh_shard_info.num_devices == 4

    def test_all_gather_kwargs_accepted(self) -> None:
        """Extra keyword arguments are accepted without error."""
        mesh = self._mesh(2)
        t = ttnn.from_torch(
            torch.ones(4, 4),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.all_gather(t, dim=0, cluster_axis=0, mesh_device=mesh)

    def test_all_gather_requires_shard_metadata(self) -> None:
        """all_gather raises ValueError when the tensor has no mesh sharding metadata."""
        t = ttnn.Tensor(torch.ones(8, 4))
        with pytest.raises(
            ValueError, match="Mesh device is required for all_gather operation"
        ):
            ttnn.all_gather(t, dim=0)


class TestSynchronizeDevice:
    """synchronize_device() is a no-op in the simulator."""

    def test_no_args(self) -> None:
        """Callable with no arguments."""
        ttnn.synchronize_device()

    def test_with_device_arg(self) -> None:
        """Callable with a positional device argument, as in real hardware code."""
        ttnn.synchronize_device("mock_device")

    def test_returns_none(self) -> None:
        """Return value is None."""
        assert ttnn.synchronize_device() is None


# ---------------------------------------------------------------------------
# 2D mesh support
# ---------------------------------------------------------------------------


def _make_2d_mesh(rows: int, cols: int) -> Any:
    return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))


def _shard_2d(
    data: torch.Tensor, rows: int, cols: int, dims: tuple[int | None, int | None]
) -> ttnn.Tensor:
    mesh = _make_2d_mesh(rows, cols)
    return ttnn.from_torch(
        data,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=dims),
    )


class TestShardTensor2dMesh:
    """ShardTensor2dMesh sets correct MeshShardInfo on the resulting Tensor."""

    def test_mesh_shape_stored(self) -> None:
        """mesh_shard_info.mesh_shape matches the requested mesh grid."""
        t = _shard_2d(torch.ones(4, 8), rows=2, cols=4, dims=(0, 1))
        assert t.mesh_shard_info is not None
        assert t.mesh_shard_info.mesh_shape == (2, 4)

    def test_dims_stored(self) -> None:
        """mesh_shard_info.dims matches the requested partition dims."""
        t = _shard_2d(torch.ones(4, 8), rows=2, cols=4, dims=(0, 1))
        assert t.mesh_shard_info.dims == (0, 1)

    def test_num_devices_is_product(self) -> None:
        """num_devices equals rows * cols."""
        t = _shard_2d(torch.ones(4, 8), rows=2, cols=4, dims=(0, 1))
        assert t.mesh_shard_info.num_devices == 8

    def test_dim_property_raises_for_2d(self) -> None:
        """dim property raises ValueError when both axes shard the tensor."""
        t = _shard_2d(torch.ones(4, 8), rows=2, cols=4, dims=(0, 1))
        with pytest.raises(ValueError, match="ambiguous"):
            _ = t.mesh_shard_info.dim

    def test_dim_property_valid_for_one_active_axis(self) -> None:
        """dim property returns the single active dim when only one axis shards."""
        t = _shard_2d(torch.ones(4, 8), rows=1, cols=4, dims=(None, 1))
        assert t.mesh_shard_info.dim == 1

    def test_negative_dims_normalized(self) -> None:
        """Negative dim values are normalized to positive indices."""
        data = torch.ones(4, 8)
        t = _shard_2d(data, rows=2, cols=4, dims=(-2, -1))
        # For a 2D tensor: -2 % 2 = 0, -1 % 2 = 1
        assert t.mesh_shard_info.dims == (0, 1)

    def test_none_sentinel_preserved(self) -> None:
        """A None sentinel in dims marks that mesh axis as inactive."""
        t = _shard_2d(torch.ones(4, 8), rows=1, cols=4, dims=(None, 0))
        assert t.mesh_shard_info.dims[0] is None
        assert t.mesh_shard_info.dims[1] == 0

    def test_underlying_data_unchanged(self) -> None:
        """from_torch does not modify the tensor data when using ShardTensor2dMesh."""
        data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        t = _shard_2d(data, rows=2, cols=2, dims=(0, 1))
        assert torch.equal(t.to_torch(), data)


class TestAllReduce2DMesh:
    """all_reduce on 2D-mesh tensors with cluster_axis support.

    The simulator represents a 2D mesh by storing the full tensor: device
    (r, c) owns the slice at dims[0][r] x dims[1][c].  For a 2x2 mesh
    with dims=(0,1) and a [4,4] tensor, device (r,c) holds t[r*2:(r+1)*2,
    c*2:(c+1)*2].

    Test data layout (values equal device-local sum contribution):

        t = [[1, 1, 2, 2],
             [1, 1, 2, 2],
             [3, 3, 4, 4],
             [3, 3, 4, 4]]

    * device (0,0): block of 1s  * device (0,1): block of 2s
    * device (1,0): block of 3s  * device (1,1): block of 4s
    """

    def _tensor(self) -> ttnn.Tensor:
        data = torch.tensor(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=torch.float32,
        )
        return _shard_2d(data, rows=2, cols=2, dims=(0, 1))

    def test_cluster_axis_0_reduces_rows(self) -> None:
        """cluster_axis=0 sums across the row mesh axis (dim 0, 2 devices).

        Each col block sees: 1+3=4 and 2+4=6, written back to both row slots.
        """
        t = self._tensor()
        result = ttnn.all_reduce(t, cluster_axis=0)
        expected = torch.tensor(
            [[4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6], [4, 4, 6, 6]],
            dtype=torch.float32,
        )
        assert torch.allclose(result.to_torch(), expected)

    def test_cluster_axis_1_reduces_cols(self) -> None:
        """cluster_axis=1 sums across the col mesh axis (dim 1, 2 devices).

        Each row block sees: 1+2=3 and 3+4=7, written back to both col slots.
        """
        t = self._tensor()
        result = ttnn.all_reduce(t, cluster_axis=1)
        expected = torch.tensor(
            [[3, 3, 3, 3], [3, 3, 3, 3], [7, 7, 7, 7], [7, 7, 7, 7]],
            dtype=torch.float32,
        )
        assert torch.allclose(result.to_torch(), expected)

    def test_cluster_axis_none_reduces_all(self) -> None:
        """cluster_axis=None reduces across both axes: sum of all four blocks (1+2+3+4=10)."""
        t = self._tensor()
        result = ttnn.all_reduce(t, cluster_axis=None)
        expected = torch.full((4, 4), 10.0)
        assert torch.allclose(result.to_torch(), expected)

    def test_cluster_axis_none_equivalent_to_sequential(self) -> None:
        """cluster_axis=None produces the same result as two sequential axis reduces."""
        t = self._tensor()
        result_none = ttnn.all_reduce(t, cluster_axis=None)
        result_seq = ttnn.all_reduce(ttnn.all_reduce(t, cluster_axis=0), cluster_axis=1)
        assert torch.allclose(result_none.to_torch(), result_seq.to_torch())

    def test_partial_reduce_leaves_other_axis_unchanged(self) -> None:
        """Reducing along cluster_axis=0 does not merge values across cols."""
        t = self._tensor()
        result = ttnn.all_reduce(t, cluster_axis=0)
        r = result.to_torch()
        # After row reduce, col blocks should still be different (4 vs 6).
        assert not torch.allclose(r[:, 0:2], r[:, 2:4])

    def test_mesh_shard_info_preserved(self) -> None:
        """all_reduce preserves mesh_shard_info on the output tensor."""
        t = self._tensor()
        result = ttnn.all_reduce(t, cluster_axis=0)
        assert result.mesh_shard_info is not None
        assert result.mesh_shard_info.mesh_shape == (2, 2)
        assert result.mesh_shard_info.dims == (0, 1)

    def test_output_shape_unchanged(self) -> None:
        """all_reduce does not change the tensor shape."""
        t = self._tensor()
        for axis in (0, 1, None):
            result = ttnn.all_reduce(t, cluster_axis=axis)
            assert result.to_torch().shape == torch.Size([4, 4])

    def test_dtype_override(self) -> None:
        """dtype argument converts the output dtype."""
        t = self._tensor()
        result = ttnn.all_reduce(t, cluster_axis=0, dtype=torch.float64)
        assert result.to_torch().dtype == torch.float64

    def test_larger_mesh_3d_tensor(self) -> None:
        """2x4 mesh sharding a 3D tensor along dims (0, 2)."""
        # shape [2, 3, 4]: rows shard dim 0 (2 devices), cols shard dim 2 (4 devices)
        data = torch.ones(2, 3, 4, dtype=torch.float32)
        t = _shard_2d(data, rows=2, cols=4, dims=(0, 2))
        # cluster_axis=0: sum 2 row devices along dim 0 → each slot doubles
        result = ttnn.all_reduce(t, cluster_axis=0)
        expected = torch.full((2, 3, 4), 2.0)
        assert torch.allclose(result.to_torch(), expected)

    def test_requires_shard_metadata(self) -> None:
        """all_reduce raises ValueError when the tensor has no mesh metadata."""
        t = ttnn.Tensor(torch.ones(4, 4))
        with pytest.raises(ValueError, match="Mesh device is required"):
            ttnn.all_reduce(t)


class TestAllGather2DMesh:
    """all_gather on 2D-mesh tensors with cluster_axis support.

    Same 2x2 mesh setup as TestAllReduce2DMesh.
    """

    def _tensor(self) -> ttnn.Tensor:
        data = torch.tensor(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=torch.float32,
        )
        return _shard_2d(data, rows=2, cols=2, dims=(0, 1))

    def test_cluster_axis_0_gathers_along_dim_0(self) -> None:
        """cluster_axis=0 gathers row shards; output dim 0 grows by rows factor.

        Each device's shard along dim 0 (shard_size=2) is gathered; all
        devices receive all 4 rows.  The result is stacked into both row
        slots along dim 0, giving shape [8, 4].
        """
        t = self._tensor()
        result = ttnn.all_gather(t, dim=0, cluster_axis=0)
        r = result.to_torch()
        assert r.shape == torch.Size([8, 4])
        # First half and second half should be identical (both devices see all rows).
        assert torch.equal(r[0:4, :], r[4:8, :])

    def test_cluster_axis_1_gathers_along_dim_1(self) -> None:
        """cluster_axis=1 gathers col shards; output dim 1 grows by cols factor.

        The result is stacked into both col slots along dim 1, giving shape [4, 8].
        """
        t = self._tensor()
        result = ttnn.all_gather(t, dim=1, cluster_axis=1)
        r = result.to_torch()
        assert r.shape == torch.Size([4, 8])
        # First half and second half of each row should be identical.
        assert torch.equal(r[:, 0:4], r[:, 4:8])

    def test_gather_dim_can_differ_from_shard_dim(self) -> None:
        """Sharding is along dim 0 but gathering is concatenated along dim 1."""
        # 1x4 mesh sharding dim 0 only (row axis inactive); gather result along dim 1.
        mesh = _make_2d_mesh(1, 4)
        data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        t = ttnn.from_torch(
            data,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(1, 4), dims=(None, 0)),
        )
        result = ttnn.all_gather(t, dim=1, cluster_axis=1)
        r = result.to_torch()
        # Gather along dim 1: each col device contributes its row-shard.
        # shard_size = 4 // 4 = 1 row per device.
        # gathered = cat([row0, row1, row2, row3], dim=1) → shape [1, 16]
        # stacked × 4 along shard_dim=0 → shape [4, 16]
        assert r.shape == torch.Size([4, 16])

    def test_cluster_axis_none_gathers_both_axes(self) -> None:
        """cluster_axis=None gathers across both mesh axes sequentially."""
        t = self._tensor()
        result = ttnn.all_gather(t, dim=0, cluster_axis=None)
        r = result.to_torch()
        # After gathering both axes (each doubles dim 0), shape is [4*2*2, 4] = [16, 4].
        # (Gather axis 0 with n=2: dim 0 × 2 = [8,4]; gather axis 1 with n=2:
        # shard_dim=1, gather_dim=0 grows again: [16, 4])
        assert r.shape[0] == 16

    def test_mesh_shard_info_preserved(self) -> None:
        """all_gather preserves the mesh_shard_info on the output tensor."""
        t = self._tensor()
        result = ttnn.all_gather(t, dim=0, cluster_axis=0)
        assert result.mesh_shard_info is not None
        assert result.mesh_shard_info.mesh_shape == (2, 2)
        assert result.mesh_shard_info.dims == (0, 1)

    def test_gathered_content_correct_axis0(self) -> None:
        """Gathered result along cluster_axis=0 contains all device rows in order."""
        t = self._tensor()
        result = ttnn.all_gather(t, dim=0, cluster_axis=0)
        r = result.to_torch()
        # First copy (rows 0:4): original rows [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
        expected_block = torch.tensor(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=torch.float32,
        )
        assert torch.equal(r[0:4, :], expected_block)
        assert torch.equal(r[4:8, :], expected_block)

    def test_requires_shard_metadata(self) -> None:
        """all_gather raises ValueError when the tensor has no mesh metadata."""
        t = ttnn.Tensor(torch.ones(4, 4))
        with pytest.raises(ValueError, match="Mesh device is required"):
            ttnn.all_gather(t, dim=0)
