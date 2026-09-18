# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware and diagnostic coverage for block squeeze/unsqueeze views."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.diagnostics import TTLangCompileError
from ttlang_test_utils import assert_allclose, to_dram, to_l1


@ttl.operation(grid=(1, 1))
def squeeze_kernel(inp, out):
    """Remove two leading unit dimensions from a rank-4 block."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1, 2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            squeezed = ttl.block.squeeze(inp_block, dims=[0, -4, 1, -3])
            out_block.store(ttl.block.squeeze(squeezed, dims=[]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:1, 0:1, 0:2, 0:3], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:3]).wait()


@ttl.operation(grid=(1, 1))
def squeeze_negative_dims_kernel(inp, out):
    """Resolve negative dimensions against the original input rank."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1, 2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.squeeze(inp_block, dims=[-4, -3]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:1, 0:1, 0:2, 0:3], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:3]).wait()


@ttl.operation(grid=(1, 1))
def squeeze_interleaved_dims_kernel(inp, out):
    """Remove interleaved unit dimensions, including the innermost one."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 1, 3, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.squeeze(inp_block, dims=[-1, -3]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:2, 0:1, 0:3, 0:1], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:3]).wait()


@ttl.operation(grid=(1, 1))
def unsqueeze_kernel(inp, out):
    """Insert two leading unit dimensions into a rank-2 block."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.unsqueeze(inp_block, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:2, 0:3], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:1, 0:1, 0:2, 0:3]).wait()


@ttl.operation(grid=(1, 1))
def unsqueeze_interleaved_dims_kernel(inp, out):
    """Insert interleaved unit dimensions using negative positions."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1, 3, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.unsqueeze(inp_block, dims=[-1, -3]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:2, 0:3], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:1, 0:3, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def squeeze_non_unit_dim_kernel(inp, out):
    """Reject removal of a dimension whose size is not one."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2, 2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.squeeze(inp_block, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def squeeze_out_of_range_dim_kernel(inp, out):
    """Reject dimensions outside the input rank."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1, 2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.squeeze(inp_block, dims=[4]))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def unsqueeze_out_of_range_dim_kernel(inp, out):
    """Reject positions outside the resulting rank."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.unsqueeze(inp_block, dims=[3]))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def unsqueeze_duplicate_dim_kernel(inp, out):
    """Reject positions that normalize to the same result axis."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            out_block.store(ttl.block.unsqueeze(inp_block, dims=[0, -4]))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


def _make_source(dtype):
    indices = torch.arange(6 * 32 * 32, dtype=torch.float32)
    # Include FP32 low bits so accidental BF16 conversion changes the result.
    values = (indices.remainder(251) - 125) / 64 + indices.remainder(7) / 8192
    return values.to(dtype).reshape(1, 1, 64, 96)


def _make_tensors(device, dtype=torch.bfloat16, memory="l1"):
    source = _make_source(dtype)
    tensor_factory = to_l1 if memory == "l1" else to_dram
    inp = tensor_factory(source, device)
    out = tensor_factory(torch.zeros((64, 96), dtype=dtype), device)
    return source, inp, out


def _make_interleaved_values(dtype):
    source = _make_source(dtype).reshape(2, 1, 96, 32)
    expected = torch.empty((64, 96), dtype=dtype)
    for tile_index in range(6):
        batch, tile_row = divmod(tile_index, 3)
        expected[
            batch * 32 : (batch + 1) * 32,
            tile_row * 32 : (tile_row + 1) * 32,
        ] = source[batch, 0, tile_row * 32 : (tile_row + 1) * 32, :]
    return source, expected


@pytest.mark.parametrize("kernel", [squeeze_kernel, squeeze_negative_dims_kernel])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_squeeze_preserves_tile_values(device, kernel, dtype, memory):
    """Squeezing unit dimensions changes only the block's logical rank.

    The positive-index kernel also covers duplicate dimensions and an empty
    second squeeze, both of which are no-ops after normalization.
    """
    source, inp, out = _make_tensors(device, dtype, memory)

    kernel(inp, out)

    assert_allclose(ttnn.to_torch(out), source.reshape(64, 96), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_squeeze_interleaved_dimensions_preserves_tile_order(device, dtype, memory):
    """Removing interleaved unit axes preserves row-major tile ordering."""
    source, expected = _make_interleaved_values(dtype)
    tensor_factory = to_l1 if memory == "l1" else to_dram
    inp = tensor_factory(source, device)
    out = tensor_factory(torch.zeros_like(expected), device)

    squeeze_interleaved_dims_kernel(inp, out)

    assert_allclose(ttnn.to_torch(out), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_unsqueeze_leading_dimensions_preserves_tile_values(device, dtype, memory):
    """Inserting leading unit axes changes only the block's logical rank."""
    source = _make_source(dtype)
    tensor_factory = to_l1 if memory == "l1" else to_dram
    inp = tensor_factory(source.reshape(64, 96), device)
    out = tensor_factory(torch.zeros_like(source), device)

    unsqueeze_kernel(inp, out)

    assert_allclose(ttnn.to_torch(out), source, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_unsqueeze_interleaved_dimensions_preserves_tile_order(device, dtype, memory):
    """Negative insertion positions preserve row-major tile ordering."""
    source, expected = _make_interleaved_values(dtype)
    tensor_factory = to_l1 if memory == "l1" else to_dram
    inp = tensor_factory(expected, device)
    output = tensor_factory(torch.zeros_like(source), device)

    unsqueeze_interleaved_dims_kernel(inp, output)

    assert_allclose(ttnn.to_torch(output), source, rtol=0.0, atol=0.0)


def test_squeeze_rejects_non_unit_dimension(device):
    """A selected grid dimension must have size one."""
    _, inp, out = _make_tensors(device)

    with pytest.raises(TTLangCompileError, match="grid size is 2, expected 1"):
        squeeze_non_unit_dim_kernel(inp, out)


def test_squeeze_rejects_out_of_range_dimension(device):
    """Dimension diagnostics include the invalid index and input rank."""
    _, inp, out = _make_tensors(device)

    with pytest.raises(TTLangCompileError, match="dimension 4.*only 4 dimensions"):
        squeeze_out_of_range_dim_kernel(inp, out)


def test_unsqueeze_rejects_out_of_range_dimension(device):
    """Insertion positions are validated against the resulting rank."""
    _, inp, out = _make_tensors(device)

    with pytest.raises(TTLangCompileError, match="dimension 3.*have 3 dimensions"):
        unsqueeze_out_of_range_dim_kernel(inp, out)


def test_unsqueeze_rejects_duplicate_dimension(device):
    """Equivalent positive and negative insertion positions are rejected."""
    _, inp, out = _make_tensors(device)

    with pytest.raises(TTLangCompileError, match="duplicate dimension -4"):
        unsqueeze_duplicate_dim_kernel(inp, out)
