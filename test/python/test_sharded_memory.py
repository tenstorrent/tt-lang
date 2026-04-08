# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for sharded and mixed memory layout combinations.

Exercises the TensorAccessor path for sharded L1 tensors (height, width, block)
and mixed configurations where inputs and outputs use different memory layouts
(interleaved L1, interleaved DRAM, sharded L1).

Tests:
- Single-core: all shard layouts with a representative binary op (add)
- Single-core: mixed memory configs (sharded + interleaved in same kernel)
- Multicore: height-sharded across 4 cores
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import assert_allclose, to_dram, to_l1, to_l1_sharded


# =============================================================================
# Kernel templates
# =============================================================================

ADD_KERNEL_TEMPLATE = """
import ttl

@ttl.operation(grid=({grid_rows}, {grid_cols}))
def add_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r, out_dfb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as blk:
            tx = ttl.copy(lhs[0, 0], blk)
            tx.wait()
        with rhs_dfb.reserve() as blk:
            tx = ttl.copy(rhs[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()
"""


def _make_add_kernel(grid_rows=1, grid_cols=1):
    code = ADD_KERNEL_TEMPLATE.format(grid_rows=grid_rows, grid_cols=grid_cols)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_add_{grid_rows}x{grid_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name
    spec = importlib.util.spec_from_file_location("add_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temp_kernel_files.append(temp_path)
    return getattr(module, "add_kernel")


# =============================================================================
# Tensor creation helpers
# =============================================================================


def _to_sharded_multicore(torch_tensor, device, num_cores):
    """Height-shard a tensor across num_cores cores (single column of cores)."""
    rows, cols = torch_tensor.shape[-2], torch_tensor.shape[-1]
    shard_rows = rows // num_cores
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))}
        ),
        (shard_rows, cols),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    dram_tensor = to_dram(torch_tensor, device)
    return ttnn.to_memory_config(dram_tensor, memory_config=sharded_config)


# =============================================================================
# Memory config factories keyed by name
# =============================================================================


def _make_tensor(torch_tensor, device, mem_type):
    """Create a TTNN tensor with the given memory configuration."""
    if mem_type == "dram":
        return to_dram(torch_tensor, device)
    elif mem_type == "l1":
        return to_l1(torch_tensor, device)
    elif mem_type == "height_sharded":
        return to_l1_sharded(torch_tensor, device, layout="height")
    elif mem_type == "width_sharded":
        return to_l1_sharded(torch_tensor, device, layout="width")
    elif mem_type == "block_sharded":
        return to_l1_sharded(torch_tensor, device, layout="block")
    else:
        raise ValueError(f"Unknown mem_type: {mem_type!r}")


# =============================================================================
# Tests: mixed memory configurations (single core)
# =============================================================================

# All interesting combos of (lhs_mem, rhs_mem, out_mem).
# We test a representative subset to keep runtime reasonable.
MIXED_MEMORY_CONFIGS = [
    # All same layout
    ("l1", "l1", "l1"),
    ("dram", "dram", "dram"),
    ("height_sharded", "height_sharded", "height_sharded"),
    ("width_sharded", "width_sharded", "width_sharded"),
    ("block_sharded", "block_sharded", "block_sharded"),
    # Sharded inputs, interleaved output
    ("height_sharded", "height_sharded", "l1"),
    ("height_sharded", "height_sharded", "dram"),
    # Interleaved inputs, sharded output
    ("l1", "l1", "height_sharded"),
    ("dram", "dram", "height_sharded"),
    # Mixed: one sharded, one interleaved
    ("height_sharded", "l1", "l1"),
    ("l1", "height_sharded", "l1"),
    ("height_sharded", "dram", "dram"),
    ("dram", "height_sharded", "height_sharded"),
    # Mixed: sharded input, dram output
    ("width_sharded", "width_sharded", "dram"),
    ("block_sharded", "block_sharded", "dram"),
    # Cross-shard layouts
    ("height_sharded", "width_sharded", "l1"),
    ("height_sharded", "block_sharded", "dram"),
]

_single_core_kernel = None


def _get_single_core_kernel():
    global _single_core_kernel
    if _single_core_kernel is None:
        _single_core_kernel = _make_add_kernel(1, 1)
    return _single_core_kernel


@pytest.mark.parametrize(
    "lhs_mem,rhs_mem,out_mem",
    MIXED_MEMORY_CONFIGS,
    ids=[f"{l}+{r}->{o}" for l, r, o in MIXED_MEMORY_CONFIGS],
)
def test_mixed_memory_add(device, lhs_mem, rhs_mem, out_mem):
    """Test add with mixed memory configurations (sharded + interleaved)."""
    kernel = _get_single_core_kernel()

    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch

    lhs = _make_tensor(lhs_torch, device, lhs_mem)
    rhs = _make_tensor(rhs_torch, device, rhs_mem)
    out = _make_tensor(out_torch, device, out_mem)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Test: multicore height-sharded
# =============================================================================

MULTICORE_ADD_KERNEL = """
import ttl

@ttl.operation(grid=(1, 4))
def multicore_add_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r, out_dfb.reserve() as o:
            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with lhs_dfb.reserve() as blk:
            tx = ttl.copy(lhs[y, x], blk)
            tx.wait()
        with rhs_dfb.reserve() as blk:
            tx = ttl.copy(rhs[y, x], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[y, x])
            tx.wait()
"""

_multicore_kernel = None


def _get_multicore_add_kernel():
    global _multicore_kernel
    if _multicore_kernel is not None:
        return _multicore_kernel
    code = MULTICORE_ADD_KERNEL
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_multicore_add_"
    ) as f:
        f.write(code)
        temp_path = f.name
    spec = importlib.util.spec_from_file_location("multicore_add_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temp_kernel_files.append(temp_path)
    _multicore_kernel = getattr(module, "multicore_add_kernel")
    return _multicore_kernel


def test_multicore_height_sharded_add(device):
    """Test add with height-sharded tensors across 4 cores (1x4 grid).

    Tensor is 128x32 (4 tile rows), height-sharded so each of the 4 cores
    gets one 32x32 tile. Each core uses ttl.node() to index its own tile.
    """
    kernel = _get_multicore_add_kernel()

    lhs_torch = torch.full((128, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((128, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((128, 32), dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch

    lhs = _to_sharded_multicore(lhs_torch, device, 4)
    rhs = _to_sharded_multicore(rhs_torch, device, 4)
    out = _to_sharded_multicore(out_torch, device, 4)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
