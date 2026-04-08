# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for binary in-place ops (max, min) with shared operands.

Binary in-place ops use the hardware form DST[dst0] = op(DST[dst0], DST[dst1]),
clobbering operand 0. When multiple in-place ops share operands (e.g.,
max(a,b) + min(a,b)), the DST allocation pass must insert copies to prevent
one op from destroying the other's inputs.

Uses the identity max(a,b) + min(a,b) = a + b for validation.
"""

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import assert_allclose, to_dram

TILE_SIZE = 32
GRANULARITY = 2

KERNEL_TEMPLATE = """
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = {granularity}

@ttl.operation(grid=(1, 1))
def max_plus_min_kernel(
    a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor
) -> None:
    row_tiles = a.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(GRANULARITY, 1), block_count=2
    )

    @ttl.compute()
    def compute():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    mx = ttl.max(a_blk, b_blk)
                    mn = ttl.min(a_blk, b_blk)
                    out_blk.store(mx + mn)

    @ttl.datamovement()
    def read():
        for row in range(row_tiles):
            r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
            for col in range(col_tiles):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    tx_a = ttl.copy(a[r0:r1, col : col + 1], a_blk)
                    tx_b = ttl.copy(b[r0:r1, col : col + 1], b_blk)
                    tx_a.wait()
                    tx_b.wait()

    @ttl.datamovement()
    def write():
        for row in range(row_tiles):
            r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
            for col in range(col_tiles):
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                    tx.wait()
"""

_kernel_cache = {}


def make_kernel(granularity: int):
    """Generate a max+min kernel for the given granularity."""
    if granularity in _kernel_cache:
        return _kernel_cache[granularity]

    code = KERNEL_TEMPLATE.format(granularity=granularity)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"kernel_max_min_g{granularity}_"
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("max_plus_min_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temp_kernel_files.append(temp_path)

    kernel = module.max_plus_min_kernel
    _kernel_cache[granularity] = kernel
    return kernel


@pytest.mark.parametrize("granularity", [1, 2], ids=["g1", "g2"])
def test_max_plus_min(device, granularity):
    """max(a,b) + min(a,b) == a + b (mathematical identity)."""
    kernel = make_kernel(granularity)

    dim = 64
    a_torch = torch.rand((dim, dim), dtype=torch.bfloat16)
    b_torch = torch.rand((dim, dim), dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(torch.zeros_like(a_torch), device)

    kernel(a, b, out)

    result = ttnn.to_torch(out)
    expected = a_torch + b_torch

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
