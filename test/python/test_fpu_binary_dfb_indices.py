# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FPU binary operands must retain their independent dataflow buffer offsets."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl

from ttlang_test_utils import to_dram, to_l1, to_l1_sharded
from utils.correctness import assert_allclose


@ttl.operation(grid=(1, 1))
def fpu_binary_dfb_indices(out):
    lhs_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

    @ttl.compute()
    def compute():
        with lhs_dfb.reserve() as lhs_block:
            lhs_block.store(
                ttl.block.fill(2.0, shape=lhs_block.shape, dtype=lhs_block.dtype)
            )
        with lhs_dfb.reserve() as lhs_block:
            lhs_block.store(
                ttl.block.fill(3.0, shape=lhs_block.shape, dtype=lhs_block.dtype)
            )

        with rhs_dfb.reserve() as rhs_block:
            rhs_block.store(
                ttl.block.fill(5.0, shape=rhs_block.shape, dtype=rhs_block.dtype)
            )
        with rhs_dfb.reserve() as rhs_block:
            rhs_block.store(
                ttl.block.fill(8.0, shape=rhs_block.shape, dtype=rhs_block.dtype)
            )

        lhs_first = lhs_dfb.wait()
        lhs_second = lhs_dfb.wait()
        rhs_first = rhs_dfb.wait()
        rhs_second = rhs_dfb.wait()

        with out_dfb.reserve() as out_block:
            out_block.store(lhs_first * rhs_second)
        with out_dfb.reserve() as out_block:
            out_block.store(rhs_second - lhs_first)
        with out_dfb.reserve() as out_block:
            out_block.store(lhs_second + rhs_first)

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 0]).wait()
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 1]).wait()
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 2]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory_config", ["dram", "l1", "height", "width", "block"])
@pytest.mark.requires_device
def test_fpu_binary_dfb_indices(device, dtype, memory_config):
    result = torch.full((32, 96), -42.0, dtype=dtype)
    if memory_config == "dram":
        out = to_dram(result, device)
    elif memory_config == "l1":
        out = to_l1(result, device)
    else:
        out = to_l1_sharded(result, device, layout=memory_config)

    fpu_binary_dfb_indices(out)
    ttnn.synchronize_device(device)

    expected = torch.cat(
        [torch.full((32, 32), value) for value in (16.0, 6.0, 8.0)], dim=1
    )
    actual = ttnn.to_torch(out).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
