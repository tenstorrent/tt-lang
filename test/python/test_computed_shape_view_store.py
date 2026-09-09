# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Execute singleton shape views of computed, compiler-materialized blocks."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram, to_l1


@ttl.operation(grid=(1, 1))
def computed_squeeze_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1, 2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            computed = ttl.math.neg(inp_block)
            out_block.store(ttl.block.squeeze(computed, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:1, 0:1, 0:2, 0:3], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:3]).wait()


@ttl.operation(grid=(1, 1))
def computed_unsqueeze_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 2, 3), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_block, out_dfb.reserve() as out_block:
            computed = ttl.math.neg(inp_block)
            out_block.store(ttl.block.unsqueeze(computed, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as block:
            ttl.copy(inp[0:2, 0:3], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:1, 0:1, 0:2, 0:3]).wait()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("memory", ["dram", "l1"])
@pytest.mark.parametrize("direction", ["squeeze", "unsqueeze"])
def test_computed_shape_view_store(device, dtype, memory, direction):
    """Materialize the computed producer, not an alias of the input CB wait."""
    indices = torch.arange(6 * 32 * 32, dtype=torch.float32)
    # Distinct values span all six tiles. FP32 low bits expose BF16 truncation.
    values = (indices.remainder(251) - 125) / 64 + indices.remainder(7) / 8192
    if direction == "squeeze":
        source = values.to(dtype).reshape(1, 1, 64, 96)
        expected = -source.reshape(64, 96)
        kernel = computed_squeeze_kernel
    else:
        source = values.to(dtype).reshape(64, 96)
        expected = -source.reshape(1, 1, 64, 96)
        kernel = computed_unsqueeze_kernel

    tensor_factory = to_l1 if memory == "l1" else to_dram
    inp = tensor_factory(source, device)
    out = tensor_factory(torch.zeros_like(expected), device)
    kernel(inp, out)

    assert_allclose(ttnn.to_torch(out), expected, rtol=0.0, atol=0.0)
