# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression for #536: consecutive waits on one DFB need intervening pops."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402

TILE = 32


def _make_kernel(*, explicit_pop_at_end=False):
    @ttl.operation(grid=(1, 1))
    def repro(inp, out):
        shape = (1, 1)
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=shape, block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=shape, block_count=2)

        @ttl.compute()
        def compute():
            with inp_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

            with inp_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

        @ttl.datamovement()
        def dm_read():
            blk = inp_cb.reserve()
            ttl.copy(inp[0, 0], blk).wait()
            blk = inp_cb.reserve()
            ttl.copy(inp[1, 0], blk).wait()

        if explicit_pop_at_end:

            @ttl.datamovement()
            def dm_write():
                blk = out_cb.wait()
                ttl.copy(blk, out[0, 0]).wait()
                blk = out_cb.wait()
                ttl.copy(blk, out[1, 0]).wait()
                blk.pop()

        else:

            @ttl.datamovement()
            def dm_write():
                blk = out_cb.wait()
                ttl.copy(blk, out[0, 0]).wait()
                blk = out_cb.wait()
                ttl.copy(blk, out[1, 0]).wait()

    return repro


def _run_kernel_and_check(device, kernel):
    torch.manual_seed(536)
    inp = torch.randn((2 * TILE, TILE), dtype=torch.bfloat16)
    inp_t = to_dram(inp, device)
    out_t = to_dram(torch.full((2 * TILE, TILE), -42.0, dtype=torch.bfloat16), device)
    kernel(inp_t, out_t)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out_t)
    assert torch.equal(result, inp), f"result:\n{result}\nexpected:\n{inp}"


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "explicit_pop_at_end",
    [False, True],
    ids=["implicit-pops", "later-explicit-pop"],
)
def test_two_waits_same_dfb_pop_placement(device, explicit_pop_at_end):
    kernel = _make_kernel(explicit_pop_at_end=explicit_pop_at_end)
    _run_kernel_and_check(device, kernel)
