# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression for #536: consecutive waits on one DFB need intervening pops."""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32


def _make_kernel():
    @ttl.operation(grid=(1, 1))
    def repro(out):
        shape = (1, 1)
        shared_cb = ttl.make_dataflow_buffer_like(out, shape=shape, block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=shape, block_count=2)

        @ttl.compute()
        def compute():
            with shared_cb.reserve() as v:
                v.store(ttl.math.fill(v, 7.0))
            with shared_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

            with shared_cb.reserve() as v:
                v.store(ttl.math.fill(v, 8.0))
            with shared_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0:1, 0:1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[1:2, 0:1]).wait()

    return repro


def _make_later_explicit_pop_kernel():
    @ttl.operation(grid=(1, 1))
    def repro(out):
        shape = (1, 1)
        shared_cb = ttl.make_dataflow_buffer_like(out, shape=shape, block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=shape, block_count=2)

        @ttl.compute()
        def compute():
            with shared_cb.reserve() as v:
                v.store(ttl.math.fill(v, 7.0))
            with shared_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

            with shared_cb.reserve() as v:
                v.store(ttl.math.fill(v, 8.0))
            with shared_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0:1, 0:1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[1:2, 0:1]).wait()
            blk.pop()

    return repro


@pytest.mark.requires_device
def test_two_waits_same_dfb_pops_between(device):
    kernel = _make_kernel()
    _run_kernel_and_check(device, kernel)


@pytest.mark.requires_device
def test_later_explicit_pop_does_not_satisfy_first_wait(device):
    kernel = _make_later_explicit_pop_kernel()
    _run_kernel_and_check(device, kernel)


def _run_kernel_and_check(device, kernel):
    out_t = to_dram(torch.full((2 * TILE, TILE), -42.0, dtype=torch.bfloat16), device)
    kernel(out_t)
    ttnn.synchronize_device(device)
    expected = torch.empty((2 * TILE, TILE), dtype=torch.float32)
    expected[:TILE, :] = 7.0
    expected[TILE:, :] = 8.0
    assert_allclose(ttnn.to_torch(out_t).float(), expected)
