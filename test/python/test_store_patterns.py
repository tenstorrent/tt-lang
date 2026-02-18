# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for store patterns: passthrough, double store, multi-output, and
store-then-forward (scratch CB reuse within compute thread)."""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

from ttl import ttl


@ttl.kernel(grid=(1, 1))
def passthrough_kernel(inp, out):
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as x, out_cb.reserve() as o:
            o.store(x)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def double_store_kernel(a, b, out):
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, b_cb.wait() as bv:
            with out_cb.reserve() as o:
                o.store(av + bv)
                o.store(av * bv)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def same_tile_two_outputs_kernel(a, b, out1, out2):
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out1_cb = ttl.make_circular_buffer_like(out1, shape=(1, 1), buffer_factor=2)
    out2_cb = ttl.make_circular_buffer_like(out2, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, b_cb.wait() as bv:
            result = av + bv
            with out1_cb.reserve() as o1, out2_cb.reserve() as o2:
                o1.store(result)
                o2.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out1_cb.wait() as blk:
            tx = ttl.copy(blk, out1[0, 0])
            tx.wait()
        with out2_cb.wait() as blk:
            tx = ttl.copy(blk, out2[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def store_then_forward_kernel(a, b, out_main, out_copy):
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    main_cb = ttl.make_circular_buffer_like(out_main, shape=(1, 1), buffer_factor=2)
    copy_cb = ttl.make_circular_buffer_like(out_copy, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, b_cb.wait() as bv:
            with main_cb.reserve() as m:
                m.store(av + bv)

        with main_cb.wait() as mv:
            with copy_cb.reserve() as c:
                c.store(mv + mv)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with main_cb.wait() as blk:
            tx = ttl.copy(blk, out_main[0, 0])
            tx.wait()
        with copy_cb.wait() as blk:
            tx = ttl.copy(blk, out_copy[0, 0])
            tx.wait()


@pytest.fixture
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_passthrough(device):
    inp = to_dram(torch.full((32, 32), 42.0, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    passthrough_kernel(inp, out)
    result = ttnn.to_torch(out).float()
    assert torch.allclose(result, torch.full_like(result, 42.0))


def test_double_store(device):
    a = to_dram(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
    b = to_dram(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    double_store_kernel(a, b, out)
    result = ttnn.to_torch(out).float()
    expected = torch.full_like(result, 6.0)
    assert torch.allclose(result, expected)


def test_same_tile_two_outputs(device):
    a = to_dram(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
    b = to_dram(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
    out1 = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    out2 = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    same_tile_two_outputs_kernel(a, b, out1, out2)
    r1 = ttnn.to_torch(out1).float()
    r2 = ttnn.to_torch(out2).float()
    expected = torch.full_like(r1, 5.0)
    assert torch.allclose(r1, expected)
    assert torch.allclose(r2, expected)


def test_store_then_forward(device):
    a = to_dram(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
    b = to_dram(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
    out_main = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    out_copy = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    store_then_forward_kernel(a, b, out_main, out_copy)
    rm = ttnn.to_torch(out_main).float()
    rc = ttnn.to_torch(out_copy).float()
    assert torch.allclose(rm, torch.full_like(rm, 5.0))
    assert torch.allclose(rc, torch.full_like(rc, 10.0))
