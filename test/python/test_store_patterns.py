# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for store patterns: passthrough, double store, multi-output, and
store-then-forward (scratch DFB reuse within compute thread)."""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

from ttl import ttl


@ttl.operation(grid=(1, 1))
def passthrough_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(x)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


@ttl.operation(grid=(1, 1))
def double_store_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out_dfb.reserve() as o:
                o.store(av + bv)
                o.store(av * bv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def _make_two_output_add_kernel(rows, cols):
    """Factory: create a kernel that adds a+b and stores to 2 outputs.
    DFB shape is (rows, cols) tiles, tensor shape is (rows*32, cols*32)."""

    @ttl.operation(grid=(1, 1))
    def kernel(a, b, out1, out2):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(rows, cols), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(rows, cols), block_count=2)
        o1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(rows, cols), block_count=2)
        o2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(rows, cols), block_count=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as av, b_dfb.wait() as bv:
                result = av + bv
                with o1_dfb.reserve() as o1, o2_dfb.reserve() as o2:
                    o1.store(result)
                    o2.store(result)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:rows, 0:cols], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[0:rows, 0:cols], blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with o1_dfb.wait() as blk:
                tx = ttl.copy(blk, out1[0:rows, 0:cols])
                tx.wait()
            with o2_dfb.wait() as blk:
                tx = ttl.copy(blk, out2[0:rows, 0:cols])
                tx.wait()

    return kernel


# (1,1)=1 tile, (2,2)=4 tiles (fits DST), (4,4)=16 tiles (triggers subblocking)
TWO_OUTPUT_SHAPES = [(1, 1), (2, 2), (4, 4)]
_two_output_kernels = {s: _make_two_output_add_kernel(*s) for s in TWO_OUTPUT_SHAPES}


@ttl.operation(grid=(1, 1))
def three_outputs_kernel(a, b, out1, out2, out3):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(1, 1), block_count=2)
    out2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(1, 1), block_count=2)
    out3_dfb = ttl.make_dataflow_buffer_like(out3, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            result = av + bv
            with (
                out1_dfb.reserve() as o1,
                out2_dfb.reserve() as o2,
                out3_dfb.reserve() as o3,
            ):
                o1.store(result)
                o2.store(result)
                o3.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out1_dfb.wait() as blk:
            tx = ttl.copy(blk, out1[0, 0])
            tx.wait()
        with out2_dfb.wait() as blk:
            tx = ttl.copy(blk, out2[0, 0])
            tx.wait()
        with out3_dfb.wait() as blk:
            tx = ttl.copy(blk, out3[0, 0])
            tx.wait()


@ttl.operation(grid=(1, 1))
def fused_bcast_two_outputs_kernel(a, b, out1, out2):
    """Fused chain: broadcast(b) + a -> store to 2 outputs.
    a is 4x1 tiles (128x32), b is 1x1 tile (32x32), output is 4x1 tiles.
    Tests broadcast inside a fused compute with multi-output."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(4, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(4, 1), block_count=2)
    out2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(4, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out1_dfb.reserve() as o1, out2_dfb.reserve() as o2:
                b_bcast = ttl.math.broadcast(bv, o1, dims=[0])
                result = av + b_bcast
                o1.store(result)
                o2.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:4, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out1_dfb.wait() as blk:
            tx = ttl.copy(blk, out1[0:4, 0])
            tx.wait()
        with out2_dfb.wait() as blk:
            tx = ttl.copy(blk, out2[0:4, 0])
            tx.wait()


@ttl.operation(grid=(1, 1))
def store_then_forward_kernel(a, b, out_main, out_copy):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    main_dfb = ttl.make_dataflow_buffer_like(out_main, shape=(1, 1), block_count=2)
    copy_dfb = ttl.make_dataflow_buffer_like(out_copy, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with main_dfb.reserve() as m:
                m.store(av + bv)

        with main_dfb.wait() as mv:
            with copy_dfb.reserve() as c:
                c.store(mv + mv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with main_dfb.wait() as blk:
            tx = ttl.copy(blk, out_main[0, 0])
            tx.wait()
        with copy_dfb.wait() as blk:
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


@pytest.mark.parametrize(
    "shape",
    TWO_OUTPUT_SHAPES,
    ids=[f"{r}x{c}" for r, c in TWO_OUTPUT_SHAPES],
)
def test_two_outputs(device, shape):
    rows, cols = shape
    h, w = rows * 32, cols * 32
    torch.manual_seed(42)
    a_data = torch.randn((h, w), dtype=torch.bfloat16)
    b_data = torch.randn((h, w), dtype=torch.bfloat16)
    a = to_dram(a_data, device)
    b = to_dram(b_data, device)
    out1 = to_dram(torch.zeros((h, w), dtype=torch.bfloat16), device)
    out2 = to_dram(torch.zeros((h, w), dtype=torch.bfloat16), device)

    _two_output_kernels[shape](a, b, out1, out2)
    expected = (a_data.float() + b_data.float()).bfloat16().float()
    r1 = ttnn.to_torch(out1).float()
    r2 = ttnn.to_torch(out2).float()
    assert torch.allclose(r1, expected, rtol=1e-2, atol=1e-2)
    assert torch.allclose(r2, expected, rtol=1e-2, atol=1e-2)


def test_three_outputs(device):
    torch.manual_seed(42)
    a_data = torch.randn((32, 32), dtype=torch.bfloat16)
    b_data = torch.randn((32, 32), dtype=torch.bfloat16)
    a = to_dram(a_data, device)
    b = to_dram(b_data, device)
    outs = [
        to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device) for _ in range(3)
    ]

    three_outputs_kernel(a, b, *outs)
    expected = (a_data.float() + b_data.float()).bfloat16().float()
    for out in outs:
        r = ttnn.to_torch(out).float()
        assert torch.allclose(r, expected, rtol=1e-2, atol=1e-2)


def test_fused_bcast_two_outputs(device):
    torch.manual_seed(42)
    a_data = torch.randn((128, 32), dtype=torch.bfloat16)
    b_data = torch.randn((32, 32), dtype=torch.bfloat16)
    a = to_dram(a_data, device)
    b = to_dram(b_data, device)
    out1 = to_dram(torch.zeros((128, 32), dtype=torch.bfloat16), device)
    out2 = to_dram(torch.zeros((128, 32), dtype=torch.bfloat16), device)

    fused_bcast_two_outputs_kernel(a, b, out1, out2)
    r1 = ttnn.to_torch(out1).float()
    r2 = ttnn.to_torch(out2).float()
    # Both outputs must be identical (the core invariant for #396).
    assert torch.equal(r1, r2), "multi-output broadcast: outputs differ"
    # Sanity: result is not all zeros (kernel actually ran).
    assert not torch.all(r1 == 0), "multi-output broadcast: output is all zeros"


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
