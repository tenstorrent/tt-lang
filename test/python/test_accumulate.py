# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for store accumulation: tile_store {acc = true}.

Validates the accumulation pipeline end-to-end via the Python DSL:
  o.store(expr, acc=True) -> zero-init + add_binary_tile + deferred pack

Parameterized over tile shapes and data types.

Groups:
  1. Basic accumulation (single store)
  2. Different expressions (binary ops) — TODO, requires compute fusion
  3. Edge cases (passthrough)
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
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose, assert_with_ulp


# =============================================================================
# Parameterization axes
# =============================================================================

TILE_SIZE = 32

TILE_SHAPE_PARAMS = [
    pytest.param((1, 1), id="1x1tiles"),
    pytest.param((2, 2), id="2x2tiles"),
    pytest.param((1, 4), id="1x4tiles"),
    pytest.param((4, 1), id="4x1tiles"),
    # Shapes large enough to trigger DST subblocking.
    # bf16: capacity=8, unrollFactor=8 → need >8 tiles.
    # f32:  capacity=4, unrollFactor=4 → need >4 tiles.
    pytest.param((2, 4), id="2x4tiles"),  # 8 tiles — triggers f32 subblocking
    pytest.param((4, 2), id="4x2tiles"),  # 8 tiles — triggers f32 subblocking
    pytest.param((4, 4), id="4x4tiles"),  # 16 tiles — triggers both bf16 and f32
]

# Cross-compute accumulation currently only supports 1x1 tile domains.
# Multi-tile requires Phase 2 (outer tile loop) of the accumulation plan.
SINGLE_TILE_PARAMS = [
    pytest.param((1, 1), id="1x1tiles"),
]

DTYPE_PARAMS = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float32, id="f32"),
]

ULP_THRESHOLD = 10  # default from assert_with_ulp


# =============================================================================
# Kernel templates — parameterized by tile shape
# =============================================================================


def _slice_syntax(rows, cols):
    if rows == 1 and cols == 1:
        return "0, 0"
    return f"0:{rows}, 0:{cols}"


def _tensor_shape(tile_rows, tile_cols):
    return (tile_rows * TILE_SIZE, tile_cols * TILE_SIZE)


# Binary acc store: out = 0 + (a + b) = a + b
ACC_ADD_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out_dfb.reserve() as o:
                o.store(av + bv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# Two acc stores (multi-store): out = 0 + a + b = a + b
ACC_TWO_STORES_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out_dfb.reserve() as o:
                o.store(av, acc=True)
                o.store(bv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# Three acc stores: out = 0 + a + b + c
ACC_THREE_STORES_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, b, c, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, c_dfb.wait() as cv:
            with out_dfb.reserve() as o:
                o.store(av, acc=True)
                o.store(bv, acc=True)
                o.store(cv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[{S}], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# Two different binary ops: out = (a + b) + (a * b)
ACC_BINARY_OPS_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out_dfb.reserve() as o:
                o.store(av + bv, acc=True)
                o.store(av * bv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# Two outputs, different views: out1 = a + b, out2 = a * b
ACC_MULTI_OUTPUT_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, b, out1, out2):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    out1_dfb = ttl.make_dataflow_buffer_like(out1, shape=({R}, {C}), buffer_factor=2)
    out2_dfb = ttl.make_dataflow_buffer_like(out2, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out1_dfb.reserve() as o1, out2_dfb.reserve() as o2:
                o1.store(av + bv, acc=True)
                o2.store(av * bv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out1_dfb.wait() as blk:
            tx = ttl.copy(blk, out1[{S}])
            tx.wait()
        with out2_dfb.wait() as blk:
            tx = ttl.copy(blk, out2[{S}])
            tx.wait()
"""

# Mixed unary/binary chain: out = exp(a * b) + abs(a - b)
ACC_MIXED_CHAIN_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with out_dfb.reserve() as o:
                o.store(ttl.exp(av * bv) + ttl.abs(av - bv), acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# Passthrough: out = 0 + a = a
ACC_PASSTHROUGH_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(a, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av:
            with out_dfb.reserve() as o:
                o.store(av, acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""


# =============================================================================
# Kernel factory — no caching, each call produces a fresh kernel function.
# The ttl runtime maintains its own per-device compiled-kernel cache, so
# caching kernel *function objects* across tests (which get different device
# instances) causes stale compiled artifacts to be reused.
# =============================================================================

_next_kernel_id = 0


def _make_kernel(template, R, C):
    global _next_kernel_id
    _next_kernel_id += 1

    code = template.format(R=R, C=C, S=_slice_syntax(R, C))
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="kernel_acc_",
    ) as f:
        f.write(code)
        path = f.name

    temp_kernel_files.append(path)
    mod_name = f"acc_kernel_{R}x{C}_{_next_kernel_id}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel


# =============================================================================
# Test infrastructure
# =============================================================================


def make_inputs(n, shape, dtype, seed=42):
    """Generate n random input tensors with a fixed seed for reproducibility."""
    torch.manual_seed(seed)
    return [torch.randn(shape, dtype=dtype) for _ in range(n)]


class AccTestBase:
    """Declarative base for accumulation tests.

    Subclasses define:
        template:    kernel code template
        num_inputs:  number of input tensors (default 2)
        golden(*inputs) -> expected torch output
        f32_accumulation: True if the hardware accumulates across multiple
            stores at f32 precision in DST registers. When set, the golden
            is computed at f32 and rounded to bf16 for comparison, matching
            the hardware's precision semantics.
    """

    template = None
    num_inputs = 2
    f32_accumulation = False

    def golden(self, *inputs):
        raise NotImplementedError

    @pytest.fixture(params=TILE_SHAPE_PARAMS)
    def tile_shape(self, request):
        return request.param

    @pytest.fixture(params=DTYPE_PARAMS)
    def dtype(self, request):
        return request.param

    def test_execute(self, device, tile_shape, dtype):
        assert self.template is not None, f"{type(self).__name__} must set 'template'"
        R, C = tile_shape
        kernel = _make_kernel(self.template, R, C)
        shape = _tensor_shape(R, C)

        inputs = make_inputs(self.num_inputs, shape, dtype)
        expected = self.golden(*inputs)

        dev_inputs = [to_dram(t, device) for t in inputs]
        dev_out = to_dram(torch.zeros(shape, dtype=dtype), device)

        kernel(*dev_inputs, dev_out)

        result = ttnn.to_torch(dev_out)
        if self.f32_accumulation:
            # Cross-compute accumulation operates at f32 precision in DST
            # registers while torch golden uses bf16 arithmetic. The
            # precision difference can cause up to ~0.03 absolute error for
            # bf16 values near rounding boundaries.
            assert_allclose(result, expected, atol=0.05, rtol=0.05)
        elif dtype == torch.float32:
            # f32 inputs are truncated to bf16 on hardware, so results near
            # zero (from cancellation of positive and negative values) have
            # large ULP deltas despite tiny absolute differences.
            assert_allclose(result, expected, atol=0.01, rtol=0.02)
        else:
            assert_with_ulp(
                expected.bfloat16(),
                result.bfloat16(),
                ulp_threshold=ULP_THRESHOLD,
            )


# =============================================================================
# Group 1: Basic accumulation
# =============================================================================


class TestAccSingleStore(AccTestBase):
    """out = 0 + (a + b) = a + b"""

    template = ACC_ADD_TEMPLATE

    def golden(self, a, b):
        return a + b


class TestAccTwoStores(AccTestBase):
    """out = 0 + a + b — two separate acc stores to the same view.

    After ConvertTTLToCompute, each store becomes a separate ttl.compute.
    FormAccumulationGroups detects these as a multi-compute accumulation group,
    AssignDST allocates a shared accumulator register, and InsertSync wraps
    them in a single sync region.
    """

    template = ACC_TWO_STORES_TEMPLATE
    f32_accumulation = True

    @pytest.fixture(params=SINGLE_TILE_PARAMS)
    def tile_shape(self, request):
        return request.param

    def golden(self, a, b):
        return a + b


class TestAccThreeStores(AccTestBase):
    """out = 0 + a + b + c — three separate acc stores to the same view."""

    template = ACC_THREE_STORES_TEMPLATE
    num_inputs = 3
    f32_accumulation = True

    @pytest.fixture(params=SINGLE_TILE_PARAMS)
    def tile_shape(self, request):
        return request.param

    def golden(self, a, b, c):
        return a + b + c


# =============================================================================
# Group 2: Different expressions
# =============================================================================


class TestAccMixedChain(AccTestBase):
    """out = exp(a * b) + abs(a - b)"""

    template = ACC_MIXED_CHAIN_TEMPLATE

    def golden(self, a, b):
        return torch.exp(a * b) + torch.abs(a - b)


class TestAccBinaryOps(AccTestBase):
    """out = (a + b) + (a * b) — two expression acc stores to the same view.

    Each store computes a different expression. After ConvertTTLToCompute,
    each becomes a separate ttl.compute with its own expression graph.
    FormAccumulationGroups groups them for cross-compute accumulation.
    """

    template = ACC_BINARY_OPS_TEMPLATE
    f32_accumulation = True

    @pytest.fixture(params=SINGLE_TILE_PARAMS)
    def tile_shape(self, request):
        return request.param

    def golden(self, a, b):
        return (a + b) + (a * b)


# =============================================================================
# Group 3: Edge cases
# =============================================================================


class TestAccMultiOutput:
    """out1 = a + b, out2 = a * b — acc stores to different output views."""

    @pytest.fixture(params=TILE_SHAPE_PARAMS)
    def tile_shape(self, request):
        return request.param

    @pytest.fixture(params=DTYPE_PARAMS)
    def dtype(self, request):
        return request.param

    def test_execute(self, device, tile_shape, dtype):
        R, C = tile_shape
        kernel = _make_kernel(ACC_MULTI_OUTPUT_TEMPLATE, R, C)
        shape = _tensor_shape(R, C)

        a, b = make_inputs(2, shape, dtype)

        da = to_dram(a, device)
        db = to_dram(b, device)
        dout1 = to_dram(torch.zeros(shape, dtype=dtype), device)
        dout2 = to_dram(torch.zeros(shape, dtype=dtype), device)

        kernel(da, db, dout1, dout2)

        if dtype == torch.float32:
            assert_allclose(ttnn.to_torch(dout1), a + b, atol=0.01, rtol=0.01)
            assert_allclose(ttnn.to_torch(dout2), a * b, atol=0.01, rtol=0.01)
        else:
            assert_with_ulp(
                (a + b).bfloat16(),
                ttnn.to_torch(dout1).bfloat16(),
                ulp_threshold=ULP_THRESHOLD,
            )
            assert_with_ulp(
                (a * b).bfloat16(),
                ttnn.to_torch(dout2).bfloat16(),
                ulp_threshold=ULP_THRESHOLD,
            )


class TestAccPassthrough(AccTestBase):
    """out = 0 + a = a"""

    template = ACC_PASSTHROUGH_TEMPLATE
    num_inputs = 1

    def golden(self, a):
        return a
