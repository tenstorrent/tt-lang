# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-compute loop accumulation.

Validates the accumulation pipeline for patterns where acc=True stores appear
inside Python loops. ConvertTTLToCompute creates one ttl.compute per store,
FormAccumulationGroups detects accumulation chains amulti-computes, AssignDST
allocates shared accumulator registers, and InsertSync places group-level sync.

Groups:
  1. Init + loop: o.store(l); for K: o.store(r, acc=True)
  2. Loop only: for K: o.store(r, acc=True)
  3. Init + loop + post: o.store(l); for K: o.store(r, acc=True); o.store(s, acc=True)
  4. Two loops: for K: o.store(a, acc=True); for M: o.store(b, acc=True)
  5. Nested loops: for K: for M: o.store(r, acc=True)
  6. Multi-store per iteration: for K: o.store(a, acc=True); o.store(b, acc=True)
  7. Init + expression loop: o.store(l); for K: o.store(a+b, acc=True)
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
# Parameterization
# =============================================================================

TILE_SIZE = 32

TILE_SHAPE_PARAMS = [
    pytest.param((1, 1), id="1x1tiles"),
    pytest.param((2, 2), id="2x2tiles"),
    pytest.param((1, 4), id="1x4tiles"),
    pytest.param((4, 1), id="4x1tiles"),
    pytest.param((2, 4), id="2x4tiles"),
    pytest.param((4, 2), id="4x2tiles"),
    pytest.param((4, 4), id="4x4tiles"),
]

DTYPE_PARAMS = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float32, id="f32"),
]

ULP_THRESHOLD = 10


# =============================================================================
# Kernel templates
# =============================================================================


def _slice_syntax(rows, cols):
    if rows == 1 and cols == 1:
        return "0, 0"
    return f"0:{rows}, 0:{cols}"


def _tensor_shape(tile_rows, tile_cols):
    return (tile_rows * TILE_SIZE, tile_cols * TILE_SIZE)


# B1: Init + loop — out = l + K*r
ACC_INIT_LOOP_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(l, r, out):
    l_dfb = ttl.make_dataflow_buffer_like(l, shape=({R}, {C}), buffer_factor=2)
    r_dfb = ttl.make_dataflow_buffer_like(r, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with l_dfb.wait() as lv, r_dfb.wait() as rv:
            with out_dfb.reserve() as o:
                o.store(lv)
                for i in range({K}):
                    o.store(rv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with l_dfb.reserve() as blk:
            tx = ttl.copy(l[{S}], blk)
            tx.wait()
        with r_dfb.reserve() as blk:
            tx = ttl.copy(r[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# B2: Loop only — out = K*r
ACC_LOOP_ONLY_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(r, out):
    r_dfb = ttl.make_dataflow_buffer_like(r, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with r_dfb.wait() as rv:
            with out_dfb.reserve() as o:
                for i in range({K}):
                    o.store(rv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with r_dfb.reserve() as blk:
            tx = ttl.copy(r[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# B3: Init + loop + post — out = l + K*r + s
ACC_INIT_LOOP_POST_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(l, r, s, out):
    l_dfb = ttl.make_dataflow_buffer_like(l, shape=({R}, {C}), buffer_factor=2)
    r_dfb = ttl.make_dataflow_buffer_like(r, shape=({R}, {C}), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with l_dfb.wait() as lv, r_dfb.wait() as rv, s_dfb.wait() as sv:
            with out_dfb.reserve() as o:
                o.store(lv)
                for i in range({K}):
                    o.store(rv, acc=True)
                o.store(sv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with l_dfb.reserve() as blk:
            tx = ttl.copy(l[{S}], blk)
            tx.wait()
        with r_dfb.reserve() as blk:
            tx = ttl.copy(r[{S}], blk)
            tx.wait()
        with s_dfb.reserve() as blk:
            tx = ttl.copy(s[{S}], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[{S}])
            tx.wait()
"""

# C1: Multi-store per loop iteration — out = K*(a+b)
ACC_MULTI_STORE_LOOP_TEMPLATE = """\
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
                for i in range({K}):
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

# C2: Init + expression loop — out = l + K*(a+b)
ACC_INIT_EXPR_LOOP_TEMPLATE = """\
import ttl

@ttl.kernel(grid=(1, 1))
def kernel(l, a, b, out):
    l_dfb = ttl.make_dataflow_buffer_like(l, shape=({R}, {C}), buffer_factor=2)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=({R}, {C}), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=({R}, {C}), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({R}, {C}), buffer_factor=2)

    @ttl.compute()
    def compute():
        with l_dfb.wait() as lv, a_dfb.wait() as av, b_dfb.wait() as bv:
            with out_dfb.reserve() as o:
                o.store(lv)
                for i in range({K}):
                    o.store(av + bv, acc=True)

    @ttl.datamovement()
    def dm_read():
        with l_dfb.reserve() as blk:
            tx = ttl.copy(l[{S}], blk)
            tx.wait()
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


# =============================================================================
# Kernel factory
# =============================================================================

_next_kernel_id = 0


def _make_kernel(template, R, C, K=4):
    global _next_kernel_id
    _next_kernel_id += 1

    code = template.format(R=R, C=C, S=_slice_syntax(R, C), K=K)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="kernel_acc_loop_",
    ) as f:
        f.write(code)
        path = f.name

    temp_kernel_files.append(path)
    mod_name = f"acc_loop_kernel_{R}x{C}_{_next_kernel_id}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel


# =============================================================================
# Test infrastructure
# =============================================================================


def make_inputs(n, shape, dtype, seed=42):
    torch.manual_seed(seed)
    return [torch.randn(shape, dtype=dtype) for _ in range(n)]


def check_result(result, expected, dtype):
    # Multi-compute accumulation happens at f32 precision in DST registers,
    # so even bf16 results may differ from torch bf16 golden. Use allclose
    # with tolerances appropriate for bf16 truncation.
    assert_allclose(result, expected, atol=0.05, rtol=0.05)


class AccLoopTestBase:
    """Base for loop accumulation tests.

    Subclasses define:
        template:    kernel code template
        num_inputs:  number of input tensors
        loop_count:  K value for the loop
        golden(*inputs) -> expected torch output
    """

    template = None
    num_inputs = 2
    loop_count = 4

    def golden(self, *inputs):
        raise NotImplementedError

    @pytest.fixture(params=TILE_SHAPE_PARAMS)
    def tile_shape(self, request):
        return request.param

    @pytest.fixture(params=DTYPE_PARAMS)
    def dtype(self, request):
        return request.param

    def test_execute(self, device, tile_shape, dtype):
        assert self.template is not None
        R, C = tile_shape
        kernel = _make_kernel(self.template, R, C, K=self.loop_count)
        shape = _tensor_shape(R, C)

        inputs = make_inputs(self.num_inputs, shape, dtype)
        expected = self.golden(*inputs)

        dev_inputs = [to_dram(t, device) for t in inputs]
        dev_out = to_dram(torch.zeros(shape, dtype=dtype), device)

        kernel(*dev_inputs, dev_out)

        result = ttnn.to_torch(dev_out)
        check_result(result, expected, dtype)


# =============================================================================
# Group 1: Init + loop (B1)
# =============================================================================


class TestAccInitLoop(AccLoopTestBase):
    """out = l + K*r — init store followed by accumulation loop."""

    template = ACC_INIT_LOOP_TEMPLATE
    loop_count = 4

    def golden(self, l, r):
        return l + self.loop_count * r


# =============================================================================
# Group 2: Loop only (B2)
# =============================================================================


class TestAccLoopOnly(AccLoopTestBase):
    """out = K*r — accumulation loop with no init store."""

    template = ACC_LOOP_ONLY_TEMPLATE
    num_inputs = 1
    loop_count = 4

    def golden(self, r):
        return self.loop_count * r


# =============================================================================
# Group 3: Init + loop + post (B3)
# =============================================================================


class TestAccInitLoopPost(AccLoopTestBase):
    """out = l + K*r + s — init, loop, and post-loop stores."""

    template = ACC_INIT_LOOP_POST_TEMPLATE
    num_inputs = 3
    loop_count = 4

    def golden(self, l, r, s):
        return l + self.loop_count * r + s


# =============================================================================
# Group 4: Multi-store per iteration (C1)
# =============================================================================


class TestAccMultiStoreLoop(AccLoopTestBase):
    """out = K*(a+b) — two acc stores per loop iteration."""

    template = ACC_MULTI_STORE_LOOP_TEMPLATE
    loop_count = 3

    def golden(self, a, b):
        return self.loop_count * (a + b)


# =============================================================================
# Group 5: Init + expression loop (C2)
# =============================================================================


class TestAccInitExprLoop(AccLoopTestBase):
    """out = l + K*(a+b) — init store + loop with expression acc store."""

    template = ACC_INIT_EXPR_LOOP_TEMPLATE
    num_inputs = 3
    loop_count = 4

    def golden(self, l, a, b):
        return l + self.loop_count * (a + b)
