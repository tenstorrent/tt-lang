# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for the ttl.math.exp hardware flags.

Flags must be passed as literal keyword arguments in the ``ttl.math.exp`` call
(the TTL AST compiler cannot resolve ``**kwargs`` splats), so each variant has
its own operation.
"""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram

EXP_SCALE = 2.0
TILE = 32
ACCURATE_TOLERANCES = {
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
    torch.float32: {"rtol": 1e-3, "atol": 1e-3},
}
APPROX_TOLERANCES = {
    torch.bfloat16: {"rtol": 5e-2, "atol": 5e-2},
    torch.float32: {"rtol": 4e-2, "atol": 5e-3},
}
MEMORY_CONFIGS = [
    pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram"),
    pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1"),
]
EXP_CONFIGS = [
    pytest.param(torch.bfloat16, ttnn.DRAM_MEMORY_CONFIG, id="bf16-dram"),
    pytest.param(torch.bfloat16, ttnn.L1_MEMORY_CONFIG, id="bf16-l1"),
    pytest.param(torch.float32, ttnn.DRAM_MEMORY_CONFIG, id="f32-dram"),
    pytest.param(torch.float32, ttnn.L1_MEMORY_CONFIG, id="f32-l1"),
]


@ttl.operation(grid=(1, 1))
def exp_default_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_approx_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x, approx=True))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_approx_scale_keyword_kernel(in_t, out_t):
    tile_rows = in_t.shape[0] // TILE
    tile_cols = in_t.shape[1] // TILE
    a_cb = ttl.make_dataflow_buffer_like(
        in_t, shape=(tile_rows, tile_cols), block_count=2
    )
    out_cb = ttl.make_dataflow_buffer_like(
        out_t, shape=(tile_rows, tile_cols), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as x, out_cb.reserve() as r:
            r.store(ttl.math.exp(x, approx=True, scale=EXP_SCALE))

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as block:
            ttl.copy(in_t[0:tile_rows, 0:tile_cols], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as block:
            ttl.copy(block, out_t[0:tile_rows, 0:tile_cols]).wait()


@ttl.operation(grid=(1, 1))
def exp_approx_scale_mul_kernel(in_t, out_t):
    tile_rows = in_t.shape[0] // TILE
    tile_cols = in_t.shape[1] // TILE
    a_cb = ttl.make_dataflow_buffer_like(
        in_t, shape=(tile_rows, tile_cols), block_count=2
    )
    out_cb = ttl.make_dataflow_buffer_like(
        out_t, shape=(tile_rows, tile_cols), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as x, out_cb.reserve() as r:
            r.store(ttl.math.exp(x * EXP_SCALE, approx=True))

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as block:
            ttl.copy(in_t[0:tile_rows, 0:tile_cols], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as block:
            ttl.copy(block, out_t[0:tile_rows, 0:tile_cols]).wait()


@ttl.operation(grid=(1, 1))
def exp_approx_skip_clamp_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x, approx=True, skip_clamp_check=True))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_scale_literal_mul_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x * 2.0))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_scale_variable_mul_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x * EXP_SCALE))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


@ttl.operation(grid=(1, 1))
def exp_scale_keyword_kernel(in_t, out_t):
    a_cb = ttl.make_dataflow_buffer_like(in_t, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out_t, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = a_cb.wait()
        r = out_cb.reserve()
        r.store(ttl.math.exp(x, scale=EXP_SCALE))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(in_t[0, 0], a_cb.reserve())

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_cb.wait(), out_t[0, 0])


def _run(kernel, inp_t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    in_t = to_dram(inp_t, device)
    out_t = to_dram(torch.zeros_like(inp_t), device)
    if memory_config == ttnn.L1_MEMORY_CONFIG:
        in_t = ttnn.to_memory_config(in_t, memory_config=memory_config)
        out_t = ttnn.to_memory_config(out_t, memory_config=memory_config)
    kernel(in_t, out_t)
    return ttnn.to_torch(out_t).reshape(inp_t.shape).to(inp_t.dtype)


def _rand_input(shape, dtype):
    return (torch.randn(shape, dtype=dtype) * 0.5).clamp(-1.0, 1.0)


@pytest.mark.parametrize("dtype,memory_config", EXP_CONFIGS)
def test_exp_default(device, dtype, memory_config):
    inp_t = _rand_input((TILE, TILE), dtype)
    expected = torch.exp(inp_t.float()).to(dtype)
    got = _run(exp_default_kernel, inp_t, device, memory_config)
    assert_allclose(got.float(), expected.float(), **ACCURATE_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype,memory_config", EXP_CONFIGS)
def test_exp_approx(device, dtype, memory_config):
    inp_t = _rand_input((TILE, TILE), dtype)
    expected = torch.exp(inp_t.float()).to(dtype)
    got = _run(exp_approx_kernel, inp_t, device, memory_config)
    assert_allclose(got.float(), expected.float(), **APPROX_TOLERANCES[dtype])


def test_exp_approx_skip_clamp(device):
    inp_t = torch.full((TILE, TILE), -100.0, dtype=torch.bfloat16)
    clamped = _run(exp_approx_kernel, inp_t, device)
    unclamped = _run(exp_approx_skip_clamp_kernel, inp_t, device)
    assert torch.isfinite(clamped).all()
    assert (clamped >= 0).all()
    assert (unclamped < 0).any()


@pytest.mark.parametrize("dtype,memory_config", EXP_CONFIGS)
def test_exp_scaled_by_literal_mul(device, dtype, memory_config):
    inp_t = _rand_input((TILE, TILE), dtype)
    expected = torch.exp(2.0 * inp_t.float()).to(dtype)
    got = _run(exp_scale_literal_mul_kernel, inp_t, device, memory_config)
    assert_allclose(got.float(), expected.float(), **ACCURATE_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype,memory_config", EXP_CONFIGS)
def test_exp_scaled_by_variable_mul(device, dtype, memory_config):
    inp_t = _rand_input((TILE, TILE), dtype)
    expected = torch.exp(EXP_SCALE * inp_t.float()).to(dtype)
    got = _run(exp_scale_variable_mul_kernel, inp_t, device, memory_config)
    assert_allclose(got.float(), expected.float(), **ACCURATE_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype,memory_config", EXP_CONFIGS)
def test_exp_scaled_by_keyword(device, dtype, memory_config):
    inp_t = _rand_input((TILE, TILE), dtype)
    expected = torch.exp(EXP_SCALE * inp_t.float()).to(dtype)
    got = _run(exp_scale_keyword_kernel, inp_t, device, memory_config)
    assert_allclose(got.float(), expected.float(), **ACCURATE_TOLERANCES[dtype])


@pytest.mark.parametrize("shape", [(TILE, TILE), (2 * TILE, 2 * TILE)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
@pytest.mark.parametrize(
    "kernel",
    [exp_approx_scale_keyword_kernel, exp_approx_scale_mul_kernel],
    ids=["keyword", "multiply"],
)
def test_exp_approx_scaled(device, shape, dtype, memory_config, kernel):
    inp_t = _rand_input(shape, dtype)
    expected = torch.exp(EXP_SCALE * inp_t.float()).to(dtype)
    got = _run(kernel, inp_t, device, memory_config)
    assert_allclose(got.float(), expected.float(), **APPROX_TOLERANCES[dtype])
