# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end coverage for scalar multiplication lowering.

Tests scalar multiplication with reduce results, non-reduce tile blocks, and
fused elementwise expressions. Matmul post-op coverage lives in
test_matmul_fused_postops.py.

Coverage:
- Both reduce kinds × both dtypes × all dim combinations × scaler values
  including 1.0 / negative / >1.
- Multi-tile inputs (the original #594 bug case for dims=[0,1]).
- `reduce_max * c` with c<0 must apply the scaler after the max.
- Two reduces with distinct scalar coefficients in one compute block.
- N reduces sharing one scaler (FillOp dedup, DFB budget).
- `fp32_dest_acc_en` with a non-1.0 scaler.
- Scaler from `torch.tensor(c, bf16).item()` — bf16-rounded host value.
- `reduce(a) * reduce(b)` — tile*tile mul over reduce results.
- `reduce(x, dims=[-1])` — negative dim index normalization.
- `reduce(x, dims=[0,1]) * c` — RHS scalar form (TensorBlock.__mul__
  direct path, not via AST commute).
- Integer scalar (`2 * reduce_sum(...)`) — int→float coercion.
- Scalar multiply over non-reduce expressions: `c * x`, `x * c`,
  `c * (x + y)`, `c * x + y`, `c0 * c1 * (x + y)`, and
  `c * (x * y + z)`.
- Zero scaler — f32 bit pattern of 0.0.
- Scaler with extreme magnitudes (1e-3, 1e3).
"""

import atexit
import importlib
import os
import tempfile
from typing import Callable, List, Tuple

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl

TILE = 32


# =============================================================================
# Kernel template
# =============================================================================

SCALAR_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def scalar_mul_kernel(inp, out):
    """{reduce_fn} on ({inp_rows},{inp_cols}) grid, dims={dims}, multiplied as `{mul_expr}`."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_rows}, {out_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            out_blk.store({mul_expr})

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[{inp_slice}], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[{out_slice}]).wait()
'''

SCALAR_NON_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def scalar_mul_kernel(inp_a, inp_b, inp_c, out):
    """Scalar multiply over a non-reduce expression."""
    a_dfb = ttl.make_dataflow_buffer_like(
        inp_a, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        inp_b, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(
        inp_c, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({tile_rows}, {tile_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            c_dfb.wait() as c_blk,
            out_dfb.reserve() as out_blk,
        ):
            out_blk.store({expr})

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as a_blk:
            ttl.copy(inp_a[{tensor_slice}], a_blk).wait()
        with b_dfb.reserve() as b_blk:
            ttl.copy(inp_b[{tensor_slice}], b_blk).wait()
        with c_dfb.reserve() as c_blk:
            ttl.copy(inp_c[{tensor_slice}], c_blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[{tensor_slice}]).wait()
'''

TWO_REDUCE_DISTINCT_SCALER_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def scalar_mul_kernel(inp_a, inp_b, out_a, out_b):
    """Two reduces in one compute block carrying distinct scalar coefficients."""
    a_in_dfb = ttl.make_dataflow_buffer_like(inp_a, shape=(1, 1), block_count=2)
    b_in_dfb = ttl.make_dataflow_buffer_like(inp_b, shape=(1, 1), block_count=2)
    a_out_dfb = ttl.make_dataflow_buffer_like(out_a, shape=(1, 1), block_count=2)
    b_out_dfb = ttl.make_dataflow_buffer_like(out_b, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_in_dfb.wait() as ai,
            b_in_dfb.wait() as bi,
            a_out_dfb.reserve() as ao,
            b_out_dfb.reserve() as bo,
        ):
            ao.store({scaler_a} * ttl.math.{reduce_fn}(ai, dims=[0, 1]))
            bo.store({scaler_b} * ttl.math.{reduce_fn}(bi, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with a_in_dfb.reserve() as ai:
            ttl.copy(inp_a[0, 0], ai).wait()
        with b_in_dfb.reserve() as bi:
            ttl.copy(inp_b[0, 0], bi).wait()

    @ttl.datamovement()
    def dm_write():
        with a_out_dfb.wait() as ao:
            ttl.copy(ao, out_a[0, 0]).wait()
        with b_out_dfb.wait() as bo:
            ttl.copy(bo, out_b[0, 0]).wait()
'''

REPEATED_SCALAR_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def scalar_mul_kernel(inp, out):
    """N reduce sites in one kernel sharing the same Python scalar."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        for _ in range({n}):
            with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
                out_blk.store(
                    {scaler} * ttl.math.{reduce_fn}(inp_blk, dims=[0, 1])
                )

    @ttl.datamovement()
    def dm_read():
        for tile_index in range({n}):
            with inp_dfb.reserve() as blk:
                ttl.copy(inp[tile_index, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        for tile_index in range({n}):
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[tile_index, 0]).wait()
'''

FP32_DEST_ACC_SCALAR_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
def scalar_mul_kernel(inp, out):
    """Single-tile scalar multiply over a reduce with fp32 destination accumulation."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            out_blk.store({scaler} * ttl.math.{reduce_fn}(inp_blk, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()
'''

REDUCE_TIMES_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def scalar_mul_kernel(inp_a, inp_b, out):
    """Two independent reduces whose 1x1 results are multiplied together."""
    a_dfb = ttl.make_dataflow_buffer_like(inp_a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(inp_b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            out_dfb.reserve() as out_blk,
        ):
            a_reduced = ttl.math.reduce_sum(a_blk, dims=[0, 1])
            b_reduced = ttl.math.{rhs_reduce_fn}(b_blk, dims=[0, 1])
            out_blk.store(a_reduced * b_reduced)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(inp_a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(inp_b[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()
'''

_kernel_cache = {}
_temp_files = []


def _build_kernel(code: str, prefix: str, cache_key: tuple) -> Callable:
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=prefix
    ) as tmp:
        tmp.write(code)
        path = tmp.name
    _temp_files.append(path)
    spec = importlib.util.spec_from_file_location("scalar_mul_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _kernel_cache[cache_key] = mod.scalar_mul_kernel
    return mod.scalar_mul_kernel


def _slice(rows: int, cols: int) -> str:
    return "0, 0" if rows == 1 and cols == 1 else f"0:{rows}, 0:{cols}"


def _out_shape(rows: int, cols: int, dims: List[int]) -> Tuple[int, int]:
    norm = {d % 2 for d in dims}
    return (1 if 0 in norm else rows, 1 if 1 in norm else cols)


def make_scalar_reduce_kernel(
    reduce_fn: str,
    inp_rows: int,
    inp_cols: int,
    dims: List[int],
    scaler_expr: str,
    *,
    scaler_on_rhs: bool = False,
) -> Callable:
    out_rows, out_cols = _out_shape(inp_rows, inp_cols, dims)
    cache_key = (reduce_fn, inp_rows, inp_cols, tuple(dims), scaler_expr, scaler_on_rhs)
    reduce_call = f"ttl.math.{reduce_fn}(inp_blk, dims={dims})"
    mul_expr = (
        f"{reduce_call} * {scaler_expr}"
        if scaler_on_rhs
        else f"{scaler_expr} * {reduce_call}"
    )
    code = SCALAR_REDUCE_KERNEL_TEMPLATE.format(
        reduce_fn=reduce_fn,
        inp_rows=inp_rows,
        inp_cols=inp_cols,
        out_rows=out_rows,
        out_cols=out_cols,
        dims=dims,
        mul_expr=mul_expr,
        inp_slice=_slice(inp_rows, inp_cols),
        out_slice=_slice(out_rows, out_cols),
    )
    return _build_kernel(
        code,
        prefix=f"scalar_reduce_{reduce_fn}_{inp_rows}x{inp_cols}_",
        cache_key=cache_key,
    )


def make_two_reduce_kernel(reduce_fn: str, scaler_a: str, scaler_b: str) -> Callable:
    cache_key = ("two_reduce", reduce_fn, scaler_a, scaler_b)
    code = TWO_REDUCE_DISTINCT_SCALER_TEMPLATE.format(
        reduce_fn=reduce_fn, scaler_a=scaler_a, scaler_b=scaler_b
    )
    return _build_kernel(code, prefix=f"two_reduce_{reduce_fn}_", cache_key=cache_key)


def make_scalar_non_reduce_kernel(
    expr: str, tile_rows: int, tile_cols: int
) -> Callable:
    """Scalar multiply over a non-reduce expression."""
    cache_key = ("scalar_non_reduce", expr, tile_rows, tile_cols)
    code = SCALAR_NON_REDUCE_KERNEL_TEMPLATE.format(
        expr=expr,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tensor_slice=_slice(tile_rows, tile_cols),
    )
    return _build_kernel(
        code, prefix=f"scalar_non_reduce_{tile_rows}x{tile_cols}_", cache_key=cache_key
    )


def make_shared_scaler_n_reduce_kernel(reduce_fn: str, n: int, scaler: float):
    """N reduce sites in one compute block all multiplied by the same scaler."""
    cache_key = ("shared_scaler_n_reduce", reduce_fn, n, scaler)
    code = REPEATED_SCALAR_REDUCE_KERNEL_TEMPLATE.format(
        reduce_fn=reduce_fn,
        n=n,
        scaler=repr(float(scaler)),
    )
    return _build_kernel(
        code, prefix=f"shared_scaler_{reduce_fn}_", cache_key=cache_key
    )


def make_fp32_dest_acc_scalar_reduce_kernel(reduce_fn: str, scaler: float):
    """Single-tile `c * reduce(...)` with `fp32_dest_acc_en=True`."""
    cache_key = ("fp32_dest_acc_scalar_reduce", reduce_fn, scaler)
    code = FP32_DEST_ACC_SCALAR_REDUCE_KERNEL_TEMPLATE.format(
        reduce_fn=reduce_fn, scaler=repr(float(scaler))
    )
    return _build_kernel(
        code, prefix=f"fp32_dest_acc_{reduce_fn}_", cache_key=cache_key
    )


def make_bf16_rounded_scaler_kernel(scaler_value: float):
    """`scaler_value * reduce_sum(x, dims=[0, 1])` where scaler_value is a
    Python float already rounded to the bf16 grid (caller does `.item()`)."""
    return make_scalar_reduce_kernel(
        "reduce_sum", 1, 1, [0, 1], repr(float(scaler_value))
    )


def make_reduce_times_reduce_kernel(reduce_fn: str):
    """`reduce(a, dims=[0,1]) * reduce(b, dims=[0,1])` — tile * tile mul on
    matching 1x1 results, no broadcast required."""
    cache_key = ("reduce_times_reduce", reduce_fn)
    code = REDUCE_TIMES_REDUCE_KERNEL_TEMPLATE.format(rhs_reduce_fn=reduce_fn)
    return _build_kernel(code, prefix=f"reduce_times_{reduce_fn}_", cache_key=cache_key)


@atexit.register
def _cleanup():
    for p in _temp_files:
        try:
            os.unlink(p)
        except OSError:
            pass


# =============================================================================
# Helpers
# =============================================================================

DTYPES = [torch.bfloat16, torch.float32]
DTYPE_IDS = ["bf16", "fp32"]


def _tolerances(dtype, scaler=1.0):
    """Per-dtype tolerances for reduce(...) * scaler comparisons.

    `atol` is scaled by `max(|scaler|, 1.0)` because any per-element
    accumulation noise in the reduce is multiplied by the scaler before
    being compared to the expected value.
    """
    scale = max(abs(float(scaler)), 1.0)
    if dtype == torch.float32:
        return dict(rtol=5e-3, atol=1e-2 * scale)
    return dict(rtol=0.05, atol=1.0 * scale)


def _expected(inp_torch, reduce_fn, dims, scaler):
    norm_dims = sorted({d % 2 for d in dims})
    val = inp_torch.float()
    if reduce_fn == "reduce_sum":
        for d in norm_dims:
            val = val.sum(dim=d, keepdim=True)
    else:
        for d in norm_dims:
            val = val.amax(dim=d, keepdim=True)
    return val * float(scaler)


def _populated(result, dims):
    norm = {d % 2 for d in dims}
    if norm == {0, 1}:
        return result[:1, :1].float().contiguous()
    if 0 in norm:
        return result[:1, :].float().contiguous()
    return result[:, :1].float().contiguous()


def _expected_non_reduce(expr_name, input_a, input_b, input_c, scaler):
    if expr_name in ("direct_lhs", "direct_rhs"):
        return input_a.float() * float(scaler)
    if expr_name in ("add_lhs", "chained_scalar_lhs"):
        return (input_a.float() + input_b.float()) * float(scaler)
    if expr_name == "mul_add_lhs":
        return input_a.float() * float(scaler) + input_b.float()
    if expr_name == "chain_lhs":
        return (input_a.float() * input_b.float() + input_c.float()) * float(scaler)
    raise AssertionError(f"unknown non-reduce expression case: {expr_name}")


# =============================================================================
# Tests
# =============================================================================

SCALERS = [
    pytest.param(1.0, id="scaler_1"),
    pytest.param(0.5, id="scaler_half"),
    pytest.param(-0.25, id="scaler_neg_quarter"),
    pytest.param(2.5, id="scaler_gt1"),
]

DIMS = [
    pytest.param([0], id="dim0"),
    pytest.param([1], id="dim1"),
    pytest.param([0, 1], id="dim01"),
]

REDUCE_FNS = ["reduce_sum", "reduce_max"]

NON_REDUCE_EXPR_CASES = [
    pytest.param(
        "direct_lhs",
        (2, 2),
        0.5,
        "0.5 * a_blk",
        id="direct_lhs_2x2_half",
    ),
    pytest.param(
        "direct_rhs",
        (1, 4),
        0.0,
        "a_blk * 0.0",
        id="direct_rhs_1x4_zero",
    ),
    pytest.param(
        "add_lhs",
        (2, 2),
        -1.5,
        "-1.5 * (a_blk + b_blk)",
        id="add_lhs_2x2_negative",
    ),
    pytest.param(
        "chained_scalar_lhs",
        (2, 2),
        10.0,
        "2 * 5 * (a_blk + b_blk)",
        id="chained_scalar_lhs_2x2",
    ),
    pytest.param(
        "mul_add_lhs",
        (2, 2),
        0.5,
        "0.5 * a_blk + b_blk",
        id="mul_add_lhs_2x2_half",
    ),
    pytest.param(
        "chain_lhs",
        (1, 4),
        1e-3,
        "0.001 * (a_blk * b_blk + c_blk)",
        id="chain_lhs_1x4_tiny",
    ),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("scaler", SCALERS)
def test_scalar_times_reduce_single_tile(device, dtype, reduce_fn, dims, scaler):
    """`c * reduce_fn(x, dims=...)` on a single-tile input."""
    kernel = make_scalar_reduce_kernel(reduce_fn, 1, 1, list(dims), repr(float(scaler)))

    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_rows, out_cols = _out_shape(1, 1, list(dims))
    out_torch = torch.zeros(out_rows * TILE, out_cols * TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    result = ttnn.to_torch(out)
    expected = _expected(inp_torch, reduce_fn, list(dims), scaler)
    actual = _populated(result, list(dims))
    assert_allclose(actual, expected, **_tolerances(dtype, scaler))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
@pytest.mark.parametrize("dims", [[0], [1], [0, 1]], ids=["dim0", "dim1", "dim01"])
@pytest.mark.parametrize(
    "scaler", [0.25, -1.5], ids=["scaler_quarter", "scaler_neg_1_5"]
)
def test_scalar_times_reduce_multi_tile(device, dtype, reduce_fn, dims, scaler):
    """`c * reduce_fn(x, dims=...)` on a 2x2 tile grid input."""
    kernel = make_scalar_reduce_kernel(reduce_fn, 2, 2, dims, repr(float(scaler)))

    inp_torch = (torch.rand(2 * TILE, 2 * TILE, dtype=dtype) - 0.5) * 4.0
    out_rows, out_cols = _out_shape(2, 2, dims)
    out_torch = torch.zeros(out_rows * TILE, out_cols * TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    result = ttnn.to_torch(out)
    expected = _expected(inp_torch, reduce_fn, dims, scaler)
    actual = _populated(result, dims)
    assert_allclose(actual, expected, **_tolerances(dtype, scaler))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
def test_two_distinct_scalers_in_one_kernel(device, dtype, reduce_fn):
    """Two reduce sites in one compute block, distinct scalar coefficients
    each. Catches any shared-FillOp leak that would apply one scaler to
    both sites."""
    scaler_a, scaler_b = 0.25, -1.5
    kernel = make_two_reduce_kernel(
        reduce_fn, repr(float(scaler_a)), repr(float(scaler_b))
    )

    inp_a_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    inp_b_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_a_torch = torch.zeros(TILE, TILE, dtype=dtype)
    out_b_torch = torch.zeros(TILE, TILE, dtype=dtype)

    inp_a = to_l1(inp_a_torch, device)
    inp_b = to_l1(inp_b_torch, device)
    out_a = to_l1(out_a_torch, device)
    out_b = to_l1(out_b_torch, device)
    kernel(inp_a, inp_b, out_a, out_b)

    expected_a = _expected(inp_a_torch, reduce_fn, [0, 1], scaler_a).reshape(())
    expected_b = _expected(inp_b_torch, reduce_fn, [0, 1], scaler_b).reshape(())
    actual_a = ttnn.to_torch(out_a)[0, 0].float()
    actual_b = ttnn.to_torch(out_b)[0, 0].float()
    assert_allclose(actual_a, expected_a, **_tolerances(dtype, scaler_a))
    assert_allclose(actual_b, expected_b, **_tolerances(dtype, scaler_b))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_reduce_max_negative_scaler_sign(device, dtype):
    """`c * reduce_max(x, dims=[0,1])` with c<0 must equal `max(x) * c`,
    not `max(x * c)`. The latter inverts ordering and picks the wrong
    extremum."""
    kernel = make_scalar_reduce_kernel("reduce_max", 1, 1, [0, 1], repr(-2.0))

    inp_torch = torch.full((TILE, TILE), 1.0, dtype=dtype)
    inp_torch[0, 0] = 5.0
    inp_torch[1, 1] = -3.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    actual = ttnn.to_torch(out)[0, 0].float().item()
    # max(x) * -2 = -10. The wrong ordering (max(x * -2)) would give +6.
    assert actual == pytest.approx(-10.0, rel=0.05, abs=1.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("scaler", [1e-3, 1e3], ids=["tiny", "large"])
def test_scalar_extreme_magnitudes(device, dtype, scaler):
    """Scaler magnitudes outside [0.1, 10] exercise the f32 bit-pattern
    encoding for `mul_unary_tile`."""
    kernel = make_scalar_reduce_kernel("reduce_sum", 1, 1, [0, 1], repr(float(scaler)))

    inp_torch = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = float(TILE * TILE) * float(scaler)
    actual = ttnn.to_torch(out)[0, 0].float().item()
    assert actual == pytest.approx(expected, rel=0.05, abs=abs(expected) * 0.05 + 1.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
def test_n_reduces_sharing_one_scaler(device, dtype, reduce_fn):
    """N reduce sites in one kernel sharing the same Python scalar."""
    n = 4
    scaler = 0.5
    kernel = make_shared_scaler_n_reduce_kernel(reduce_fn, n, scaler)

    inp_torch = (torch.rand(n * TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(n * TILE, TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    result = ttnn.to_torch(out)
    tol = _tolerances(dtype, scaler)
    for i in range(n):
        tile_in = inp_torch[i * TILE : (i + 1) * TILE, :TILE]
        expected = _expected(tile_in, reduce_fn, [0, 1], scaler).reshape(())
        actual = result[i * TILE, 0].float()
        assert_allclose(actual, expected, **tol)


@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
def test_scalar_times_reduce_fp32_dest_acc(device, reduce_fn):
    """fp32 destination accumulation + non-1.0 scaler at dims=[0,1]."""
    scaler = 0.5
    kernel = make_fp32_dest_acc_scalar_reduce_kernel(reduce_fn, scaler)

    inp_torch = (torch.rand(TILE, TILE, dtype=torch.float32) - 0.5) * 4.0
    out_torch = torch.zeros(TILE, TILE, dtype=torch.float32)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = _expected(inp_torch, reduce_fn, [0, 1], scaler).reshape(())
    actual = ttnn.to_torch(out)[0, 0].float()
    assert_allclose(actual, expected, **_tolerances(torch.float32, scaler))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_bf16_rounded_scalar(device, dtype):
    """Scalar computed from a torch 0-dim bf16 tensor via `.item()`. The
    extracted Python float is already snapped to the bf16 grid, so the
    result matches the bf16-rounded value rather than the raw input."""
    raw_value = 0.31  # not exactly representable in bf16
    scaler_value = torch.tensor(raw_value, dtype=torch.bfloat16).item()
    kernel = make_bf16_rounded_scaler_kernel(scaler_value)

    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = _expected(inp_torch, "reduce_sum", [0, 1], scaler_value).reshape(())
    actual = ttnn.to_torch(out)[0, 0].float()
    assert_allclose(actual, expected, **_tolerances(dtype, scaler_value))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
def test_reduce_times_reduce(device, dtype, reduce_fn):
    """Two independent reduces whose 1x1 results are multiplied together
    via `ttl.mul`. No broadcast needed since both results are 1x1."""
    kernel = make_reduce_times_reduce_kernel(reduce_fn)

    inp_a_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    inp_b_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp_a = to_l1(inp_a_torch, device)
    inp_b = to_l1(inp_b_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp_a, inp_b, out)

    a_reduced = inp_a_torch.float().sum().item()
    if reduce_fn == "reduce_sum":
        b_reduced = inp_b_torch.float().sum().item()
    else:
        b_reduced = inp_b_torch.float().amax().item()
    expected = torch.tensor(a_reduced * b_reduced, dtype=torch.float32)
    actual = ttnn.to_torch(out)[0, 0].float()
    # Each reduce can be ~|sum of 1024 ~U(-2,2)| ≈ 30; product amplifies error.
    assert_allclose(
        actual, expected, **_tolerances(dtype, max(abs(a_reduced), abs(b_reduced)))
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_negative_dim_index(device, dtype):
    """`dims=[-1]` must lower the same as `dims=[1]` (negative indexing
    normalized to `% rank`). One case suffices since the negative index
    only affects the front-end op's `dims` attribute."""
    kernel = make_scalar_reduce_kernel("reduce_sum", 1, 1, [-1], repr(0.5))

    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = _expected(inp_torch, "reduce_sum", [-1], 0.5)
    actual = _populated(ttnn.to_torch(out), [-1])
    assert_allclose(actual, expected, **_tolerances(dtype, 0.5))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
def test_reduce_times_scalar_rhs_form(device, dtype, reduce_fn):
    """`reduce_fn(x, dims=[0,1]) * c` — scalar on the RHS. Goes through
    `TensorBlock.__mul__` directly (no commute), distinct from the
    `c * reduce(...)` cases that exercise the AST-level `__rmul__` swap."""
    scaler = 0.5
    kernel = make_scalar_reduce_kernel(
        reduce_fn, 1, 1, [0, 1], repr(scaler), scaler_on_rhs=True
    )

    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = _expected(inp_torch, reduce_fn, [0, 1], scaler).reshape(())
    actual = ttnn.to_torch(out)[0, 0].float()
    assert_allclose(actual, expected, **_tolerances(dtype, scaler))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_integer_scalar(device, dtype):
    """Python int scalar (no `.0`). The DSL must coerce to float before
    emitting `mul_unary_const`."""
    kernel = make_scalar_reduce_kernel("reduce_sum", 1, 1, [0, 1], "2")

    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = _expected(inp_torch, "reduce_sum", [0, 1], 2).reshape(())
    actual = ttnn.to_torch(out)[0, 0].float()
    assert_allclose(actual, expected, **_tolerances(dtype, 2))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_zero_scaler(device, dtype):
    """`0.0 * reduce_sum(...)` must produce zero output. The f32 bit
    pattern of 0.0 is 0x00000000 — distinct code path in some encoders
    from non-zero floats."""
    kernel = make_scalar_reduce_kernel("reduce_sum", 1, 1, [0, 1], repr(0.0))

    inp_torch = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out_torch = torch.full(
        (TILE, TILE), 7.0, dtype=dtype
    )  # nonzero so a no-op write fails
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    actual = ttnn.to_torch(out)[0, 0].float().item()
    assert actual == 0.0


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("expr_name,tile_shape,scaler,expr", NON_REDUCE_EXPR_CASES)
def test_scalar_times_non_reduce_expression(
    device, dtype, expr_name, tile_shape, scaler, expr
):
    """Scalar multiply over non-reduce expressions and multi-tile blocks."""
    if expr_name == "mul_add_lhs" and dtype == torch.float32:
        pytest.xfail(
            "fp32 scalar unary feeding SFPU add corrupts output. Tracked in #612."
        )

    tile_rows, tile_cols = tile_shape
    kernel = make_scalar_non_reduce_kernel(expr, tile_rows, tile_cols)

    tensor_shape = (tile_rows * TILE, tile_cols * TILE)
    input_a_torch = (torch.rand(tensor_shape, dtype=dtype) - 0.5) * 4.0
    input_b_torch = (torch.rand(tensor_shape, dtype=dtype) - 0.5) * 4.0
    input_c_torch = (torch.rand(tensor_shape, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(tensor_shape, dtype=dtype)
    input_a = to_l1(input_a_torch, device)
    input_b = to_l1(input_b_torch, device)
    input_c = to_l1(input_c_torch, device)
    out = to_l1(out_torch, device)
    kernel(input_a, input_b, input_c, out)

    expected = _expected_non_reduce(
        expr_name, input_a_torch, input_b_torch, input_c_torch, scaler
    )
    actual = ttnn.to_torch(out).float()
    assert_allclose(actual, expected, **_tolerances(dtype, scaler))
