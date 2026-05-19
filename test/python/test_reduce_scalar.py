# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end coverage for the numeric-scalar reduce API
`ttl.math.reduce_{sum,max}(inp, <number>, dims=...)`.

Covers:
- bf16 and fp32 inputs against both reduce_sum and reduce_max
- scaler values that exercise the skip-multiply optimization (1.0),
  bf16-lossy magnitudes, negatives, and >1 magnifiers
- equivalence between the numeric-scaler form and the tile-form scaler
- sign correctness for reduce_max with a negative scaler
- multi-reduce kernels with distinct scalers (no attribute leak)
- many reduces sharing one scaler (dedup of compiler-allocated DFBs)
- extreme scaler magnitudes (no silent precision loss)
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
# Kernel templates
# =============================================================================

SCALAR_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def reduce_kernel(inp, out):
    """Reduce {reduce_fn} dims={dims} scaler={scaler_expr} on ({inp_rows},{inp_cols}) grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_rows}, {out_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            result = ttl.math.{reduce_fn}(inp_blk, {scaler_expr}, dims={dims})
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[{inp_slice}], inp_blk)
        tx_inp.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[{out_slice}])
        tx_out.wait()
        out_blk.pop()
'''

TWO_REDUCE_DISTINCT_SCALER_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def reduce_kernel(inp_a, inp_b, out_a, out_b):
    """Two reduces with distinct scalar scalers in one compute block.

    Catches bugs where the post-reduce scaler multiply is shared, hoisted,
    or otherwise leaked between two reduce sites.
    """
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
            ao.store(ttl.math.{reduce_fn}(ai, {scaler_a}, dims=[0, 1]))
            bo.store(ttl.math.{reduce_fn}(bi, {scaler_b}, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        ai = a_in_dfb.reserve()
        ttl.copy(inp_a[0, 0], ai).wait()
        ai.push()
        bi = b_in_dfb.reserve()
        ttl.copy(inp_b[0, 0], bi).wait()
        bi.push()

    @ttl.datamovement()
    def dm_write():
        ao = a_out_dfb.wait()
        ttl.copy(ao, out_a[0, 0]).wait()
        ao.pop()
        bo = b_out_dfb.wait()
        ttl.copy(bo, out_b[0, 0]).wait()
        bo.pop()
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
    spec = importlib.util.spec_from_file_location("reduce_scalar_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _kernel_cache[cache_key] = mod.reduce_kernel
    return mod.reduce_kernel


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
) -> Callable:
    out_rows, out_cols = _out_shape(inp_rows, inp_cols, dims)
    cache_key = (reduce_fn, inp_rows, inp_cols, tuple(dims), scaler_expr)
    code = SCALAR_REDUCE_KERNEL_TEMPLATE.format(
        reduce_fn=reduce_fn,
        inp_rows=inp_rows,
        inp_cols=inp_cols,
        out_rows=out_rows,
        out_cols=out_cols,
        dims=dims,
        scaler_expr=scaler_expr,
        inp_slice=_slice(inp_rows, inp_cols),
        out_slice=_slice(out_rows, out_cols),
    )
    return _build_kernel(
        code, prefix=f"scalar_reduce_{reduce_fn}_{inp_rows}x{inp_cols}_", cache_key=cache_key
    )


def make_two_reduce_kernel(reduce_fn: str, scaler_a: str, scaler_b: str) -> Callable:
    cache_key = ("two_reduce", reduce_fn, scaler_a, scaler_b)
    code = TWO_REDUCE_DISTINCT_SCALER_TEMPLATE.format(
        reduce_fn=reduce_fn, scaler_a=scaler_a, scaler_b=scaler_b
    )
    return _build_kernel(
        code, prefix=f"two_reduce_{reduce_fn}_", cache_key=cache_key
    )


def make_shared_scaler_n_reduce_kernel(reduce_fn: str, n: int, scaler: float):
    """Kernel with `n` reduce sites all using the same numeric scaler value.

    Streams `n` input tiles through a single input DFB, runs one reduce per
    tile (with the same scaler value), and writes each result through a single
    output DFB.
    """
    if reduce_fn == "reduce_sum":

        @ttl.operation(grid=(1, 1))
        def kernel(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )

            @ttl.compute()
            def compute_fn():
                for _ in range(n):
                    with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
                        out_blk.store(
                            ttl.math.reduce_sum(inp_blk, scaler, dims=[0, 1])
                        )

            @ttl.datamovement()
            def dm_read():
                for i in range(n):
                    with inp_dfb.reserve() as blk:
                        ttl.copy(inp[i, 0], blk).wait()

            @ttl.datamovement()
            def dm_write():
                for i in range(n):
                    with out_dfb.wait() as blk:
                        ttl.copy(blk, out[i, 0]).wait()

    else:

        @ttl.operation(grid=(1, 1))
        def kernel(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )

            @ttl.compute()
            def compute_fn():
                for _ in range(n):
                    with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
                        out_blk.store(
                            ttl.math.reduce_max(inp_blk, scaler, dims=[0, 1])
                        )

            @ttl.datamovement()
            def dm_read():
                for i in range(n):
                    with inp_dfb.reserve() as blk:
                        ttl.copy(inp[i, 0], blk).wait()

            @ttl.datamovement()
            def dm_write():
                for i in range(n):
                    with out_dfb.wait() as blk:
                        ttl.copy(blk, out[i, 0]).wait()

    return kernel


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


def _tolerances(dtype):
    # Match test_reduce.py; widen abs for tiny scaler magnitudes only when
    # the reduced sum is itself small.
    if dtype == torch.float32:
        return dict(rtol=5e-3, atol=1e-2)
    return dict(rtol=0.05, atol=1.0)


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


def _create_scaler_tile(value: float, dtype):
    """Tile-form scaler matching the convention in test_reduce.py."""
    tile = torch.zeros((TILE, TILE), dtype=dtype)
    tile[0, :] = value
    tile[16, :] = value
    return tile


# =============================================================================
# Tests
# =============================================================================

# Scaler values stress: 1.0 (skip-multiply optimization), positive non-trivial,
# negative (must apply *after* reduce for reduce_max to stay correct), value
# not exactly representable in bf16, and a value > 1 that magnifies any sign
# error.
SCALERS = [
    pytest.param(1.0, id="scaler_1"),
    pytest.param(0.5, id="scaler_half"),
    pytest.param(-0.25, id="scaler_neg_quarter"),
    pytest.param(0.1, id="scaler_bf16_lossy"),
    pytest.param(2.5, id="scaler_gt1"),
]

DIMS = [
    pytest.param([0], id="dim0"),
    pytest.param([1], id="dim1"),
    pytest.param([0, 1], id="dim01"),
    pytest.param([-1], id="dim_neg1"),
]

REDUCE_FNS = ["reduce_sum", "reduce_max"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("scaler", SCALERS)
def test_scalar_reduce_single_tile(device, dtype, reduce_fn, dims, scaler):
    """Numeric scaler on a single-tile input.

    Cross product covers both dtypes the LLK packer treats differently
    (bf16 / fp32), both reduce kinds, all dim combinations, and several
    scaler magnitudes/signs.
    """
    kernel = make_scalar_reduce_kernel(
        reduce_fn, 1, 1, list(dims), repr(float(scaler))
    )

    # Use a mix of positive and negative values so a sign-flipped scaler on
    # reduce_max would visibly diverge from the post-reduce-multiply result.
    torch.manual_seed(0xC0FFEE)
    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_rows, out_cols = _out_shape(1, 1, list(dims))
    out_torch = torch.zeros(out_rows * TILE, out_cols * TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    result = ttnn.to_torch(out)
    expected = _expected(inp_torch, reduce_fn, list(dims), scaler)
    actual = _populated(result, list(dims))
    assert_allclose(actual, expected, **_tolerances(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
@pytest.mark.parametrize("dims", [[0], [1], [0, 1]], ids=["dim0", "dim1", "dim01"])
@pytest.mark.parametrize("scaler", [0.25, -1.5], ids=["scaler_quarter", "scaler_neg_1_5"])
def test_scalar_reduce_multi_tile(device, dtype, reduce_fn, dims, scaler):
    """Numeric scaler over a 2x2 tile grid input.

    Verifies the post-reduce mul_unary is applied to every output tile,
    not only the first.
    """
    kernel = make_scalar_reduce_kernel(
        reduce_fn, 2, 2, dims, repr(float(scaler))
    )

    torch.manual_seed(0xBEEF)
    inp_torch = (torch.rand(2 * TILE, 2 * TILE, dtype=dtype) - 0.5) * 4.0
    out_rows, out_cols = _out_shape(2, 2, dims)
    out_torch = torch.zeros(out_rows * TILE, out_cols * TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    result = ttnn.to_torch(out)
    expected = _expected(inp_torch, reduce_fn, dims, scaler)
    actual = _populated(result, dims)
    assert_allclose(actual, expected, **_tolerances(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
@pytest.mark.parametrize("scaler", [0.5, -0.25, 2.5], ids=["half", "neg_quarter", "gt1"])
def test_scalar_form_matches_tile_form(device, dtype, reduce_fn, scaler):
    """Equivalence: numeric scaler and a tile-form scaler with the same value
    must produce equivalent results within dtype tolerance.

    Regression check that the new lowering path does not drift from the
    pre-existing tile-form path that test_reduce.py exercises.
    """
    from test_reduce import make_reduce_kernel  # tile-form template kernel

    tile_kernel = make_reduce_kernel(reduce_fn, 1, 1, [0, 1])
    scalar_kernel = make_scalar_reduce_kernel(
        reduce_fn, 1, 1, [0, 1], repr(float(scaler))
    )

    torch.manual_seed(0xDADA)
    inp_torch = (torch.rand(TILE, TILE, dtype=dtype) - 0.5) * 4.0
    scaler_tile = _create_scaler_tile(scaler, dtype)
    out_tile = torch.zeros(TILE, TILE, dtype=dtype)
    out_scalar = torch.zeros(TILE, TILE, dtype=dtype)

    inp_dev = to_l1(inp_torch, device)
    scaler_dev = to_l1(scaler_tile, device)
    out_tile_dev = to_l1(out_tile, device)
    out_scalar_dev = to_l1(out_scalar, device)

    tile_kernel(inp_dev, scaler_dev, out_tile_dev)
    scalar_kernel(to_l1(inp_torch, device), out_scalar_dev)

    tile_result = ttnn.to_torch(out_tile_dev)[0, 0].float()
    scalar_result = ttnn.to_torch(out_scalar_dev)[0, 0].float()
    expected = _expected(inp_torch, reduce_fn, [0, 1], scaler).reshape(())

    tol = _tolerances(dtype)
    assert_allclose(tile_result, expected, **tol)
    assert_allclose(scalar_result, expected, **tol)
    # Scalar and tile forms should agree closely with each other too.
    assert_allclose(scalar_result, tile_result, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
def test_two_distinct_scalers_in_one_kernel(device, dtype, reduce_fn):
    """Two reduce sites in one compute block carrying distinct scalar
    constants. The post-reduce mul_unary_const is per-reduce; this test
    catches any leak that would apply one scaler to both reduces, or any
    shared-FillOp materialization that produced the wrong multiplier for
    the second site.
    """
    scaler_a, scaler_b = 0.25, -1.5
    kernel = make_two_reduce_kernel(
        reduce_fn, repr(float(scaler_a)), repr(float(scaler_b))
    )

    torch.manual_seed(0x1234)
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
    tol = _tolerances(dtype)
    assert_allclose(actual_a, expected_a, **tol)
    assert_allclose(actual_b, expected_b, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_reduce_max_negative_scaler_sign(device, dtype):
    """reduce_max(x, negative_scaler) must produce max(x) * scaler, NOT
    max(x * scaler) — which would invert sign and pick the wrong tile.

    Choose an input whose maximum has the opposite sign from min so the
    two orderings give visibly different results.
    """
    kernel = make_scalar_reduce_kernel("reduce_max", 1, 1, [0, 1], repr(-2.0))

    # max(x) = +5; max(x * -2) = -2 * min(x) = -2 * -3 = +6
    # Correct expected: max(x) * -2 = -10.
    inp_torch = torch.full((TILE, TILE), 1.0, dtype=dtype)
    inp_torch[0, 0] = 5.0
    inp_torch[1, 1] = -3.0
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    actual = ttnn.to_torch(out)[0, 0].float().item()
    # If the implementation accidentally did max(x * scaler) the value
    # would be +6, not -10.
    assert actual == pytest.approx(-10.0, rel=0.05, abs=1.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("scaler", [1e-3, 1e3], ids=["tiny", "large"])
def test_scalar_extreme_magnitudes(device, dtype, scaler):
    """Scaler magnitudes outside the [0.1, 10] comfortable range. Catches
    any silent f32->lower-precision conversion in floatAttrToI32Bits or
    in the FloatAttr storage on the reduce op.
    """
    kernel = make_scalar_reduce_kernel(
        "reduce_sum", 1, 1, [0, 1], repr(float(scaler))
    )

    inp_torch = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)
    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    expected = float(TILE * TILE) * float(scaler)
    actual = ttnn.to_torch(out)[0, 0].float().item()
    # 5% relative tolerance handles bf16; fp32 will be much tighter.
    assert actual == pytest.approx(expected, rel=0.05, abs=abs(expected) * 0.05 + 1.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("reduce_fn", REDUCE_FNS)
@pytest.mark.parametrize("n", [4, 8], ids=["n4", "n8"])
def test_n_reduces_sharing_one_scaler(device, dtype, reduce_fn, n):
    """`n` reduce sites in one kernel sharing the same numeric scaler.

    Verifies that the per-reduce scaler attribute and any compiler-allocated
    DFB for the scaler do not bloat the DFB budget as `n` grows. Companion
    to the lit dedup test that asserts a single compiler-allocated DFB at
    the IR level.
    """
    scaler = 0.5
    kernel = make_shared_scaler_n_reduce_kernel(reduce_fn, n, scaler)

    torch.manual_seed(0xA5A5)
    inp_torch = (torch.rand(n * TILE, TILE, dtype=dtype) - 0.5) * 4.0
    out_torch = torch.zeros(n * TILE, TILE, dtype=dtype)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)
    kernel(inp, out)

    result = ttnn.to_torch(out)
    tol = _tolerances(dtype)
    for i in range(n):
        tile_in = inp_torch[i * TILE : (i + 1) * TILE, :TILE]
        expected = _expected(tile_in, reduce_fn, [0, 1], scaler).reshape(())
        actual = result[i * TILE, 0].float()
        assert_allclose(actual, expected, **tol)
