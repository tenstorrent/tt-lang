# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Three-pass streaming LayerNorm: mean, variance, normalize+affine.

Three factories exercising different sync styles on the same kernel shape:

* `make_layernorm_kernel` — assignment style, no `with:`, no explicit
  `.push()` / `.pop()`. The case that deadlocks on hardware without the
  `TTLInsertCBSync` fix from issue #524.
* `make_layernorm_kernel_explicit` — user-written `.push()` / `.pop()`
  after every `.reserve()` / `.wait()`. Bypasses the sync pass and
  serves as the hand-synced reference.
* `make_layernorm_kernel_minimal_dfbs` — `with:` context managers plus
  `+=` L1 accumulation into user-carry DFBs for mean and inv_std.
  Exercises the multi-store-into-one-reserved-slot pattern that
  interacts with `ConvertTTLToCompute`'s `cb_push` relocation.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32


def _torch_layernorm(x, weight, bias, eps=1e-6):
    """Row-wise layernorm with per-element weight/bias (matching the kernel)."""
    xf = x.float()
    mean = xf.mean(dim=1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    return ((xf - mean) * inv_std) * weight.float() + bias.float()


# ---------------------------------------------------------------------------
# Shared allocation + data-movement helpers
# ---------------------------------------------------------------------------


def _alloc_dfbs(x, weight, ln_bias, scaler, mean_scale, out, *, full):
    """Allocate the layernorm DFB set.

    `full=True` includes the intermediate `red`/`acc`/`bcast`/`sq` DFBs used
    by the assignment- and explicit-sync factories. `full=False` is the
    minimal set for the `with:` + `+=` L1-accumulation factory, which keeps
    those intermediates SSA / compiler-allocated.
    """
    dfbs = {
        "x": ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2),
        "sc": ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1),
        "ms": ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), block_count=1),
        "w": ttl.make_dataflow_buffer_like(weight, shape=(1, 1), block_count=2),
        "b": ttl.make_dataflow_buffer_like(ln_bias, shape=(1, 1), block_count=2),
        "out": ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2),
        "mean": ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2),
        "istd": ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2),
    }
    if full:
        dfbs.update(
            {
                "red": ttl.make_dataflow_buffer_like(
                    scaler, shape=(1, 1), block_count=2
                ),
                "acc": ttl.make_dataflow_buffer_like(
                    scaler, shape=(1, 1), block_count=2
                ),
                "bcast": ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2),
                "sq": ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2),
            }
        )
    return dfbs


def _define_dm(
    x,
    weight,
    ln_bias,
    scaler,
    mean_scale,
    out,
    dfbs,
    tiles_per_core,
    seq_tiles,
    dim_tiles,
    *,
    use_with,
):
    """Register the `dm_read` and `dm_write` threads.

    Both variants load scaler + mean_scale once, read x three times (one
    pass per layernorm stage) plus weight / bias once, and write one
    output tile per input tile. `use_with=True` wraps every
    reserve/wait in a `with:` block; `use_with=False` uses bare
    `.reserve()` / `.wait()` calls."""
    x_dfb, sc_dfb, ms_dfb = dfbs["x"], dfbs["sc"], dfbs["ms"]
    w_dfb, b_dfb, out_dfb = dfbs["w"], dfbs["b"], dfbs["out"]

    if use_with:

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            with sc_dfb.reserve() as blk:
                ttl.copy(scaler[0, 0], blk).wait()
            with ms_dfb.reserve() as blk:
                ttl.copy(mean_scale[0, 0], blk).wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for _pass in range(3):
                        for j in range(dim_tiles):
                            with x_dfb.reserve() as blk:
                                ttl.copy(x[tile_idx, j], blk).wait()
                    for j in range(dim_tiles):
                        with w_dfb.reserve() as blk:
                            ttl.copy(weight[tile_idx, j], blk).wait()
                        with b_dfb.reserve() as blk:
                            ttl.copy(ln_bias[tile_idx, j], blk).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(dim_tiles):
                        with out_dfb.wait() as blk:
                            ttl.copy(blk, out[tile_idx, j]).wait()

        return

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        ttl.copy(scaler[0, 0], sc_dfb.reserve()).wait()
        ttl.copy(mean_scale[0, 0], ms_dfb.reserve()).wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for _pass in range(3):
                    for j in range(dim_tiles):
                        ttl.copy(x[tile_idx, j], x_dfb.reserve()).wait()
                for j in range(dim_tiles):
                    ttl.copy(weight[tile_idx, j], w_dfb.reserve()).wait()
                    ttl.copy(ln_bias[tile_idx, j], b_dfb.reserve()).wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(dim_tiles):
                    ttl.copy(out_dfb.wait(), out[tile_idx, j]).wait()


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def make_layernorm_kernel(dim_tiles):
    """Assignment-style layernorm — no `with:`, no explicit push/pop.
    The sync pass places every `cb_push` / `cb_pop`."""

    @ttl.operation(grid="auto")
    def layernorm_kernel(x, weight, ln_bias, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        dfbs = _alloc_dfbs(x, weight, ln_bias, scaler, mean_scale, out, full=True)
        x_dfb, sc_dfb, ms_dfb = dfbs["x"], dfbs["sc"], dfbs["ms"]
        w_dfb, b_dfb, out_dfb = dfbs["w"], dfbs["b"], dfbs["out"]
        mean_dfb, istd_dfb = dfbs["mean"], dfbs["istd"]
        red_dfb, acc_dfb = dfbs["red"], dfbs["acc"]
        bcast_dfb, sq_dfb = dfbs["bcast"], dfbs["sq"]

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            sc = sc_dfb.wait()
            ms = ms_dfb.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: mean
                    x0 = x_dfb.wait()
                    r = red_dfb.reserve()
                    r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))
                    rv = red_dfb.wait()
                    acc = acc_dfb.reserve()
                    acc.store(rv)
                    for _ in range(dim_tiles - 1):
                        xj = x_dfb.wait()
                        r = red_dfb.reserve()
                        r.store(ttl.math.reduce_sum(xj, sc, dims=[1]))
                        rv = red_dfb.wait()
                        av = acc_dfb.wait()
                        acc = acc_dfb.reserve()
                        acc.store(av + rv)
                    sum_x = acc_dfb.wait()
                    bc = bcast_dfb.reserve()
                    bc.store(ttl.math.broadcast(sum_x, bc, dims=[1]))
                    sum_x_bc = bcast_dfb.wait()
                    mean_out = mean_dfb.reserve()
                    mean_out.store(sum_x_bc * ms)
                    # Pass 2: variance
                    mean_val = mean_dfb.wait()
                    x0 = x_dfb.wait()
                    diff = x0 - mean_val
                    sq = sq_dfb.reserve()
                    sq.store(diff * diff)
                    sqv = sq_dfb.wait()
                    r = red_dfb.reserve()
                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    rv = red_dfb.wait()
                    acc = acc_dfb.reserve()
                    acc.store(rv)
                    for _ in range(dim_tiles - 1):
                        xj = x_dfb.wait()
                        diff = xj - mean_val
                        sq = sq_dfb.reserve()
                        sq.store(diff * diff)
                        sqv = sq_dfb.wait()
                        r = red_dfb.reserve()
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        rv = red_dfb.wait()
                        av = acc_dfb.wait()
                        acc = acc_dfb.reserve()
                        acc.store(av + rv)
                    sum_sq = acc_dfb.wait()
                    bc = bcast_dfb.reserve()
                    bc.store(ttl.math.broadcast(sum_sq, bc, dims=[1]))
                    var_bc = bcast_dfb.wait()
                    istd = istd_dfb.reserve()
                    istd.store(
                        ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6))
                    )
                    # Pass 3: normalize + affine
                    inv_std = istd_dfb.wait()
                    for _ in range(dim_tiles):
                        xj = x_dfb.wait()
                        wj = w_dfb.wait()
                        bj = b_dfb.wait()
                        o = out_dfb.reserve()
                        o.store((xj - mean_val) * inv_std * wj + bj)

        _define_dm(
            x,
            weight,
            ln_bias,
            scaler,
            mean_scale,
            out,
            dfbs,
            tiles_per_core,
            seq_tiles,
            dim_tiles,
            use_with=False,
        )

    return layernorm_kernel


def make_layernorm_kernel_explicit(dim_tiles):
    """Same kernel shape as `make_layernorm_kernel` with explicit
    `.push()` / `.pop()` after every reserve/wait. Bypasses the sync
    pass and serves as a hand-synced reference."""

    @ttl.operation(grid="auto")
    def layernorm_kernel(x, weight, ln_bias, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        dfbs = _alloc_dfbs(x, weight, ln_bias, scaler, mean_scale, out, full=True)
        x_dfb, sc_dfb, ms_dfb = dfbs["x"], dfbs["sc"], dfbs["ms"]
        w_dfb, b_dfb, out_dfb = dfbs["w"], dfbs["b"], dfbs["out"]
        mean_dfb, istd_dfb = dfbs["mean"], dfbs["istd"]
        red_dfb, acc_dfb = dfbs["red"], dfbs["acc"]
        bcast_dfb, sq_dfb = dfbs["bcast"], dfbs["sq"]

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            sc = sc_dfb.wait()
            ms = ms_dfb.wait()
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Pass 1: mean
                    x0 = x_dfb.wait()
                    r = red_dfb.reserve()
                    r.store(ttl.math.reduce_sum(x0, sc, dims=[1]))
                    r.push()
                    x0.pop()
                    rv = red_dfb.wait()
                    acc = acc_dfb.reserve()
                    acc.store(rv)
                    acc.push()
                    rv.pop()
                    for _ in range(dim_tiles - 1):
                        xj = x_dfb.wait()
                        r = red_dfb.reserve()
                        r.store(ttl.math.reduce_sum(xj, sc, dims=[1]))
                        r.push()
                        xj.pop()
                        rv = red_dfb.wait()
                        av = acc_dfb.wait()
                        acc = acc_dfb.reserve()
                        acc.store(av + rv)
                        acc.push()
                        rv.pop()
                        av.pop()
                    sum_x = acc_dfb.wait()
                    bc = bcast_dfb.reserve()
                    bc.store(ttl.math.broadcast(sum_x, bc, dims=[1]))
                    bc.push()
                    sum_x.pop()
                    sum_x_bc = bcast_dfb.wait()
                    mean_out = mean_dfb.reserve()
                    mean_out.store(sum_x_bc * ms)
                    mean_out.push()
                    sum_x_bc.pop()
                    # Pass 2: variance
                    mean_val = mean_dfb.wait()
                    x0 = x_dfb.wait()
                    diff = x0 - mean_val
                    sq = sq_dfb.reserve()
                    sq.store(diff * diff)
                    sq.push()
                    x0.pop()
                    sqv = sq_dfb.wait()
                    r = red_dfb.reserve()
                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                    r.push()
                    sqv.pop()
                    rv = red_dfb.wait()
                    acc = acc_dfb.reserve()
                    acc.store(rv)
                    acc.push()
                    rv.pop()
                    for _ in range(dim_tiles - 1):
                        xj = x_dfb.wait()
                        diff = xj - mean_val
                        sq = sq_dfb.reserve()
                        sq.store(diff * diff)
                        sq.push()
                        xj.pop()
                        sqv = sq_dfb.wait()
                        r = red_dfb.reserve()
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[1]))
                        r.push()
                        sqv.pop()
                        rv = red_dfb.wait()
                        av = acc_dfb.wait()
                        acc = acc_dfb.reserve()
                        acc.store(av + rv)
                        acc.push()
                        rv.pop()
                        av.pop()
                    sum_sq = acc_dfb.wait()
                    bc = bcast_dfb.reserve()
                    bc.store(ttl.math.broadcast(sum_sq, bc, dims=[1]))
                    bc.push()
                    sum_sq.pop()
                    var_bc = bcast_dfb.wait()
                    istd = istd_dfb.reserve()
                    istd.store(
                        ttl.math.rsqrt(var_bc * ms + ttl.math.fill(var_bc, 1e-6))
                    )
                    istd.push()
                    var_bc.pop()
                    # Pass 3: normalize + affine
                    inv_std = istd_dfb.wait()
                    for _ in range(dim_tiles):
                        xj = x_dfb.wait()
                        wj = w_dfb.wait()
                        bj = b_dfb.wait()
                        o = out_dfb.reserve()
                        o.store((xj - mean_val) * inv_std * wj + bj)
                        o.push()
                        xj.pop()
                        wj.pop()
                        bj.pop()
                    inv_std.pop()
                    mean_val.pop()

        _define_dm(
            x,
            weight,
            ln_bias,
            scaler,
            mean_scale,
            out,
            dfbs,
            tiles_per_core,
            seq_tiles,
            dim_tiles,
            use_with=False,
        )

    return layernorm_kernel


def make_layernorm_kernel_minimal_dfbs(dim_tiles):
    """Minimal-user-DFB layernorm: only inputs/outputs plus mean/inv_std
    carry. Intermediates are SSA or compiler-allocated. Uses `with:`
    context managers and `+=` L1 accumulation."""

    @ttl.operation(grid="auto")
    def layernorm_kernel(x, weight, ln_bias, scaler, mean_scale, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = x.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)
        dfbs = _alloc_dfbs(x, weight, ln_bias, scaler, mean_scale, out, full=False)
        x_dfb, sc_dfb, ms_dfb = dfbs["x"], dfbs["sc"], dfbs["ms"]
        w_dfb, b_dfb, out_dfb = dfbs["w"], dfbs["b"], dfbs["out"]
        mean_dfb, istd_dfb = dfbs["mean"], dfbs["istd"]

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        # Pass 1: mean via += L1 accumulation, then
                        # broadcast * ms.
                        with mean_dfb.reserve() as mean_blk:
                            mean_blk.store(ttl.math.fill(mean_blk, 0))
                            for _ in range(dim_tiles):
                                with x_dfb.wait() as xj:
                                    mean_blk += ttl.math.reduce_sum(xj, sc, dims=[1])
                            mean_blk.store(
                                ttl.math.broadcast(mean_blk, mean_blk, dims=[1]) * ms
                            )

                        # Pass 2 + Pass 3 share one mean wait scope.
                        with mean_dfb.wait() as mean_val:
                            # Pass 2: variance into istd_dfb, then rsqrt.
                            with istd_dfb.reserve() as var_blk:
                                var_blk.store(ttl.math.fill(var_blk, 0))
                                for _ in range(dim_tiles):
                                    with x_dfb.wait() as xj:
                                        diff = xj - mean_val
                                        var_blk += ttl.math.reduce_sum(
                                            diff * diff, sc, dims=[1]
                                        )
                                var_blk.store(
                                    ttl.math.rsqrt(
                                        ttl.math.broadcast(var_blk, var_blk, dims=[1])
                                        * ms
                                        + ttl.math.fill(var_blk, 1e-6)
                                    )
                                )

                            # Pass 3: normalize + affine.
                            with istd_dfb.wait() as inv_std:
                                for _ in range(dim_tiles):
                                    with (
                                        x_dfb.wait() as xj,
                                        w_dfb.wait() as wj,
                                        b_dfb.wait() as bj,
                                        out_dfb.reserve() as o,
                                    ):
                                        o.store((xj - mean_val) * inv_std * wj + bj)

        _define_dm(
            x,
            weight,
            ln_bias,
            scaler,
            mean_scale,
            out,
            dfbs,
            tiles_per_core,
            seq_tiles,
            dim_tiles,
            use_with=True,
        )

    return layernorm_kernel


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_layernorm(kernel_factory, seq_tiles, dim_tiles, device):
    torch.manual_seed(42)
    M, N = seq_tiles * TILE, dim_tiles * TILE
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(M, N, dtype=torch.bfloat16)
    bias = torch.randn(M, N, dtype=torch.bfloat16)
    scaler = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale = torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16)

    golden = _torch_layernorm(x, weight, bias)

    x_dev = to_dram(x, device)
    weight_dev = to_dram(weight, device)
    bias_dev = to_dram(bias, device)
    scaler_dev = to_dram(scaler, device)
    mean_scale_dev = to_dram(mean_scale, device)
    out_dev = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = kernel_factory(dim_tiles)
    kernel(x_dev, weight_dev, bias_dev, scaler_dev, mean_scale_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result, threshold=0.99)


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
def test_layernorm(seq_tiles, dim_tiles, device):
    """Assignment-style layernorm; the original reproducer for #524.
    Without the `TTLInsertCBSync` fix, `cb_push` for the intra-thread
    red / acc DFBs is placed after the `scf.for` body that reads them
    and the first iteration deadlocks."""
    _run_layernorm(make_layernorm_kernel, seq_tiles, dim_tiles, device)


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
def test_layernorm_explicit(seq_tiles, dim_tiles, device):
    """Hand-synced reference — push/pop written by the user bypass the
    sync pass."""
    _run_layernorm(make_layernorm_kernel_explicit, seq_tiles, dim_tiles, device)


@pytest.mark.parametrize("seq_tiles,dim_tiles", [(2, 2)], ids=["2x2"])
@pytest.mark.requires_device
@pytest.mark.xfail(
    strict=True,
    reason=(
        "PCC 0.9887 < 0.99 threshold on bf16 for the with: + += L1 "
        "accumulation path. Tracked in #526"
    ),
)
def test_layernorm_minimal_dfbs(seq_tiles, dim_tiles, device):
    """`with:` + `+=` L1 accumulation. Exercises the
    multi-store-per-reserve pattern that `ConvertTTLToCompute`'s
    `cb_push` relocation must not move ahead of subsequent stores."""
    _run_layernorm(make_layernorm_kernel_minimal_dfbs, seq_tiles, dim_tiles, device)
