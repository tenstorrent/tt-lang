# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for supported accumulation strategy options."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32
N_ITERS = 3
N_DFB_INITIAL_MODE_ITERS = 32
N_L1_PACK_PRECISION_ITERS = 32
N_TRIP_COUNT_ONE_ITERS = 1
MULTI_TILE_BLOCK_ROWS = 2
MULTI_TILE_BLOCK_COLS = 2
MULTI_TILE_SHAPE = (MULTI_TILE_BLOCK_ROWS, MULTI_TILE_BLOCK_COLS)
GRID_COLS = 2
GRID_ROWS = 2
ACCUMULATION_STRATEGIES = ["auto", "dst", "l1-pack"]
ACCUMULATION_STRATEGY_IDS = ["auto", "dst", "l1"]

MULTI_TILE_ROWS = MULTI_TILE_BLOCK_ROWS * TILE
MULTI_TILE_COLS = MULTI_TILE_BLOCK_COLS * TILE
MULTICORE_ROWS = GRID_ROWS * TILE
MULTICORE_COLS = GRID_COLS * TILE
MULTICORE_MULTI_TILE_ROWS = GRID_ROWS * MULTI_TILE_BLOCK_ROWS * TILE
MULTICORE_MULTI_TILE_COLS = GRID_COLS * MULTI_TILE_BLOCK_COLS * TILE

_DTYPE_TOL = {
    torch.bfloat16: dict(rtol=5e-2, atol=1.0),
    torch.float32: dict(rtol=1e-3, atol=1e-3),
}


def _make_tiled_constant_tensor(tile_rows, tile_cols, dtype, base):
    """Create a tensor whose tile values encode logical tile coordinates."""
    tensor = torch.empty((tile_rows * TILE, tile_cols * TILE), dtype=dtype)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            value = base + tile_row * tile_cols + tile_col
            tensor[
                tile_row * TILE : (tile_row + 1) * TILE,
                tile_col * TILE : (tile_col + 1) * TILE,
            ] = value
    return tensor


def _run_accumulation_kernel(
    kernel,
    in_tensors,
    out_tensor,
    expected,
    dtype,
    device,
    accumulation_strategy,
    rtol=None,
    atol=None,
):
    """Run one accumulation kernel with an explicit strategy option."""
    in_devs = [to_dram(tensor, device) for tensor in in_tensors]
    out_dev = to_dram(out_tensor, device)
    kernel(
        *in_devs,
        out_dev,
        options=f"--ttl-accumulation-strategy={accumulation_strategy}",
    )
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    tolerances = dict(_DTYPE_TOL[dtype])
    if rtol is not None:
        tolerances["rtol"] = rtol
    if atol is not None:
        tolerances["atol"] = atol
    assert_allclose(result, expected.float(), **tolerances)


def _make_single_tile_tensor_recurrence_kernel(iterations):
    @ttl.operation(grid=(1, 1))
    def kernel(initial, delta, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=(1, 1), block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=iterations
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as acc:
                for _ in range(iterations):
                    with delta_dfb.wait() as delta_blk:
                        acc = acc + delta_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

        @ttl.datamovement()
        def reader():
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[0:1, 0:1], initial_blk).wait()
            for _ in range(iterations):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(delta[0:1, 0:1], delta_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_tensor_recurrence_epilogue_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(initial, delta, scale, bias, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=(1, 1), block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS
        )
        scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), block_count=2)
        bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with (
                initial_dfb.wait() as acc,
                scale_dfb.wait() as scale_blk,
                bias_dfb.wait() as bias_blk,
            ):
                for _ in range(N_ITERS):
                    with delta_dfb.wait() as delta_blk:
                        acc = acc + delta_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc * scale_blk + bias_blk)

        @ttl.datamovement()
        def reader():
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[0:1, 0:1], initial_blk).wait()
            with scale_dfb.reserve() as scale_blk:
                ttl.copy(scale[0:1, 0:1], scale_blk).wait()
            with bias_dfb.reserve() as bias_blk:
                ttl.copy(bias[0:1, 0:1], bias_blk).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(delta[0:1, 0:1], delta_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_multi_tile_tensor_recurrence_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(initial, delta, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=MULTI_TILE_SHAPE, block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            delta, shape=MULTI_TILE_SHAPE, block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=MULTI_TILE_SHAPE, block_count=2
        )

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as acc:
                for _ in range(N_ITERS):
                    with delta_dfb.wait() as delta_blk:
                        acc = acc + delta_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

        @ttl.datamovement()
        def reader():
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(
                    initial[0:MULTI_TILE_BLOCK_ROWS, 0:MULTI_TILE_BLOCK_COLS],
                    initial_blk,
                ).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(
                        delta[0:MULTI_TILE_BLOCK_ROWS, 0:MULTI_TILE_BLOCK_COLS],
                        delta_blk,
                    ).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(
                    out_blk,
                    out[0:MULTI_TILE_BLOCK_ROWS, 0:MULTI_TILE_BLOCK_COLS],
                ).wait()

    return kernel


def _make_multicore_tensor_recurrence_kernel():
    @ttl.operation(grid=(GRID_COLS, GRID_ROWS))
    def kernel(initial, delta, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=(1, 1), block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as acc:
                for _ in range(N_ITERS):
                    with delta_dfb.wait() as delta_blk:
                        acc = acc + delta_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

        @ttl.datamovement()
        def reader():
            node_col, node_row = ttl.node(dims=2)
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[node_row, node_col], initial_blk).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(delta[node_row, node_col], delta_blk).wait()

        @ttl.datamovement()
        def writer():
            node_col, node_row = ttl.node(dims=2)
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[node_row, node_col]).wait()

    return kernel


def _make_multicore_multi_tile_tensor_recurrence_kernel():
    @ttl.operation(grid=(GRID_COLS, GRID_ROWS))
    def kernel(initial, delta, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=MULTI_TILE_SHAPE, block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            delta, shape=MULTI_TILE_SHAPE, block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=MULTI_TILE_SHAPE, block_count=2
        )

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as acc:
                for _ in range(N_ITERS):
                    with delta_dfb.wait() as delta_blk:
                        acc = acc + delta_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

        @ttl.datamovement()
        def reader():
            node_col, node_row = ttl.node(dims=2)
            row = node_row * MULTI_TILE_BLOCK_ROWS
            col = node_col * MULTI_TILE_BLOCK_COLS
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(
                    initial[
                        row : row + MULTI_TILE_BLOCK_ROWS,
                        col : col + MULTI_TILE_BLOCK_COLS,
                    ],
                    initial_blk,
                ).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(
                        delta[
                            row : row + MULTI_TILE_BLOCK_ROWS,
                            col : col + MULTI_TILE_BLOCK_COLS,
                        ],
                        delta_blk,
                    ).wait()

        @ttl.datamovement()
        def writer():
            node_col, node_row = ttl.node(dims=2)
            row = node_row * MULTI_TILE_BLOCK_ROWS
            col = node_col * MULTI_TILE_BLOCK_COLS
            with out_dfb.wait() as out_blk:
                ttl.copy(
                    out_blk,
                    out[
                        row : row + MULTI_TILE_BLOCK_ROWS,
                        col : col + MULTI_TILE_BLOCK_COLS,
                    ],
                ).wait()

    return kernel


def _make_single_tile_dfb_accumulation_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(inp, out):
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=N_ITERS)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            for _ in range(N_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(inp[0:1, 0:1], inp_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_fused_elementwise_dfb_accumulation_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(lhs, rhs, out):
        lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=N_ITERS)
        rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=N_ITERS)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with lhs_dfb.wait() as lhs_blk, rhs_dfb.wait() as rhs_blk:
                        out_blk += lhs_blk + rhs_blk

        @ttl.datamovement()
        def reader():
            for _ in range(N_ITERS):
                with lhs_dfb.reserve() as lhs_blk:
                    ttl.copy(lhs[0:1, 0:1], lhs_blk).wait()
                with rhs_dfb.reserve() as rhs_blk:
                    ttl.copy(rhs[0:1, 0:1], rhs_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_fused_matmul_dfb_accumulation_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(lhs, rhs, out):
        lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=N_ITERS)
        rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=N_ITERS)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with lhs_dfb.wait() as lhs_blk, rhs_dfb.wait() as rhs_blk:
                        out_blk += lhs_blk @ rhs_blk

        @ttl.datamovement()
        def reader():
            for kt in range(N_ITERS):
                with lhs_dfb.reserve() as lhs_blk:
                    ttl.copy(lhs[kt : kt + 1, 0:1], lhs_blk).wait()
                with rhs_dfb.reserve() as rhs_blk:
                    ttl.copy(rhs[kt : kt + 1, 0:1], rhs_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_seeded_dfb_accumulate_existing_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(seed, inp, out):
        seed_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
        inp_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=N_DFB_INITIAL_MODE_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with seed_dfb.wait() as seed_blk, out_dfb.reserve() as out_blk:
                out_blk.store(seed_blk)
                for _ in range(N_DFB_INITIAL_MODE_ITERS):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            with seed_dfb.reserve() as seed_blk:
                ttl.copy(seed[0:1, 0:1], seed_blk).wait()
            for _ in range(N_DFB_INITIAL_MODE_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(inp[0:1, 0:1], inp_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_guarded_seeded_dfb_accumulate_existing_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(seed, inp, out):
        seed_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
        inp_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=N_DFB_INITIAL_MODE_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            node_col, _ = ttl.node(dims=2)
            route_is_active = node_col == 0
            with seed_dfb.wait() as seed_blk:
                if route_is_active:
                    out_blk = out_dfb.reserve()
                    out_blk.store(seed_blk)

            for _ in range(N_DFB_INITIAL_MODE_ITERS):
                with inp_dfb.wait() as inp_blk:
                    if route_is_active:
                        out_blk += inp_blk

            if route_is_active:
                out_blk.push()

        @ttl.datamovement()
        def reader():
            with seed_dfb.reserve() as seed_blk:
                ttl.copy(seed[0:1, 0:1], seed_blk).wait()
            for _ in range(N_DFB_INITIAL_MODE_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(inp[0:1, 0:1], inp_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_reused_slot_dfb_overwrite_kernel(iterations):
    @ttl.operation(grid=(1, 1))
    def kernel(seed, inp, out):
        seed_dfb = ttl.make_dataflow_buffer_like(seed, shape=(1, 1), block_count=2)
        inp_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=iterations
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            with seed_dfb.wait() as seed_blk, out_dfb.reserve() as out_blk:
                out_blk.store(seed_blk)

            with out_dfb.reserve() as out_blk:
                for _ in range(iterations):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            with seed_dfb.reserve() as seed_blk:
                ttl.copy(seed[0:1, 0:1], seed_blk).wait()
            for _ in range(iterations):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(inp[0:1, 0:1], inp_blk).wait()

        @ttl.datamovement()
        def writer():
            for _ in range(2):
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_multi_tile_dfb_accumulation_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(inp, out):
        inp_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=MULTI_TILE_SHAPE, block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=MULTI_TILE_SHAPE, block_count=2
        )

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            for _ in range(N_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(
                        inp[0:MULTI_TILE_BLOCK_ROWS, 0:MULTI_TILE_BLOCK_COLS],
                        inp_blk,
                    ).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(
                    out_blk,
                    out[0:MULTI_TILE_BLOCK_ROWS, 0:MULTI_TILE_BLOCK_COLS],
                ).wait()

    return kernel


def _make_multicore_dfb_accumulation_kernel():
    @ttl.operation(grid=(GRID_COLS, GRID_ROWS))
    def kernel(inp, out):
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=N_ITERS)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            node_col, node_row = ttl.node(dims=2)
            for _ in range(N_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(inp[node_row, node_col], inp_blk).wait()

        @ttl.datamovement()
        def writer():
            node_col, node_row = ttl.node(dims=2)
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[node_row, node_col]).wait()

    return kernel


def _make_multicore_multi_tile_dfb_accumulation_kernel():
    @ttl.operation(grid=(GRID_COLS, GRID_ROWS))
    def kernel(inp, out):
        inp_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=MULTI_TILE_SHAPE, block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=MULTI_TILE_SHAPE, block_count=2
        )

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            node_col, node_row = ttl.node(dims=2)
            row = node_row * MULTI_TILE_BLOCK_ROWS
            col = node_col * MULTI_TILE_BLOCK_COLS
            for _ in range(N_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(
                        inp[
                            row : row + MULTI_TILE_BLOCK_ROWS,
                            col : col + MULTI_TILE_BLOCK_COLS,
                        ],
                        inp_blk,
                    ).wait()

        @ttl.datamovement()
        def writer():
            node_col, node_row = ttl.node(dims=2)
            row = node_row * MULTI_TILE_BLOCK_ROWS
            col = node_col * MULTI_TILE_BLOCK_COLS
            with out_dfb.wait() as out_blk:
                ttl.copy(
                    out_blk,
                    out[
                        row : row + MULTI_TILE_BLOCK_ROWS,
                        col : col + MULTI_TILE_BLOCK_COLS,
                    ],
                ).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_multi_tile_tensor_recurrence_strategy(device, dtype, accumulation_strategy):
    """Supported strategies preserve multi-tile tensor recurrences."""
    initial = _make_tiled_constant_tensor(
        MULTI_TILE_BLOCK_ROWS, MULTI_TILE_BLOCK_COLS, dtype, base=4.0
    )
    delta = _make_tiled_constant_tensor(
        MULTI_TILE_BLOCK_ROWS, MULTI_TILE_BLOCK_COLS, dtype, base=2.0
    )
    out = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)
    expected = initial.float() + N_ITERS * delta.float()
    _run_accumulation_kernel(
        _make_multi_tile_tensor_recurrence_kernel(),
        [initial, delta],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_multicore_tensor_recurrence_strategy(device, dtype, accumulation_strategy):
    """Supported strategies preserve per-core tensor recurrences."""
    initial = _make_tiled_constant_tensor(GRID_ROWS, GRID_COLS, dtype, base=4.0)
    delta = _make_tiled_constant_tensor(GRID_ROWS, GRID_COLS, dtype, base=2.0)
    out = torch.zeros((MULTICORE_ROWS, MULTICORE_COLS), dtype=dtype)
    expected = initial.float() + N_ITERS * delta.float()
    _run_accumulation_kernel(
        _make_multicore_tensor_recurrence_kernel(),
        [initial, delta],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_multicore_multi_tile_tensor_recurrence_strategy(
    device, dtype, accumulation_strategy
):
    """Supported strategies preserve multi-tile tensor recurrences on every core."""
    initial = _make_tiled_constant_tensor(
        GRID_ROWS * MULTI_TILE_BLOCK_ROWS,
        GRID_COLS * MULTI_TILE_BLOCK_COLS,
        dtype,
        base=4.0,
    )
    delta = _make_tiled_constant_tensor(
        GRID_ROWS * MULTI_TILE_BLOCK_ROWS,
        GRID_COLS * MULTI_TILE_BLOCK_COLS,
        dtype,
        base=2.0,
    )
    out = torch.zeros(
        (MULTICORE_MULTI_TILE_ROWS, MULTICORE_MULTI_TILE_COLS), dtype=dtype
    )
    expected = initial.float() + N_ITERS * delta.float()
    _run_accumulation_kernel(
        _make_multicore_multi_tile_tensor_recurrence_kernel(),
        [initial, delta],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_tensor_recurrence_with_fused_epilogue(device, dtype, accumulation_strategy):
    """Accumulation strategies preserve fused consumers of the recurrent value."""
    initial = torch.full((TILE, TILE), 1.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 0.5, dtype=dtype)
    scale = torch.full((TILE, TILE), 2.0, dtype=dtype)
    bias = torch.full((TILE, TILE), 3.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = (initial.float() + N_ITERS * delta.float()) * scale.float()
    expected = expected + bias.float()
    _run_accumulation_kernel(
        _make_tensor_recurrence_epilogue_kernel(),
        [initial, delta, scale, bias],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
def test_l1_pack_fp32_non_exact_accumulation_precision(device):
    """L1 packer accumulation handles non-exact fp32 increments."""
    initial = torch.full((TILE, TILE), 0.3, dtype=torch.float32)
    delta = torch.full((TILE, TILE), 0.1, dtype=torch.float32)
    out = torch.zeros((TILE, TILE), dtype=torch.float32)
    expected = initial + N_L1_PACK_PRECISION_ITERS * delta
    _run_accumulation_kernel(
        _make_single_tile_tensor_recurrence_kernel(N_L1_PACK_PRECISION_ITERS),
        [initial, delta],
        out,
        expected,
        torch.float32,
        device,
        "l1-pack",
        rtol=3e-4,
        atol=1e-4,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_single_tile_dfb_accumulation_strategy(device, dtype, accumulation_strategy):
    """Supported strategies preserve single-tile DFB accumulation."""
    inp = _make_tiled_constant_tensor(1, 1, dtype, base=1.0)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_single_tile_dfb_accumulation_kernel(),
        [inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_fused_elementwise_dfb_accumulation_strategy(
    device, dtype, accumulation_strategy
):
    """DFB += supports fused elementwise RHS expressions under every strategy."""
    lhs = torch.full((TILE, TILE), 1.0, dtype=dtype)
    rhs = torch.full((TILE, TILE), 0.25, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_ITERS * (lhs.float() + rhs.float())
    _run_accumulation_kernel(
        _make_fused_elementwise_dfb_accumulation_kernel(),
        [lhs, rhs],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_fused_matmul_dfb_accumulation_ignores_dst_strategy(device, dtype):
    """Forced DST remains L1 packer accumulation for DFB += with matmul RHS."""
    lhs = torch.full((N_ITERS * TILE, TILE), 0.125, dtype=dtype)
    rhs = torch.full((N_ITERS * TILE, TILE), 0.25, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = torch.zeros((TILE, TILE), dtype=torch.float32)
    for kt in range(N_ITERS):
        lhs_tile = lhs[kt * TILE : (kt + 1) * TILE, :].float()
        rhs_tile = rhs[kt * TILE : (kt + 1) * TILE, :].float()
        expected = expected + lhs_tile @ rhs_tile

    _run_accumulation_kernel(
        _make_fused_matmul_dfb_accumulation_kernel(),
        [lhs, rhs],
        out,
        expected,
        dtype,
        device,
        "dst",
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_seeded_dfb_accumulation_uses_existing_output(
    device, dtype, accumulation_strategy
):
    """A prior store into the reserved output block must remain part of DFB +=."""
    seed = _make_tiled_constant_tensor(1, 1, dtype, base=3.0)
    inp = _make_tiled_constant_tensor(1, 1, dtype, base=0.125)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = seed.float() + N_DFB_INITIAL_MODE_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_seeded_dfb_accumulate_existing_kernel(),
        [seed, inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_guarded_seeded_dfb_accumulation_uses_existing_output(
    device, dtype, accumulation_strategy
):
    """A same-guard seed store must remain part of guarded DFB +=."""
    seed = _make_tiled_constant_tensor(1, 1, dtype, base=5.0)
    inp = _make_tiled_constant_tensor(1, 1, dtype, base=0.125)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = seed.float() + N_DFB_INITIAL_MODE_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_guarded_seeded_dfb_accumulate_existing_kernel(),
        [seed, inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_reused_dfb_slot_overwrite_discards_prior_output(
    device, dtype, accumulation_strategy
):
    """DFB += without an in-reservation seed must overwrite on the first update."""
    seed = _make_tiled_constant_tensor(1, 1, dtype, base=7.0)
    inp = _make_tiled_constant_tensor(1, 1, dtype, base=0.125)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_DFB_INITIAL_MODE_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_reused_slot_dfb_overwrite_kernel(N_DFB_INITIAL_MODE_ITERS),
        [seed, inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_reused_dfb_slot_trip_count_one_overwrite(device, dtype, accumulation_strategy):
    """A single DFB += update must discard stale output slot contents."""
    seed = _make_tiled_constant_tensor(1, 1, dtype, base=7.0)
    inp = _make_tiled_constant_tensor(1, 1, dtype, base=0.125)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_TRIP_COUNT_ONE_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_reused_slot_dfb_overwrite_kernel(N_TRIP_COUNT_ONE_ITERS),
        [seed, inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_multi_tile_dfb_accumulation_strategy(device, dtype, accumulation_strategy):
    """Supported strategies preserve multi-tile DFB accumulation."""
    inp = _make_tiled_constant_tensor(
        MULTI_TILE_BLOCK_ROWS, MULTI_TILE_BLOCK_COLS, dtype, base=1.0
    )
    out = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)
    expected = N_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_multi_tile_dfb_accumulation_kernel(),
        [inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_multicore_dfb_accumulation_strategy(device, dtype, accumulation_strategy):
    """Supported strategies preserve per-core DFB accumulation."""
    inp = _make_tiled_constant_tensor(GRID_ROWS, GRID_COLS, dtype, base=1.0)
    out = torch.zeros((MULTICORE_ROWS, MULTICORE_COLS), dtype=dtype)
    expected = N_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_multicore_dfb_accumulation_kernel(),
        [inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ACCUMULATION_STRATEGIES,
    ids=ACCUMULATION_STRATEGY_IDS,
)
def test_multicore_multi_tile_dfb_accumulation_strategy(
    device, dtype, accumulation_strategy
):
    """Supported strategies preserve multi-tile DFB accumulation on every core."""
    inp = _make_tiled_constant_tensor(
        GRID_ROWS * MULTI_TILE_BLOCK_ROWS,
        GRID_COLS * MULTI_TILE_BLOCK_COLS,
        dtype,
        base=1.0,
    )
    out = torch.zeros(
        (MULTICORE_MULTI_TILE_ROWS, MULTICORE_MULTI_TILE_COLS), dtype=dtype
    )
    expected = N_ITERS * inp.float()
    _run_accumulation_kernel(
        _make_multicore_multi_tile_dfb_accumulation_kernel(),
        [inp],
        out,
        expected,
        dtype,
        device,
        accumulation_strategy,
    )
