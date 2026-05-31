# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for the dst accumulation strategy option."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32
N_ITERS = 3
MULTI_TILE_SHAPE = (2, 2)
GRID_COLS = 2
GRID_ROWS = 2

MULTI_TILE_ROWS = MULTI_TILE_SHAPE[0] * TILE
MULTI_TILE_COLS = MULTI_TILE_SHAPE[1] * TILE
MULTICORE_ROWS = GRID_ROWS * TILE
MULTICORE_COLS = GRID_COLS * TILE

_DTYPE_TOL = {
    torch.bfloat16: dict(rtol=5e-2, atol=1.0),
    torch.float32: dict(rtol=1e-3, atol=1e-3),
}


def _make_tiled_constant_tensor(tile_rows, tile_cols, dtype, base):
    tensor = torch.empty((tile_rows * TILE, tile_cols * TILE), dtype=dtype)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            value = base + tile_row * tile_cols + tile_col
            tensor[
                tile_row * TILE : (tile_row + 1) * TILE,
                tile_col * TILE : (tile_col + 1) * TILE,
            ] = value
    return tensor


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
                    initial[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]],
                    initial_blk,
                ).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(
                        delta[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]],
                        delta_blk,
                    ).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(
                    out_blk,
                    out[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]],
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
            x, y = ttl.node(dims=2)
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[y, x], initial_blk).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(delta[y, x], delta_blk).wait()

        @ttl.datamovement()
        def writer():
            x, y = ttl.node(dims=2)
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[y, x]).wait()

    return kernel


def _make_dfb_accumulation_kernel():
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


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ["auto", "dst", "l1-pack"],
    ids=["auto", "dst", "l1"],
)
def test_multi_tile_tensor_recurrence_strategy(device, dtype, accumulation_strategy):
    """The dst strategy must agree with other strategies for multi-tile tensor
    recurrences."""
    initial = _make_tiled_constant_tensor(
        MULTI_TILE_SHAPE[0], MULTI_TILE_SHAPE[1], dtype, base=4.0
    )
    delta = _make_tiled_constant_tensor(
        MULTI_TILE_SHAPE[0], MULTI_TILE_SHAPE[1], dtype, base=2.0
    )
    out = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)
    expected = initial.float() + N_ITERS * delta.float()

    initial_dev = to_dram(initial, device)
    delta_dev = to_dram(delta, device)
    out_dev = to_dram(out, device)

    kernel = _make_multi_tile_tensor_recurrence_kernel()
    kernel(
        initial_dev,
        delta_dev,
        out_dev,
        options=f"--ttl-accumulation-strategy={accumulation_strategy}",
    )
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "accumulation_strategy",
    ["auto", "dst", "l1-pack"],
    ids=["auto", "dst", "l1"],
)
def test_multicore_tensor_recurrence_strategy(device, dtype, accumulation_strategy):
    """The dst strategy must agree with other strategies for independent
    per-core tensor recurrences."""
    initial = _make_tiled_constant_tensor(GRID_ROWS, GRID_COLS, dtype, base=4.0)
    delta = _make_tiled_constant_tensor(GRID_ROWS, GRID_COLS, dtype, base=2.0)
    out = torch.zeros((MULTICORE_ROWS, MULTICORE_COLS), dtype=dtype)
    expected = initial.float() + N_ITERS * delta.float()

    initial_dev = to_dram(initial, device)
    delta_dev = to_dram(delta, device)
    out_dev = to_dram(out, device)

    kernel = _make_multicore_tensor_recurrence_kernel()
    kernel(
        initial_dev,
        delta_dev,
        out_dev,
        options=f"--ttl-accumulation-strategy={accumulation_strategy}",
    )
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_dfb_accumulation_ignores_dst_tensor_strategy(device, dtype):
    """`--ttl-accumulation-strategy=dst` selects tensor recurrence storage.
    User-written DFB accumulation still lowers to L1 packer accumulation."""
    inp = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_ITERS * inp.float()

    inp_dev = to_dram(inp, device)
    out_dev = to_dram(out, device)

    kernel = _make_dfb_accumulation_kernel()
    kernel(inp_dev, out_dev, options="--ttl-accumulation-strategy=dst")
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])
