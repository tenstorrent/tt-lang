# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Linear layer kernels for Qwen 2.5 0.5B.

Multi-core (13×10 = 130 cores on Blackhole):
  - linear_kernel: Y = X @ W (no bias)
  - linear_bias_kernel: Y = X @ W + bias

Uses ceiling-division work distribution: each core processes
ceil(num_output_tiles / num_cores) tiles. Cores with tile_id >= total
simply do no work (loop body is skipped via DFB flow).
"""

import torch
import ttl
import ttnn

TILE = 32
GRID_Y = 11
GRID_X = 10


@ttl.kernel(grid=(GRID_Y, GRID_X))
def linear_kernel(X, W, Y):
    """Y = X @ W. Multi-core, K-accumulation via prev + a @ b."""
    Mt = X.shape[0] // TILE
    Kt = X.shape[1] // TILE
    Nt = W.shape[1] // TILE
    num_output_tiles = Mt * Nt

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(W, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    # Each core gets a contiguous range of tiles
    # Core nid gets tiles [nid * chunk, (nid+1) * chunk), clamped to num_output_tiles
    chunk = (num_output_tiles + num_cores - 1) // num_cores  # ceil division

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        ntiles = chunk
        for tid in range(ntiles):
            out_idx = tile_start + tid
            # Clamp to last valid tile for excess cores
            clamped = out_idx % num_output_tiles
            m = clamped // Nt
            n = clamped % Nt
            for k in range(Kt):
                with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                    tx_x = ttl.copy(X[m, k], x_blk)
                    tx_x.wait()
                    tx_w = ttl.copy(W[k, n], w_blk)
                    tx_w.wait()

    @ttl.compute()
    def compute():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        ntiles = chunk
        for _ in range(ntiles):
            with x_dfb.wait() as x_blk, w_dfb.wait() as w_blk:
                with acc_dfb.reserve() as acc:
                    acc.store(x_blk @ w_blk)
            for _ in range(Kt - 1):
                with (
                    x_dfb.wait() as x_blk,
                    w_dfb.wait() as w_blk,
                    acc_dfb.wait() as prev,
                ):
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + x_blk @ w_blk)
            with acc_dfb.wait() as result:
                with y_dfb.reserve() as y_blk:
                    y_blk.store(result)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        ntiles = chunk
        for tid in range(ntiles):
            out_idx = tile_start + tid
            # Clamp to last valid tile for excess cores
            clamped = out_idx % num_output_tiles
            m = clamped // Nt
            n = clamped % Nt
            with y_dfb.wait() as y_blk:
                tx = ttl.copy(y_blk, Y[m, n])
                tx.wait()


@ttl.kernel(grid=(GRID_Y, GRID_X))
def linear_bias_kernel(X, W, bias, Y):
    """Y = X @ W + bias. Multi-core."""
    Mt = X.shape[0] // TILE
    Kt = X.shape[1] // TILE
    Nt = W.shape[1] // TILE
    num_output_tiles = Mt * Nt

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(W, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (num_output_tiles + num_cores - 1) // num_cores

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        ntiles = chunk
        for tid in range(ntiles):
            out_idx = tile_start + tid
            # Clamp to last valid tile for excess cores
            clamped = out_idx % num_output_tiles
            m = clamped // Nt
            n = clamped % Nt
            with b_dfb.reserve() as b_blk:
                tx = ttl.copy(bias[0, n], b_blk)
                tx.wait()
            for k in range(Kt):
                with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                    tx_x = ttl.copy(X[m, k], x_blk)
                    tx_x.wait()
                    tx_w = ttl.copy(W[k, n], w_blk)
                    tx_w.wait()

    @ttl.compute()
    def compute():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        ntiles = chunk
        for _ in range(ntiles):
            with b_dfb.wait() as b_blk:
                with acc_dfb.reserve() as acc:
                    acc.store(b_blk)
            for _ in range(Kt):
                with (
                    x_dfb.wait() as x_blk,
                    w_dfb.wait() as w_blk,
                    acc_dfb.wait() as prev,
                ):
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + x_blk @ w_blk)
            with acc_dfb.wait() as result:
                with y_dfb.reserve() as y_blk:
                    y_blk.store(result)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        ntiles = chunk
        for tid in range(ntiles):
            out_idx = tile_start + tid
            # Clamp to last valid tile for excess cores
            clamped = out_idx % num_output_tiles
            m = clamped // Nt
            n = clamped % Nt
            with y_dfb.wait() as y_blk:
                tx = ttl.copy(y_blk, Y[m, n])
                tx.wait()


# =========================================================================
# K-split down_proj: parallelize across K dimension
# =========================================================================

DOWN_K_SPLITS = 4


@ttl.kernel(grid=(GRID_Y, GRID_X))
def down_proj_partial_kernel(X, W, partial_out):
    """Partial matmul with K-dimension splitting.

    Each work item computes a partial sum over Kt_per_split K tiles for
    one output tile. With K_SPLITS=4 and Nt=28, there are 112 work items
    distributed across 110 cores.

    X:           [Mt, Kt] — input (Mt=1 for decode)
    W:           [Kt, Nt] — down_proj weight
    partial_out: [Mt, Nt * K_SPLITS * TILE] — partial sums
    """
    Kt = X.shape[1] // TILE
    Nt = W.shape[1] // TILE
    Kt_per_split = (Kt + DOWN_K_SPLITS - 1) // DOWN_K_SPLITS
    num_items = Nt * DOWN_K_SPLITS

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(W, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(partial_out, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(partial_out, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (num_items + num_cores - 1) // num_cores

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        for item_idx in range(chunk):
            item = (tile_start + item_idx) % num_items
            n = item // DOWN_K_SPLITS
            ks = item % DOWN_K_SPLITS
            k_begin = ks * Kt_per_split
            for ki in range(Kt_per_split):
                k = k_begin + ki
                # Clamp k to valid range (last split may be shorter)
                k_clamped = k % Kt
                with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                    tx_x = ttl.copy(X[0, k_clamped], x_blk)
                    tx_x.wait()
                    tx_w = ttl.copy(W[k_clamped, n], w_blk)
                    tx_w.wait()

    @ttl.compute()
    def compute():
        for _ in range(chunk):
            # First K tile: init accumulator
            with x_dfb.wait() as x_blk, w_dfb.wait() as w_blk:
                with acc_dfb.reserve() as acc:
                    acc.store(x_blk @ w_blk)
            # Remaining K tiles: accumulate
            for _ in range(Kt_per_split - 1):
                with (
                    x_dfb.wait() as x_blk,
                    w_dfb.wait() as w_blk,
                    acc_dfb.wait() as prev,
                ):
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + x_blk @ w_blk)
            # Move to output CB
            with acc_dfb.wait() as result:
                with y_dfb.reserve() as y_blk:
                    y_blk.store(result)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        for item_idx in range(chunk):
            item = (tile_start + item_idx) % num_items
            with y_dfb.wait() as y_blk:
                tx = ttl.copy(y_blk, partial_out[0, item])
                tx.wait()


@ttl.kernel(grid=(7, 4))
def down_proj_reduce_kernel(partial_out, Y):
    """Sum K_SPLITS partial tiles per output tile.

    partial_out: [Mt, Nt * K_SPLITS * TILE] — from partial kernel
    Y:           [Mt, Nt * TILE]            — final output
    """
    Nt = Y.shape[1] // TILE

    p_dfb = ttl.make_dataflow_buffer_like(partial_out, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (Nt + num_cores - 1) // num_cores

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        for tid in range(chunk):
            n = (tile_start + tid) % Nt
            for ks in range(DOWN_K_SPLITS):
                with p_dfb.reserve() as blk:
                    tx = ttl.copy(partial_out[0, n * DOWN_K_SPLITS + ks], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(chunk):
            # First partial: init accumulator
            with p_dfb.wait() as p_blk:
                with acc_dfb.reserve() as acc:
                    acc.store(p_blk)
            # Remaining partials: add
            for _ in range(DOWN_K_SPLITS - 1):
                with p_dfb.wait() as p_blk, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(prev + p_blk)
            # Move to output
            with acc_dfb.wait() as result:
                with y_dfb.reserve() as y_blk:
                    y_blk.store(result)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        for tid in range(chunk):
            n = (tile_start + tid) % Nt
            with y_dfb.wait() as y_blk:
                tx = ttl.copy(y_blk, Y[0, n])
                tx.wait()


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_linear(device):
    M, K, N = 512, 896, 128
    print(f"  linear [{M}x{K}] @ [{K}x{N}] (11x10 grid)...", end="", flush=True)

    X_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    W_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    X = _to_device(X_t, device)
    W = _to_device(W_t, device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    linear_kernel(X, W, Y)

    result = ttnn.to_torch(Y)
    expected = (X_t.float() @ W_t.float()).bfloat16()
    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


def test_linear_bias(device):
    M, K, N = 512, 896, 896
    print(f"  linear_bias [{M}x{K}] @ [{K}x{N}] + bias (11x10 grid)...", end="", flush=True)

    X_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    W_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    bias_flat = torch.randn(N, dtype=torch.bfloat16) * 0.01
    bias_tiled = bias_flat.unsqueeze(0).expand(TILE, -1).contiguous()

    X = _to_device(X_t, device)
    W = _to_device(W_t, device)
    bias = _to_device(bias_tiled, device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    linear_bias_kernel(X, W, bias, Y)

    result = ttnn.to_torch(Y)
    expected = (X_t.float() @ W_t.float() + bias_flat.float().unsqueeze(0)).bfloat16()
    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


def test_down_proj_ksplit(device):
    """Test K-split down_proj: partial + reduce matches linear_kernel."""
    M, K, N = 32, 4864, 896
    print(f"  down_proj K-split [{M}x{K}] @ [{K}x{N}] (partial+reduce)...", end="", flush=True)

    X_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    W_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

    X = _to_device(X_t, device)
    W = _to_device(W_t, device)
    Nt = N // TILE
    partial = _to_device(
        torch.zeros(M, Nt * DOWN_K_SPLITS * TILE, dtype=torch.bfloat16), device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    down_proj_partial_kernel(X, W, partial)
    down_proj_reduce_kernel(partial, Y)

    result = ttnn.to_torch(Y)
    expected = (X_t.float() @ W_t.float()).bfloat16()
    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Linear layer tests (multi-core):")
        test_linear(device)
        test_linear_bias(device)
        test_down_proj_ksplit(device)
        print("All linear tests passed!")
    finally:
        ttnn.close_device(device)
