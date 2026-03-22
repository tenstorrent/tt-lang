# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-side RMSNorm for Qwen 2.5 0.5B.

RMSNorm(x) = x * rsqrt(mean(x^2)) * gamma

Two-kernel approach:
  1. reduce_norm_kernel: square → reduce_sum(dims=[1], scaler=1/N) → accumulate
     across tiles → rsqrt → scalar broadcast → output scale tile
  2. rmsnorm_mul_kernel: x * scale * gamma (multi-core, 110 cores)

The reduce scaler is 1/hidden_size, so reduce_sum directly gives mean(x^2).
"""

import torch
import ttl
import ttnn

TILE = 32
GRID_Y = 11
GRID_X = 10


@ttl.kernel(grid=(1, 1))
def reduce_norm_kernel(X, mean_scaler, scale_out):
    """Compute rsqrt(mean(x^2)) → scalar-broadcast tile.

    X: [Mt, Nt] tiles
    mean_scaler: [1, 1] tile filled with 1/hidden_size
    scale_out: [Mt, 1] tile — output norm scale (same value everywhere)
    """
    Mt = X.shape[0] // TILE
    Nt = X.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(mean_scaler, shape=(1, 1), buffer_factor=2)
    # Compute-local DFBs
    sq_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scale_out, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scale_out, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(scale_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(mean_scaler[0, 0], blk)
                tx.wait()
            for col in range(Nt):
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(X[row, col], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            with sc_dfb.wait() as sc_blk:
                # First column tile
                with x_dfb.wait() as x_blk:
                    with sq_dfb.reserve() as sq:
                        sq.store(x_blk * x_blk)
                with sq_dfb.wait() as sq_blk:
                    with red_dfb.reserve() as rd:
                        rd.store(ttl.math.reduce_sum(sq_blk, sc_blk, rd, dims=[1]))
                with red_dfb.wait() as reduced:
                    with acc_dfb.reserve() as acc:
                        acc.store(reduced)

                # Remaining column tiles
                for _ in range(Nt - 1):
                    with x_dfb.wait() as x_blk:
                        with sq_dfb.reserve() as sq:
                            sq.store(x_blk * x_blk)
                    with sq_dfb.wait() as sq_blk:
                        with red_dfb.reserve() as rd:
                            rd.store(ttl.math.reduce_sum(sq_blk, sc_blk, rd, dims=[1]))
                    with red_dfb.wait() as reduced, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + reduced)

            # acc[0,0] = mean(x^2) for row 0
            # rsqrt then scalar broadcast
            with acc_dfb.wait() as mean_sq:
                with red_dfb.reserve() as rsqrt_tile:
                    rsqrt_tile.store(ttl.math.rsqrt(mean_sq))
            with red_dfb.wait() as rsqrt_blk:
                with out_dfb.reserve() as out:
                    out.store(ttl.math.broadcast(rsqrt_blk, out, dims=[0, 1]))

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, scale_out[row, 0])
                tx.wait()


@ttl.kernel(grid=(GRID_Y, GRID_X))
def rmsnorm_mul_kernel(X, scale, gamma, Y):
    """Y = X * scale * gamma, multi-core.

    X: [Mt, Nt] tiles
    scale: [Mt, 1] tiles — rsqrt(mean(x^2)) broadcast scalar
    gamma: [1, Nt] tiles — learnable weight
    Y: [Mt, Nt] tiles
    """
    Mt = X.shape[0] // TILE
    Nt = X.shape[1] // TILE
    num_tiles = Mt * Nt

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (num_tiles + num_cores - 1) // num_cores

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        for tid in range(chunk):
            idx = (tile_start + tid) % num_tiles
            m = idx // Nt
            n = idx % Nt
            with x_dfb.reserve() as blk:
                tx = ttl.copy(X[m, n], blk)
                tx.wait()
            with s_dfb.reserve() as blk:
                tx = ttl.copy(scale[m, 0], blk)
                tx.wait()
            with g_dfb.reserve() as blk:
                tx = ttl.copy(gamma[0, n], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(chunk):
            with (
                x_dfb.wait() as x_blk,
                s_dfb.wait() as s_blk,
                g_dfb.wait() as g_blk,
                y_dfb.reserve() as y_blk,
            ):
                y_blk.store(x_blk * s_blk * g_blk)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk
        for tid in range(chunk):
            idx = (tile_start + tid) % num_tiles
            m = idx // Nt
            n = idx % Nt
            with y_dfb.wait() as blk:
                tx = ttl.copy(blk, Y[m, n])
                tx.wait()


@ttl.kernel(grid=(1, 1))
def fused_rmsnorm_kernel(X, mean_scaler, gamma, Y):
    """Full RMSNorm in one kernel: reduce → rsqrt → broadcast → multiply.

    Pass 1 (DM reads X): square, reduce_sum, cross-tile accumulate, rsqrt, broadcast
    Pass 2 (DM re-reads X + gamma): Y = X * scale * gamma

    X: [Mt, Nt] tiles
    mean_scaler: [1, 1] tile (1/hidden_size)
    gamma: [1, Nt] tiles (learnable weight, replicated across rows)
    Y: [Mt, Nt] tiles
    """
    Mt = X.shape[0] // TILE
    Nt = X.shape[1] // TILE

    # DFBs for both passes (shared DM reader)
    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(mean_scaler, shape=(1, 1), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    # Compute-local
    sq_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(mean_scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(mean_scaler, shape=(1, 1), buffer_factor=2)
    scale_dfb = ttl.make_dataflow_buffer_like(mean_scaler, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            # Pass 1: read scaler + X tiles for reduction
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(mean_scaler[0, 0], blk)
                tx.wait()
            for col in range(Nt):
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(X[row, col], blk)
                    tx.wait()
            # Pass 2: re-read X + gamma for multiply
            for col in range(Nt):
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(X[row, col], blk)
                    tx.wait()
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gamma[0, col], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            # Pass 1: reduce
            with sc_dfb.wait() as sc_blk:
                with x_dfb.wait() as x_blk:
                    with sq_dfb.reserve() as sq:
                        sq.store(x_blk * x_blk)
                with sq_dfb.wait() as sq_blk:
                    with red_dfb.reserve() as rd:
                        rd.store(ttl.math.reduce_sum(sq_blk, sc_blk, rd, dims=[1]))
                with red_dfb.wait() as reduced:
                    with acc_dfb.reserve() as acc:
                        acc.store(reduced)

                for _ in range(Nt - 1):
                    with x_dfb.wait() as x_blk:
                        with sq_dfb.reserve() as sq:
                            sq.store(x_blk * x_blk)
                    with sq_dfb.wait() as sq_blk:
                        with red_dfb.reserve() as rd:
                            rd.store(ttl.math.reduce_sum(sq_blk, sc_blk, rd, dims=[1]))
                    with red_dfb.wait() as reduced, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + reduced)

            # rsqrt + broadcast → scale stays in compute-local DFB
            with acc_dfb.wait() as mean_sq:
                with red_dfb.reserve() as rsqrt_tile:
                    rsqrt_tile.store(ttl.math.rsqrt(mean_sq))
            with red_dfb.wait() as rsqrt_blk:
                with scale_dfb.reserve() as sc:
                    sc.store(ttl.math.broadcast(rsqrt_blk, sc, dims=[0, 1]))

            # Pass 2: multiply X * scale * gamma
            with scale_dfb.wait() as scale_blk:
                for _ in range(Nt):
                    with x_dfb.wait() as x_blk, g_dfb.wait() as g_blk:
                        with y_dfb.reserve() as out:
                            out.store(x_blk * scale_blk * g_blk)

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            for col in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, Y[row, col])
                    tx.wait()


def fused_device_rmsnorm(x_device, gamma_device, mean_scaler_device, device):
    """Full device-side RMSNorm in a single kernel call."""
    rows = x_device.shape[0]
    cols = x_device.shape[1]
    y = ttnn.from_torch(
        torch.zeros(rows, cols, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fused_rmsnorm_kernel(x_device, mean_scaler_device, gamma_device, y)
    return y


def device_rmsnorm(x_device, gamma_device, mean_scaler_device, device):
    """Full device-side RMSNorm. No host transfers.

    Args:
        x_device: [seq, hidden] on device
        gamma_device: [1, hidden] tiles on device (gamma replicated across rows)
        mean_scaler_device: [32, 32] tile filled with 1/hidden_size, on device
        device: TTNN device

    Returns:
        y_device: [seq, hidden] on device
    """
    rows = x_device.shape[0]
    cols = x_device.shape[1]

    def alloc(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    scale = alloc((rows, TILE))
    y = alloc((rows, cols))

    reduce_norm_kernel(x_device, mean_scaler_device, scale)
    rmsnorm_mul_kernel(x_device, scale, gamma_device, y)

    return y


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_rmsnorm(device):
    M, N = 32, 896  # decode size (1 tile row)
    eps = 1e-6
    print(f"  rmsnorm [{M}x{N}] (device-only)...", end="", flush=True)

    X_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    gamma_t = torch.randn(N, dtype=torch.bfloat16) * 0.1 + 1.0
    gamma_tiled = gamma_t.unsqueeze(0).expand(TILE, -1).contiguous()

    X = _to_device(X_t, device)
    gamma = _to_device(gamma_tiled, device)
    mean_scaler = _to_device(
        torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16), device
    )

    Y = device_rmsnorm(X, gamma, mean_scaler, device)
    result = ttnn.to_torch(Y)

    # PyTorch reference
    x_float = X_t.float()
    rms = torch.rsqrt((x_float ** 2).mean(dim=-1, keepdim=True) + eps)
    expected = (x_float * rms * gamma_t.float().unsqueeze(0)).bfloat16()

    score = torch.corrcoef(
        torch.stack([result[0].float(), expected[0].float()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.98, f" FAIL"
    print(" PASS")


def test_rmsnorm_prefill(device):
    M, N = 512, 896
    eps = 1e-6
    print(f"  rmsnorm [{M}x{N}] (prefill, device)...", end="", flush=True)

    X_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    gamma_t = torch.randn(N, dtype=torch.bfloat16) * 0.1 + 1.0
    gamma_tiled = gamma_t.unsqueeze(0).expand(TILE, -1).contiguous()

    X = _to_device(X_t, device)
    gamma = _to_device(gamma_tiled, device)
    mean_scaler = _to_device(
        torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16), device
    )

    Y = device_rmsnorm(X, gamma, mean_scaler, device)
    result = ttnn.to_torch(Y)

    x_float = X_t.float()
    rms = torch.rsqrt((x_float ** 2).mean(dim=-1, keepdim=True) + eps)
    expected = (x_float * rms * gamma_t.float().unsqueeze(0)).bfloat16()

    # Check row 0 (most important for correctness)
    score = torch.corrcoef(
        torch.stack([result[0].float(), expected[0].float()])
    )[0, 1].item()
    print(f" PCC={score:.6f} (row 0)", end="")
    assert score > 0.98, f" FAIL"
    print(" PASS")


def test_fused_rmsnorm(device):
    M, N = 32, 896
    eps = 1e-6
    print(f"  fused_rmsnorm [{M}x{N}]...", end="", flush=True)

    X_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    gamma_t = torch.randn(N, dtype=torch.bfloat16) * 0.1 + 1.0
    gamma_tiled = gamma_t.unsqueeze(0).expand(TILE, -1).contiguous()

    X = _to_device(X_t, device)
    gamma = _to_device(gamma_tiled, device)
    mean_scaler = _to_device(
        torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16), device
    )

    Y = fused_device_rmsnorm(X, gamma, mean_scaler, device)
    result = ttnn.to_torch(Y)

    x_float = X_t.float()
    rms = torch.rsqrt((x_float ** 2).mean(dim=-1, keepdim=True) + eps)
    expected = (x_float * rms * gamma_t.float().unsqueeze(0)).bfloat16()

    score = torch.corrcoef(
        torch.stack([result[0].float(), expected[0].float()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.98, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("RMSNorm tests:")
        test_rmsnorm(device)
        test_fused_rmsnorm(device)
        print("All RMSNorm tests passed!")
    finally:
        ttnn.close_device(device)
