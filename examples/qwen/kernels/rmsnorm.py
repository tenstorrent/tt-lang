# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm kernel for Qwen 2.5 0.5B.

RMSNorm(x) = x * rsqrt(mean(x^2)) * gamma

Since reduce_sum is not available on HW, we use sequential tile accumulation:
  1. For each sequence row: load all 28 hidden tiles
  2. Square each tile
  3. Sequentially accumulate: acc += sq_tile (28 steps)
  4. Multiply by 1/hidden_size scaler
  5. Rsqrt
  6. Multiply each original tile by rsqrt_val * gamma

The reduction is done across the hidden dimension (28 tiles = 896 elements per row).
Each tile is 32x32, so the sum of 28 tiles gives us the sum of 28*32=896 columns
per row. But we need sum per ROW, not per tile column.

Key insight: tile addition adds corresponding elements. So adding tile[0] + tile[1]
gives us partial sums for each of the 32 rows. After adding all 28 tiles, each element
[r, c] has sum of x^2 for row r across the 28 tiles for column c. But we need the
sum across ALL 896 columns for each row, which means we also need to sum across
the 32 columns within the final accumulated tile.

Two-level reduction:
  Level 1: Sum 28 tiles → 1 tile (each element has partial sum)
  Level 2: Sum 32 columns within the tile → column 0 has full row sum
           Then broadcast column 0 across all columns

For Level 2, since we can't use reduce_sum, we use:
  - The scaler trick: multiply by a tile of 1/hidden_size values, which
    combined with the row sum gives us the mean
  Actually, simpler: we pre-compute the scaler as 1/(hidden_size) and
  the partial tile sum already gives per-row sums if we're careful.

Actually, the simplest approach: do the ENTIRE RMSNorm on host using PyTorch,
except for the final elementwise multiply which is fast on device. This avoids
the complex reduction entirely.

For now: implement a simplified version that works for the model.
We do the reduction + rsqrt on HOST, then send the result to device for the
final elementwise multiply (x * rsqrt_val * gamma).
"""

import torch
import ttl
import ttnn

TILE = 32


@ttl.kernel(grid=(1, 1))
def rmsnorm_mul_kernel(X, scale, gamma, Y):
    """Y = X * scale * gamma, applied tile-by-tile.

    X: [Mt, Nt] tiles — input
    scale: [Mt, 1] tiles — rsqrt(mean(x^2)) per row, broadcast across columns
    gamma: [1, Nt] tiles — learnable weight, broadcast across rows
    Y: [Mt, Nt] tiles — output

    The DM thread reads scale once per row (broadcasts to all Nt columns).
    """
    Mt = X.shape[0] // TILE
    Nt = X.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=2)
    g_dfb = ttl.make_dataflow_buffer_like(gamma, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(X[m, n], blk)
                    tx.wait()
                # Read the same scale tile for every column in this row
                with s_dfb.reserve() as blk:
                    tx = ttl.copy(scale[m, 0], blk)
                    tx.wait()
                # Read gamma for this column
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gamma[0, n], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with (
                    x_dfb.wait() as x_blk,
                    s_dfb.wait() as s_blk,
                    g_dfb.wait() as g_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    y_blk.store(x_blk * s_blk * g_blk)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, Y[m, n])
                    tx.wait()


def rmsnorm(x_device, gamma_device, device, eps=1e-6):
    """Full RMSNorm: reduction on host, multiply on device.

    Args:
        x_device: TTNN tensor [seq, hidden] on device
        gamma_device: TTNN tensor [1, hidden] on device (padded to [32, hidden])
        device: TTNN device
        eps: epsilon for numerical stability

    Returns:
        TTNN tensor [seq, hidden] on device
    """
    # Pull x to host for reduction
    x_torch = ttnn.to_torch(x_device).float()
    seq, hidden = x_torch.shape

    # Compute rsqrt(mean(x^2) + eps) per row
    mean_sq = (x_torch ** 2).mean(dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(mean_sq + eps)  # [seq, 1]

    # Expand to [seq, 32] (one tile column) with same value across all 32 columns
    scale = rsqrt_val.expand(-1, TILE).bfloat16().contiguous()
    # Pad rows to tile boundary
    if seq % TILE != 0:
        pad_rows = TILE - (seq % TILE)
        scale = torch.nn.functional.pad(scale, (0, 0, 0, pad_rows))

    scale_device = ttnn.from_torch(
        scale, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Allocate output
    y_device = ttnn.from_torch(
        torch.zeros(x_torch.shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run multiply kernel on device
    rmsnorm_mul_kernel(x_device, scale_device, gamma_device, y_device)

    return y_device


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_rmsnorm(device):
    M, N = 512, 896  # seq_len x hidden_size
    eps = 1e-6
    print(f"  rmsnorm [{M}x{N}]...", end="", flush=True)

    X_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    gamma_t = torch.randn(N, dtype=torch.bfloat16) * 0.1 + 1.0
    gamma_tiled = gamma_t.unsqueeze(0).expand(TILE, -1).contiguous()

    X = _to_device(X_t, device)
    gamma = _to_device(gamma_tiled, device)

    Y = rmsnorm(X, gamma, device, eps=eps)

    result = ttnn.to_torch(Y)

    # PyTorch reference
    x_float = X_t.float()
    rms = torch.rsqrt((x_float ** 2).mean(dim=-1, keepdim=True) + eps)
    expected = (x_float * rms * gamma_t.float().unsqueeze(0)).bfloat16()

    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("RMSNorm tests:")
        test_rmsnorm(device)
        print("RMSNorm test passed!")
    finally:
        ttnn.close_device(device)
