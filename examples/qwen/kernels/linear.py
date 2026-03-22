# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Linear layer kernels for Qwen 2.5 0.5B.

Two variants:
  - linear_kernel: Y = X @ W (no bias) — for O/gate/up/down projections
  - linear_bias_kernel: Y = X @ W + bias — for Q/K/V projections

Uses the confirmed `prev + a @ b` fused matmul pattern with K-accumulation.
Block size: 1x1 tiles for simplicity. Output tiles computed one at a time.

Pattern from matmul_acc.py:
  1. DM reads bias (or zero-init) into acc
  2. K-loop: compute does prev + a @ b
  3. DM writes result
"""

import torch
import ttl
import ttnn

TILE = 32


@ttl.kernel(grid=(1, 1))
def linear_kernel(X, W, Y):
    """Y = X @ W. No bias.

    X: [Mt, Kt] tiles
    W: [Kt, Nt] tiles
    Y: [Mt, Nt] tiles
    """
    Mt = X.shape[0] // TILE
    Kt = X.shape[1] // TILE
    Nt = W.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(W, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                        tx_x = ttl.copy(X[m, k], x_blk)
                        tx_x.wait()
                        tx_w = ttl.copy(W[k, n], w_blk)
                        tx_w.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                # First K step: just the matmul result
                with x_dfb.wait() as x_blk, w_dfb.wait() as w_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(x_blk @ w_blk)

                # Remaining K steps: accumulate prev + x @ w
                for _ in range(Kt - 1):
                    with (
                        x_dfb.wait() as x_blk,
                        w_dfb.wait() as w_blk,
                        acc_dfb.wait() as prev,
                    ):
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + x_blk @ w_blk)

                # Write final accumulated result
                with acc_dfb.wait() as result:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(result)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, Y[m, n])
                    tx.wait()


@ttl.kernel(grid=(1, 1))
def linear_bias_kernel(X, W, bias, Y):
    """Y = X @ W + bias (broadcast along rows).

    X: [Mt, Kt] tiles
    W: [Kt, Nt] tiles
    bias: [1, Nt] tiles (each tile has bias values in row 0, to be row-broadcast)
    Y: [Mt, Nt] tiles
    """
    Mt = X.shape[0] // TILE
    Kt = X.shape[1] // TILE
    Nt = W.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(X, shape=(1, 1), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(W, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                # Load bias tile for this output column
                with b_dfb.reserve() as b_blk:
                    tx = ttl.copy(bias[0, n], b_blk)
                    tx.wait()
                # Stream K pairs
                for k in range(Kt):
                    with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                        tx_x = ttl.copy(X[m, k], x_blk)
                        tx_x.wait()
                        tx_w = ttl.copy(W[k, n], w_blk)
                        tx_w.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                # Initialize accumulator with bias
                with b_dfb.wait() as b_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(b_blk)

                # K-loop: accumulate prev + x @ w
                for _ in range(Kt):
                    with (
                        x_dfb.wait() as x_blk,
                        w_dfb.wait() as w_blk,
                        acc_dfb.wait() as prev,
                    ):
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + x_blk @ w_blk)

                # Write result
                with acc_dfb.wait() as result:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(result)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, Y[m, n])
                    tx.wait()


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_linear(device):
    """Test Y = X @ W with K-accumulation."""
    M, K, N = 512, 896, 128  # Mimics seq×hidden → seq×kv_dim
    print(f"  linear [{M}x{K}] @ [{K}x{N}]...", end="", flush=True)

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
    """Test Y = X @ W + bias."""
    M, K, N = 512, 896, 896  # Mimics Q projection
    print(f"  linear_bias [{M}x{K}] @ [{K}x{N}] + bias...", end="", flush=True)

    X_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
    W_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

    # Bias: [1_tile_row, Nt] — each tile filled with the same bias value
    # In practice, bias is [896] → reshaped to [1, 896] → padded to [32, 896]
    # Each 32x32 tile has the bias values replicated across all 32 rows
    bias_flat = torch.randn(N, dtype=torch.bfloat16) * 0.01
    bias_tiled = bias_flat.unsqueeze(0).expand(TILE, -1).contiguous()  # [32, 896]

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


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Linear layer tests:")
        test_linear(device)
        test_linear_bias(device)
        print("All linear tests passed!")
    finally:
        ttnn.close_device(device)
