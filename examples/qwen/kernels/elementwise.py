# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise kernels for Qwen 2.5 0.5B. Multi-core (11×10 = 110 cores).

- add_kernel: Y = A + B (residual connections)
- silu_mul_kernel: Y = silu(gate) * up = gate * sigmoid(gate) * up (SwiGLU)
"""

import torch
import ttl
import ttnn

TILE = 32
GRID_Y = 11
GRID_X = 10


@ttl.kernel(grid=(GRID_Y, GRID_X))
def add_kernel(A, B, Y):
    """Y = A + B elementwise. Multi-core."""
    Mt = A.shape[0] // TILE
    Nt = A.shape[1] // TILE
    num_tiles = Mt * Nt

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), buffer_factor=2)
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
            with a_dfb.reserve() as blk:
                tx = ttl.copy(A[m, n], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(B[m, n], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(chunk):
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                with y_dfb.reserve() as y_blk:
                    y_blk.store(a_blk + b_blk)

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


@ttl.kernel(grid=(GRID_Y, GRID_X))
def silu_mul_kernel(gate, up, Y):
    """Y = silu(gate) * up = gate * sigmoid(gate) * up. Multi-core."""
    Mt = gate.shape[0] // TILE
    Nt = gate.shape[1] // TILE
    num_tiles = Mt * Nt

    g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
    u_dfb = ttl.make_dataflow_buffer_like(up, shape=(1, 1), buffer_factor=2)
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
            with g_dfb.reserve() as blk:
                tx = ttl.copy(gate[m, n], blk)
                tx.wait()
            with u_dfb.reserve() as blk:
                tx = ttl.copy(up[m, n], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(chunk):
            with g_dfb.wait() as g_blk, u_dfb.wait() as u_blk:
                with y_dfb.reserve() as y_blk:
                    y_blk.store(g_blk * ttl.math.sigmoid(g_blk) * u_blk)

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


# =========================================================================
# Tests
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_add(device):
    M, N = 512, 896
    print(f"  add [{M}x{N}] (11x10)...", end="", flush=True)
    A_t = torch.randn(M, N, dtype=torch.bfloat16)
    B_t = torch.randn(M, N, dtype=torch.bfloat16)
    A = _to_device(A_t, device)
    B = _to_device(B_t, device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)
    add_kernel(A, B, Y)
    result = ttnn.to_torch(Y)
    expected = (A_t.float() + B_t.float()).bfloat16()
    score = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.999, f" FAIL"
    print(" PASS")


def test_silu_mul(device):
    M, N = 512, 4864
    print(f"  silu_mul [{M}x{N}] (11x10)...", end="", flush=True)
    gate_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    up_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    gate = _to_device(gate_t, device)
    up = _to_device(up_t, device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)
    silu_mul_kernel(gate, up, Y)
    result = ttnn.to_torch(Y)
    expected = (torch.nn.functional.silu(gate_t.float()) * up_t.float()).bfloat16()
    score = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Elementwise tests (multi-core):")
        test_add(device)
        test_silu_mul(device)
        print("All elementwise tests passed!")
    finally:
        ttnn.close_device(device)
