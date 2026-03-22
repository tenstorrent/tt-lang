# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Elementwise kernels for Qwen 2.5 0.5B.

- add_kernel: Y = A + B (residual connections)
- silu_mul_kernel: Y = silu(gate) * up = gate * sigmoid(gate) * up (SwiGLU)
"""

import torch
import ttl
import ttnn

TILE = 32


@ttl.kernel(grid=(1, 1))
def add_kernel(A, B, Y):
    """Y = A + B elementwise."""
    Mt = A.shape[0] // TILE
    Nt = A.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(A[m, n], blk)
                    tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(B[m, n], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, Y[m, n])
                    tx.wait()


@ttl.kernel(grid=(1, 1))
def silu_mul_kernel(gate, up, Y):
    """Y = silu(gate) * up = gate * sigmoid(gate) * up.

    All tensors same shape. Used in SwiGLU MLP.
    """
    Mt = gate.shape[0] // TILE
    Nt = gate.shape[1] // TILE

    g_dfb = ttl.make_dataflow_buffer_like(gate, shape=(1, 1), buffer_factor=2)
    u_dfb = ttl.make_dataflow_buffer_like(up, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for m in range(Mt):
            for n in range(Nt):
                with g_dfb.reserve() as blk:
                    tx = ttl.copy(gate[m, n], blk)
                    tx.wait()
                with u_dfb.reserve() as blk:
                    tx = ttl.copy(up[m, n], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with g_dfb.wait() as g_blk, u_dfb.wait() as u_blk:
                    with y_dfb.reserve() as y_blk:
                        # silu(gate) * up = gate * sigmoid(gate) * up
                        y_blk.store(g_blk * ttl.math.sigmoid(g_blk) * u_blk)

    @ttl.datamovement()
    def write():
        for m in range(Mt):
            for n in range(Nt):
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
    A_t = torch.randn(M, N, dtype=torch.bfloat16)
    B_t = torch.randn(M, N, dtype=torch.bfloat16)
    A = _to_device(A_t, device)
    B = _to_device(B_t, device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    add_kernel(A, B, Y)

    result = ttnn.to_torch(Y)
    expected = (A_t.float() + B_t.float()).bfloat16()
    score = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"  add [{M}x{N}]: PCC={score:.6f}", end="")
    assert score > 0.999, f" FAIL"
    print(" PASS")


def test_silu_mul(device):
    M, N = 512, 4864
    gate_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    up_t = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    gate = _to_device(gate_t, device)
    up = _to_device(up_t, device)
    Y = _to_device(torch.zeros(M, N, dtype=torch.bfloat16), device)

    silu_mul_kernel(gate, up, Y)

    result = ttnn.to_torch(Y)
    expected = (torch.nn.functional.silu(gate_t.float()) * up_t.float()).bfloat16()
    score = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"  silu_mul [{M}x{N}]: PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Elementwise tests:")
        test_add(device)
        test_silu_mul(device)
        print("All elementwise tests passed!")
    finally:
        ttnn.close_device(device)
