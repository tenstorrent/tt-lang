# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-side RoPE (Rotary Position Embeddings) for Qwen 2.5.

Key insight: head_dim=64, head_dim/2=32=TILE_SIZE.
Each head's data is exactly 2 tiles: tile0 = x[..., :32], tile1 = x[..., 32:64].

RoPE formula (rotate_half variant):
  out[..., :32]  = x[..., :32] * cos[..., :32] - x[..., 32:64] * sin[..., :32]
  out[..., 32:64] = x[..., 32:64] * cos[..., 32:64] + x[..., :32] * sin[..., 32:64]

In tile terms:
  out_tile0 = x_tile0 * cos_tile0 - x_tile1 * sin_tile0
  out_tile1 = x_tile1 * cos_tile1 + x_tile0 * sin_tile1

All ops are elementwise mul/add/sub — fully HW supported.
"""

import torch
import ttl
import ttnn

TILE = 32


@ttl.kernel(grid=(1, 1))
def rope_kernel(x_in, cos_table, sin_table, x_out):
    """Apply RoPE to x_in, writing result to x_out.

    x_in: [Mt, 2] tiles — Mt rows of (tile0, tile1) pairs per head
    cos_table: [Mt, 2] tiles — cos values for each position
    sin_table: [Mt, 2] tiles — sin values for each position
    x_out: [Mt, 2] tiles — output with RoPE applied

    Processes one row at a time: reads both tiles of x_in for that row,
    both tiles of cos/sin, computes the rotation, writes both output tiles.
    """
    Mt = x_in.shape[0] // TILE

    # x tile0 and tile1
    x0_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), buffer_factor=2)
    x1_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), buffer_factor=2)
    # cos/sin tile0 and tile1
    c0_dfb = ttl.make_dataflow_buffer_like(cos_table, shape=(1, 1), buffer_factor=2)
    c1_dfb = ttl.make_dataflow_buffer_like(cos_table, shape=(1, 1), buffer_factor=2)
    s0_dfb = ttl.make_dataflow_buffer_like(sin_table, shape=(1, 1), buffer_factor=2)
    s1_dfb = ttl.make_dataflow_buffer_like(sin_table, shape=(1, 1), buffer_factor=2)
    # output tile0 and tile1
    o0_dfb = ttl.make_dataflow_buffer_like(x_out, shape=(1, 1), buffer_factor=2)
    o1_dfb = ttl.make_dataflow_buffer_like(x_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            with x0_dfb.reserve() as blk:
                tx = ttl.copy(x_in[row, 0], blk)
                tx.wait()
            with x1_dfb.reserve() as blk:
                tx = ttl.copy(x_in[row, 1], blk)
                tx.wait()
            with c0_dfb.reserve() as blk:
                tx = ttl.copy(cos_table[row, 0], blk)
                tx.wait()
            with c1_dfb.reserve() as blk:
                tx = ttl.copy(cos_table[row, 1], blk)
                tx.wait()
            with s0_dfb.reserve() as blk:
                tx = ttl.copy(sin_table[row, 0], blk)
                tx.wait()
            with s1_dfb.reserve() as blk:
                tx = ttl.copy(sin_table[row, 1], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            with (
                x0_dfb.wait() as x0,
                x1_dfb.wait() as x1,
                c0_dfb.wait() as c0,
                c1_dfb.wait() as c1,
                s0_dfb.wait() as s0,
                s1_dfb.wait() as s1,
            ):
                # out_tile0 = x0 * c0 - x1 * s0
                with o0_dfb.reserve() as out0:
                    out0.store(x0 * c0 - x1 * s0)
                # out_tile1 = x1 * c1 + x0 * s1
                with o1_dfb.reserve() as out1:
                    out1.store(x1 * c1 + x0 * s1)

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            with o0_dfb.wait() as blk:
                tx = ttl.copy(blk, x_out[row, 0])
                tx.wait()
            with o1_dfb.wait() as blk:
                tx = ttl.copy(blk, x_out[row, 1])
                tx.wait()


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_rope(device):
    """Test RoPE on a single head's Q values."""
    seq_len = 512
    head_dim = 64
    print(f"  rope [{seq_len}x{head_dim}]...", end="", flush=True)

    # Random Q values for one head
    x_t = torch.randn(seq_len, head_dim, dtype=torch.bfloat16) * 0.1

    # RoPE cos/sin tables
    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq, head_dim/2]
    cos_t = torch.cos(freqs).repeat(1, 2).bfloat16()  # [seq, head_dim]
    sin_t = torch.sin(freqs).repeat(1, 2).bfloat16()

    x = _to_device(x_t, device)
    cos_dev = _to_device(cos_t, device)
    sin_dev = _to_device(sin_t, device)
    out = _to_device(torch.zeros(seq_len, head_dim, dtype=torch.bfloat16), device)

    rope_kernel(x, cos_dev, sin_dev, out)

    result = ttnn.to_torch(out)

    # PyTorch reference
    def rotate_half(t):
        t1 = t[..., :head_dim // 2]
        t2 = t[..., head_dim // 2:]
        return torch.cat((-t2, t1), dim=-1)

    expected = (x_t.float() * cos_t.float() + rotate_half(x_t.float()) * sin_t.float()).bfloat16()

    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("RoPE tests:")
        test_rope(device)
        print("RoPE test passed!")
    finally:
        ttnn.close_device(device)
