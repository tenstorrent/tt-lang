# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-side RoPE (Rotary Position Embeddings) for Qwen 2.5.

Two variants:
  - rope_kernel: single head pair [Mt, 2] tiles
  - batch_rope_kernel: all head pairs in a combined tensor [Mt, N] tiles

Key insight: head_dim=64, head_dim/2=32=TILE_SIZE.
Each head's data is exactly 2 tiles: tile0 = x[..., :32], tile1 = x[..., 32:64].

RoPE formula:
  out_tile0 = x_tile0 * cos_tile0 - x_tile1 * sin_tile0
  out_tile1 = x_tile1 * cos_tile1 + x_tile0 * sin_tile1
"""

import torch
import ttl
import ttnn

TILE = 32


@ttl.kernel(grid=(1, 1))
def rope_kernel(x_in, cos_table, sin_table, x_out):
    """Apply RoPE to a single head: x_in [Mt, 2] tiles."""
    Mt = x_in.shape[0] // TILE

    x0_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), buffer_factor=2)
    x1_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), buffer_factor=2)
    c0_dfb = ttl.make_dataflow_buffer_like(cos_table, shape=(1, 1), buffer_factor=2)
    c1_dfb = ttl.make_dataflow_buffer_like(cos_table, shape=(1, 1), buffer_factor=2)
    s0_dfb = ttl.make_dataflow_buffer_like(sin_table, shape=(1, 1), buffer_factor=2)
    s1_dfb = ttl.make_dataflow_buffer_like(sin_table, shape=(1, 1), buffer_factor=2)
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
                x0_dfb.wait() as x0, x1_dfb.wait() as x1,
                c0_dfb.wait() as c0, c1_dfb.wait() as c1,
                s0_dfb.wait() as s0, s1_dfb.wait() as s1,
            ):
                with o0_dfb.reserve() as out0:
                    out0.store(x0 * c0 - x1 * s0)
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


@ttl.kernel(grid=(1, 1))
def batch_rope_kernel(x_in, cos_table, sin_table, x_out):
    """Apply RoPE to ALL head pairs in a combined tensor.

    x_in: [Mt, Nt] tiles where Nt = num_heads * 2 (e.g., 28 for 14 Q heads)
    cos_table: [Mt, 2] tiles — same cos for all heads at same position
    sin_table: [Mt, 2] tiles — same sin for all heads
    x_out: [Mt, Nt] tiles — output

    Processes Nt/2 head pairs per row. Reads cos/sin once per row, reuses
    for all pairs. Uses runtime column offsets: x_in[row, pair*2] and x_in[row, pair*2+1].
    """
    Mt = x_in.shape[0] // TILE
    Nt = x_in.shape[1] // TILE
    num_pairs = Nt // 2

    x0_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), buffer_factor=2)
    x1_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), buffer_factor=2)
    c0_dfb = ttl.make_dataflow_buffer_like(cos_table, shape=(1, 1), buffer_factor=2)
    c1_dfb = ttl.make_dataflow_buffer_like(cos_table, shape=(1, 1), buffer_factor=2)
    s0_dfb = ttl.make_dataflow_buffer_like(sin_table, shape=(1, 1), buffer_factor=2)
    s1_dfb = ttl.make_dataflow_buffer_like(sin_table, shape=(1, 1), buffer_factor=2)
    o0_dfb = ttl.make_dataflow_buffer_like(x_out, shape=(1, 1), buffer_factor=2)
    o1_dfb = ttl.make_dataflow_buffer_like(x_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            # Read cos/sin once per row
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
            # Read all head pairs
            for pair in range(num_pairs):
                col0 = pair * 2
                col1 = pair * 2 + 1
                with x0_dfb.reserve() as blk:
                    tx = ttl.copy(x_in[row, col0], blk)
                    tx.wait()
                with x1_dfb.reserve() as blk:
                    tx = ttl.copy(x_in[row, col1], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            # cos/sin consumed once, kept alive for all pairs
            with (
                c0_dfb.wait() as c0, c1_dfb.wait() as c1,
                s0_dfb.wait() as s0, s1_dfb.wait() as s1,
            ):
                for _ in range(num_pairs):
                    with x0_dfb.wait() as x0, x1_dfb.wait() as x1:
                        with o0_dfb.reserve() as out0:
                            out0.store(x0 * c0 - x1 * s0)
                        with o1_dfb.reserve() as out1:
                            out1.store(x1 * c1 + x0 * s1)

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            for pair in range(num_pairs):
                col0 = pair * 2
                col1 = pair * 2 + 1
                with o0_dfb.wait() as blk:
                    tx = ttl.copy(blk, x_out[row, col0])
                    tx.wait()
                with o1_dfb.wait() as blk:
                    tx = ttl.copy(blk, x_out[row, col1])
                    tx.wait()


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_batch_rope(device):
    """Test batch RoPE on Q-combined [512, 896] = 14 head pairs."""
    seq_len, hidden = 512, 896
    num_heads = hidden // 64  # 14
    head_dim = 64
    print(f"  batch_rope [{seq_len}x{hidden}] ({num_heads} heads)...", end="", flush=True)

    x_t = torch.randn(seq_len, hidden, dtype=torch.bfloat16) * 0.1

    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_t = torch.cos(freqs).repeat(1, 2).bfloat16()
    sin_t = torch.sin(freqs).repeat(1, 2).bfloat16()

    x = _to_device(x_t, device)
    cos_dev = _to_device(cos_t, device)
    sin_dev = _to_device(sin_t, device)
    out = _to_device(torch.zeros(seq_len, hidden, dtype=torch.bfloat16), device)

    batch_rope_kernel(x, cos_dev, sin_dev, out)
    result = ttnn.to_torch(out)

    # Reference: apply RoPE to each head independently
    def rotate_half(t):
        t1 = t[..., :head_dim // 2]
        t2 = t[..., head_dim // 2:]
        return torch.cat((-t2, t1), dim=-1)

    x_heads = x_t.float().view(seq_len, num_heads, head_dim)
    cos_exp = cos_t.float().unsqueeze(1)
    sin_exp = sin_t.float().unsqueeze(1)
    expected = (x_heads * cos_exp + rotate_half(x_heads) * sin_exp).view(seq_len, hidden).bfloat16()

    score = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.99, f" FAIL"
    print(" PASS")


def test_batch_rope_k(device):
    """Test batch RoPE on K-combined [512, 128] = 2 head pairs."""
    seq_len, kv_dim = 512, 128
    print(f"  batch_rope [{seq_len}x{kv_dim}] (2 heads)...", end="", flush=True)

    x_t = torch.randn(seq_len, kv_dim, dtype=torch.bfloat16) * 0.1

    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_t = torch.cos(freqs).repeat(1, 2).bfloat16()
    sin_t = torch.sin(freqs).repeat(1, 2).bfloat16()

    x = _to_device(x_t, device)
    cos_dev = _to_device(cos_t, device)
    sin_dev = _to_device(sin_t, device)
    out = _to_device(torch.zeros(seq_len, kv_dim, dtype=torch.bfloat16), device)

    batch_rope_kernel(x, cos_dev, sin_dev, out)
    result = ttnn.to_torch(out)

    def rotate_half(t):
        t1 = t[..., :32]
        t2 = t[..., 32:]
        return torch.cat((-t2, t1), dim=-1)

    x_heads = x_t.float().view(seq_len, 2, 64)
    cos_exp = cos_t.float().unsqueeze(1)
    sin_exp = sin_t.float().unsqueeze(1)
    expected = (x_heads * cos_exp + rotate_half(x_heads) * sin_exp).view(seq_len, kv_dim).bfloat16()

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
        test_batch_rope(device)
        test_batch_rope_k(device)
        print("All RoPE tests passed!")
    finally:
        ttnn.close_device(device)
