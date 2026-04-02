# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 Multi-Head Attention -- Multinode (1 head per core).

Distributes attention heads across Tensix cores:
  grid=(NUM_HEADS, 1) -- each core handles one head independently.

Config:
  hidden_dim = 128 (4 tiles)
  num_heads  = 4 (grid x-dim)
  head_dim   = 32 (1 tile)
  seq_len    = 64 (2 tiles) -- larger to show streaming

Each core:
  1. Reads its head's Q, K, V slices from DRAM (using node index for column offset)
  2. Computes attention: exp(Q @ K^T * scale) @ V
  3. Writes result back to the corresponding head slice

This is the natural parallelism for multi-head attention on Tenstorrent:
each head is completely independent, so each core does a full attention
on its own head with zero communication.
"""

import torch

import ttnn
import ttl

SEQ = 2          # 2 tiles = 64 tokens (larger to show multi-tile seq)
HIDDEN = 4       # 4 tiles = 128 dim
NUM_HEADS = 4
HEAD_DIM = 1     # 1 tile = 32 per head


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(NUM_HEADS, 1))
def multihead_attention_kernel(q, k, v, scale, out):
    """
    Multi-head attention with 1 head per core.
    Each core reads its head slice, computes attention, writes back.

    Q, K, V: (SEQ, HIDDEN) = (2, 4) tiles -- each head is (2, 1) slice
    scale: (1, 1) tile
    out: (SEQ, HIDDEN) = (2, 4) tiles
    """
    # Per-core DFBs: each core works with (SEQ, HEAD_DIM) blocks
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(SEQ, HEAD_DIM), buffer_factor=2)
    k_dfb = ttl.make_dataflow_buffer_like(k, shape=(SEQ, HEAD_DIM), buffer_factor=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(SEQ, HEAD_DIM), buffer_factor=2)
    # Scale as full (SEQ, SEQ) to avoid broadcast issues in sim
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(SEQ, SEQ), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ, HEAD_DIM), buffer_factor=2)

    # Intermediates
    kt_dfb = ttl.make_dataflow_buffer_like(k, shape=(HEAD_DIM, SEQ), buffer_factor=2)
    scores_dfb = ttl.make_dataflow_buffer_like(q, shape=(SEQ, SEQ), buffer_factor=2)

    @ttl.compute()
    def compute():
        with scale_dfb.wait() as sv:
            # Transpose K
            with k_dfb.wait() as kv, kt_dfb.reserve() as kt:
                kt.store(ttl.transpose(kv, kt))

            # Q @ K^T: (2,1) @ (1,2) = (2,2)
            with q_dfb.wait() as qv, kt_dfb.wait() as ktv:
                with scores_dfb.reserve() as sc:
                    sc.store(ttl.math.matmul(qv, ktv, sc))

                # Scale and exp (scale is (SEQ,SEQ) -- same shape as scores)
                with scores_dfb.wait() as scv, scores_dfb.reserve() as esc:
                    esc.store(ttl.math.exp(scv * sv))

                # Attn output: (2,2) @ (2,1) = (2,1)
                with scores_dfb.wait() as ev, v_dfb.wait() as vv:
                    with out_dfb.reserve() as o:
                        o.store(ttl.math.matmul(ev, vv, o))

    @ttl.datamovement()
    def dm_read():
        # Each core reads its own head slice
        x, y = ttl.node(dims=2)
        head_col = x * HEAD_DIM  # Column offset for this head

        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:SEQ, head_col:head_col+HEAD_DIM], blk)
            tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(k[0:SEQ, head_col:head_col+HEAD_DIM], blk)
            tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(v[0:SEQ, head_col:head_col+HEAD_DIM], blk)
            tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0:SEQ, 0:SEQ], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        head_col = x * HEAD_DIM

        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ, head_col:head_col+HEAD_DIM])
            tx.wait()


# ============================================================================
# Multinode MLP: distribute output columns across cores
# ============================================================================
@ttl.operation(grid=(NUM_HEADS, 1))
def multinode_linear_kernel(x, w, out):
    """
    Distributed linear: each core computes a slice of the output columns.
    x: (SEQ, HIDDEN) -- all cores read full input
    w: (HIDDEN, HIDDEN) -- each core reads its column slice (HIDDEN, HEAD_DIM)
    out: (SEQ, HIDDEN) -- each core writes its column slice
    """
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ, HIDDEN), buffer_factor=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(HIDDEN, HEAD_DIM), buffer_factor=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ, HEAD_DIM), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv, w_dfb.wait() as wv:
            with out_dfb.reserve() as o:
                o.store(ttl.math.matmul(xv, wv, o))

    @ttl.datamovement()
    def dm_read():
        x_node, y_node = ttl.node(dims=2)
        col = x_node * HEAD_DIM

        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ, 0:HIDDEN], blk)
            tx.wait()
        with w_dfb.reserve() as blk:
            tx = ttl.copy(w[0:HIDDEN, col:col+HEAD_DIM], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x_node, y_node = ttl.node(dims=2)
        col = x_node * HEAD_DIM

        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ, col:col+HEAD_DIM])
            tx.wait()


# ============================================================================
# Test
# ============================================================================
def test_multinode(device):
    """Test multinode attention and linear."""
    torch.manual_seed(42)

    seq = SEQ * 32        # 64
    hid = HIDDEN * 32     # 128
    head = HEAD_DIM * 32  # 32
    scale_val = 1.0 / (head ** 0.5)

    def d(t):
        return to_device(t, device)

    def rand(r, c, s=0.02):
        return torch.randn(r, c, dtype=torch.bfloat16) * s

    # Inputs
    q_torch = rand(seq, hid, 0.1)
    k_torch = rand(seq, hid, 0.1)
    v_torch = rand(seq, hid, 0.1)
    w_torch = rand(hid, hid)
    scale_tile = torch.full((seq, seq), scale_val, dtype=torch.bfloat16)

    attn_out = d(torch.zeros(seq, hid, dtype=torch.bfloat16))
    linear_out = d(torch.zeros(seq, hid, dtype=torch.bfloat16))

    print(f"=== Multinode Test: grid=({NUM_HEADS}, 1), seq={seq}, hidden={hid} ===\n")

    # Test 1: Multihead attention
    print(f"Kernel 1: Multihead attention ({NUM_HEADS} cores, 1 head each)...")
    multihead_attention_kernel(d(q_torch), d(k_torch), d(v_torch), d(scale_tile), attn_out)

    # Reference: per-head attention
    attn_parts = []
    for h in range(NUM_HEADS):
        qh = q_torch[:, h*head:(h+1)*head].float()
        kh = k_torch[:, h*head:(h+1)*head].float()
        vh = v_torch[:, h*head:(h+1)*head].float()
        scores = qh @ kh.T
        attn_parts.append((torch.exp(scores * scale_val) @ vh))
    attn_expected = torch.cat(attn_parts, dim=-1).bfloat16()
    attn_result = ttnn.to_torch(attn_out)

    attn_corr = torch.corrcoef(
        torch.stack([attn_result.float().flatten(), attn_expected.float().flatten()])
    )[0, 1].item()
    print(f"  Attention correlation: {attn_corr:.6f}")

    # Test 2: Distributed linear
    print(f"\nKernel 2: Distributed linear ({NUM_HEADS} cores)...")
    multinode_linear_kernel(d(q_torch), d(w_torch), linear_out)

    lin_expected = (q_torch.float() @ w_torch.float()).bfloat16()
    lin_result = ttnn.to_torch(linear_out)

    lin_corr = torch.corrcoef(
        torch.stack([lin_result.float().flatten(), lin_expected.float().flatten()])
    )[0, 1].item()
    print(f"  Linear correlation: {lin_corr:.6f}")

    # Summary
    print(f"\n{'='*50}")
    all_pass = attn_corr > 0.95 and lin_corr > 0.95
    if all_pass:
        print("PASSED: All multinode kernels match reference")
    else:
        print("FAILED")
    print(f"  Attention:  {attn_corr:.6f}")
    print(f"  Linear:     {lin_corr:.6f}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_multinode(device)
    finally:
        ttnn.close_device(device)
