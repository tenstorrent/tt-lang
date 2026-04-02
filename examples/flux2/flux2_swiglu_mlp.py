# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 SwiGLU MLP kernel for TT-Lang.

Implements the Flux2FeedForward block used in both single-stream and
double-stream transformer blocks:
  1. linear_in: x @ W_in -> [2 * mlp_hidden]  (one linear produces both halves)
  2. SwiGLU: SiLU(x1) * x2 where x1, x2 = split(linear_out)
  3. linear_out: swiglu_out @ W_out -> [hidden_dim]

SiLU(x) = x * sigmoid(x)

Test configuration (scaled down):
  hidden_dim = 128 (4 tiles)
  mlp_hidden = 384 (12 tiles) -> linear_in output = 768 (24 tiles)
  seq_len = 32 (1 tile)
"""

import torch
import torch.nn.functional as F

import ttnn
import ttl

# Test config (scaled down from FLUX.2 klein-4B: 3072 hidden, 9216 mlp)
SEQ_TILES = 1       # 32 tokens
HIDDEN_TILES = 4    # 128 hidden dim
MLP_TILES = 12      # 384 mlp hidden dim

# For SwiGLU, linear_in projects to 2x mlp_hidden (one half for gate, one for value)
# We split this into two separate weight matrices for TT-Lang since we can't
# easily split a single large output in the compute thread.
SWIGLU_TILES = MLP_TILES  # Each half is mlp_hidden


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# SwiGLU MLP Kernel
# ============================================================================
@ttl.operation(grid=(1, 1))
def swiglu_mlp_kernel(x, w_gate, w_up, w_down, out):
    """
    SwiGLU MLP: out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Split the linear_in into two matmuls (gate and up projections)
    to avoid needing to split a tensor in-kernel.
    """
    x_dfb = ttl.make_dataflow_buffer_like(
        x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    w_gate_dfb = ttl.make_dataflow_buffer_like(
        w_gate, shape=(HIDDEN_TILES, SWIGLU_TILES), buffer_factor=1
    )
    w_up_dfb = ttl.make_dataflow_buffer_like(
        w_up, shape=(HIDDEN_TILES, SWIGLU_TILES), buffer_factor=1
    )
    w_down_dfb = ttl.make_dataflow_buffer_like(
        w_down, shape=(SWIGLU_TILES, HIDDEN_TILES), buffer_factor=1
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )

    # Intermediates
    gate_dfb = ttl.make_dataflow_buffer_like(
        w_gate, shape=(SEQ_TILES, SWIGLU_TILES), buffer_factor=2
    )
    up_dfb = ttl.make_dataflow_buffer_like(
        w_up, shape=(SEQ_TILES, SWIGLU_TILES), buffer_factor=2
    )
    swiglu_dfb = ttl.make_dataflow_buffer_like(
        w_up, shape=(SEQ_TILES, SWIGLU_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        # Keep x in scope for both projections
        with x_dfb.wait() as xv:
            # Gate projection: x @ W_gate
            with w_gate_dfb.wait() as wg, gate_dfb.reserve() as g:
                g.store(ttl.math.matmul(xv, wg, g))

            # Up projection: x @ W_up
            with w_up_dfb.wait() as wu, up_dfb.reserve() as u:
                u.store(ttl.math.matmul(xv, wu, u))

        # SwiGLU: SiLU(gate) * up = (gate * sigmoid(gate)) * up
        with gate_dfb.wait() as gv, up_dfb.wait() as uv, swiglu_dfb.reserve() as sw:
            silu_gate = gv * ttl.math.sigmoid(gv)
            sw.store(silu_gate * uv)

        # Down projection: swiglu_out @ W_down
        with swiglu_dfb.wait() as swv, w_down_dfb.wait() as wd:
            with out_dfb.reserve() as o:
                o.store(ttl.math.matmul(swv, wd, o))

    @ttl.datamovement()
    def dm_read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with w_gate_dfb.reserve() as blk:
            tx = ttl.copy(w_gate[0:HIDDEN_TILES, 0:SWIGLU_TILES], blk)
            tx.wait()
        with w_up_dfb.reserve() as blk:
            tx = ttl.copy(w_up[0:HIDDEN_TILES, 0:SWIGLU_TILES], blk)
            tx.wait()
        with w_down_dfb.reserve() as blk:
            tx = ttl.copy(w_down[0:SWIGLU_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


def test_swiglu_mlp(device):
    """Test SwiGLU MLP against PyTorch reference."""
    torch.manual_seed(42)

    seq_len = SEQ_TILES * 32
    hidden_dim = HIDDEN_TILES * 32
    mlp_hidden = MLP_TILES * 32

    # Random inputs and weights (small init for numerical stability)
    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    w_gate_torch = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    w_up_torch = torch.randn(hidden_dim, mlp_hidden, dtype=torch.bfloat16) * 0.02
    w_down_torch = torch.randn(mlp_hidden, hidden_dim, dtype=torch.bfloat16) * 0.02
    out_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)

    # Convert to TTNN
    x = to_device(x_torch, device)
    w_gate = to_device(w_gate_torch, device)
    w_up = to_device(w_up_torch, device)
    w_down = to_device(w_down_torch, device)
    out = to_device(out_torch, device)

    # Run kernel
    print("Running SwiGLU MLP kernel...")
    swiglu_mlp_kernel(x, w_gate, w_up, w_down, out)

    # Read result
    result = ttnn.to_torch(out)

    # PyTorch reference
    x_f = x_torch.float()
    gate = x_f @ w_gate_torch.float()
    up = x_f @ w_up_torch.float()
    silu_gate = gate * torch.sigmoid(gate)  # SiLU
    swiglu_out = silu_gate * up
    expected = (swiglu_out @ w_down_torch.float()).bfloat16()

    print(f"Expected[0,:8]: {expected[0,:8]}")
    print(f"Result[0,:8]:   {result[0,:8]}")

    if torch.isnan(result).any():
        print("WARNING: Result contains NaN")
        return

    abs_diff = torch.abs(result.float() - expected.float())
    print(f"Max absolute diff: {abs_diff.max().item():.6f}")
    print(f"Mean absolute diff: {abs_diff.mean().item():.6f}")

    # Check correlation
    correlation = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.95:
        print("\nPASSED: SwiGLU MLP matches PyTorch reference")
    else:
        print(f"\nFAILED: Correlation too low: {correlation}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_swiglu_mlp(device)
    finally:
        ttnn.close_device(device)
