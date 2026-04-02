# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 Adaptive LayerNorm (AdaLN) kernel for TT-Lang.

Implements: AdaLN(x, shift, scale) = (1 + scale) * RMSNorm(x) + shift

Uses RMSNorm (no mean subtraction) as the normalization:
  RMSNorm(x) = x * rsqrt(mean(x^2))

In FLUX.2, shift and scale come from the timestep modulation MLP.

Note: Uses scalar reduction/broadcast (dims=[0, 1]) which computes a
global RMS across the entire tile block. For per-token normalization on
hardware, use dims=[0]/dims=[1] with the compiler (not supported in sim).

Test configuration:
  hidden_dim = 32 (1 tile)
  seq_len = 32 (1 tile)
"""

import torch

import ttnn
import ttl

SEQ_TILES = 1       # 32 tokens
HIDDEN_TILES = 1    # 32 hidden dim


def to_device(tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(1, 1))
def adaln_kernel(x, shift, scale, scaler, out):
    """
    AdaLN: out = (1 + scale) * RMSNorm(x) + shift

    Steps:
      1. sq = x * x
      2. mean_sq = reduce_sum(sq) * scaler  (scaler = 1/num_elements)
      3. rstd = rsqrt(mean_sq)
      4. norm = x * broadcast(rstd)
      5. out = norm + scale * norm + shift   (= (1+scale)*norm + shift)
    """
    x_dfb = ttl.make_dataflow_buffer_like(
        x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    shift_dfb = ttl.make_dataflow_buffer_like(
        shift, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1
    )
    scale_dfb = ttl.make_dataflow_buffer_like(
        scale, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=1
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )

    # Intermediates
    sq_dfb = ttl.make_dataflow_buffer_like(
        x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )
    reduce_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=2
    )
    bcast_dfb = ttl.make_dataflow_buffer_like(
        x, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv, scaler_dfb.wait() as sc:
            # 1. Square
            with sq_dfb.reserve() as sq:
                sq.store(xv * xv)

            # 2. Scalar reduce: sum of squares
            with sq_dfb.wait() as sqv, reduce_dfb.reserve() as red:
                red.store(ttl.math.reduce_sum(sqv, sc, red, dims=[0, 1]))

            # 3. Rsqrt
            with reduce_dfb.wait() as rv, reduce_dfb.reserve() as rs:
                rs.store(ttl.math.rsqrt(rv))

            # 4. Broadcast rstd back to full size
            with reduce_dfb.wait() as rsv, bcast_dfb.reserve() as bc:
                bc.store(ttl.math.broadcast(rsv, bc, dims=[0, 1]))

            # 5. Normalize and apply AdaLN modulation
            with (
                bcast_dfb.wait() as rstd,
                scale_dfb.wait() as scalev,
                shift_dfb.wait() as shiftv,
            ):
                normalized = xv * rstd
                with out_dfb.reserve() as o:
                    # (1 + scale) * normalized + shift
                    o.store(normalized + scalev * normalized + shiftv)

    @ttl.datamovement()
    def dm_read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with shift_dfb.reserve() as blk:
            tx = ttl.copy(shift[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


def test_adaln(device):
    """Test AdaLN against PyTorch reference."""
    torch.manual_seed(42)

    seq_len = SEQ_TILES * 32
    hidden_dim = HIDDEN_TILES * 32
    num_elements = seq_len * hidden_dim

    # Inputs
    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    shift_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    scale_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    out_torch = torch.zeros(seq_len, hidden_dim, dtype=torch.bfloat16)

    # Scaler: 1/num_elements for global RMS computation
    scaler_val = 1.0 / num_elements
    scaler_torch = torch.full((32, 32), scaler_val, dtype=torch.bfloat16)

    x = to_device(x_torch, device)
    shift = to_device(shift_torch, device)
    scale = to_device(scale_torch, device)
    scaler = to_device(scaler_torch, device)
    out = to_device(out_torch, device)

    print("Running AdaLN kernel...")
    adaln_kernel(x, shift, scale, scaler, out)

    result = ttnn.to_torch(out)

    # PyTorch reference: global RMSNorm + AdaLN modulation
    x_f = x_torch.float()
    rms = torch.sqrt((x_f ** 2).mean() + 1e-6)
    normalized = x_f / rms
    expected = ((1 + scale_torch.float()) * normalized + shift_torch.float()).bfloat16()

    print(f"Expected[0,:8]: {expected[0,:8]}")
    print(f"Result[0,:8]:   {result[0,:8]}")

    if torch.isnan(result).any():
        print("WARNING: Result contains NaN")
        return

    abs_diff = torch.abs(result.float() - expected.float())
    print(f"Max absolute diff: {abs_diff.max().item():.6f}")
    print(f"Mean absolute diff: {abs_diff.mean().item():.6f}")

    correlation = torch.corrcoef(
        torch.stack([result.float().flatten(), expected.float().flatten()])
    )[0, 1].item()
    print(f"Correlation: {correlation:.6f}")

    if correlation > 0.95:
        print("\nPASSED: AdaLN matches PyTorch reference")
    else:
        print(f"\nFAILED: Correlation too low: {correlation}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_adaln(device)
    finally:
        ttnn.close_device(device)
