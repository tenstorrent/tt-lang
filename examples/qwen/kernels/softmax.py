# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-side softmax for Qwen attention.

Uses hardware reduce_max and reduce_sum (dims=[1] = row-wise).
Three-kernel approach — data stays on device between kernels.

For decode: scores are [32, 512] = [1, 16] tiles, only row 0 matters.
dims=[1] row-wise reduce puts per-row results in row 0, perfect for decode.
"""

import torch
import ttl
import ttnn

TILE = 32


@ttl.kernel(grid=(1, 1))
def fused_mask_max_kernel(scores, mask, scaler, masked_out, max_out):
    """Apply mask and find row-wise max across all column tiles.

    For each column tile: masked = scores + mask, then reduce_max(dims=[1]).
    Accumulate max across tiles via element-wise max.

    scores, mask, masked_out: [Mt, Nt] tiles
    scaler: [1, 1] tile (ones)
    max_out: [Mt, 1] tile — row-wise max in row 0
    """
    Mt = scores.shape[0] // TILE
    Nt = scores.shape[1] // TILE

    s_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(mask, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    o_dfb = ttl.make_dataflow_buffer_like(masked_out, shape=(1, 1), buffer_factor=2)
    # Compute-local: masked value (needed as CB input for reduce)
    masked_dfb = ttl.make_dataflow_buffer_like(scores, shape=(1, 1), buffer_factor=2)
    # Compute-local accumulator for cross-tile max
    acc_dfb = ttl.make_dataflow_buffer_like(max_out, shape=(1, 1), buffer_factor=2)
    mx_dfb = ttl.make_dataflow_buffer_like(max_out, shape=(1, 1), buffer_factor=2)
    # Temp for within-tile reduce result
    tmp_dfb = ttl.make_dataflow_buffer_like(max_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()
            for col in range(Nt):
                with s_dfb.reserve() as blk:
                    tx = ttl.copy(scores[row, col], blk)
                    tx.wait()
                with m_dfb.reserve() as blk:
                    tx = ttl.copy(mask[row, col], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            with sc_dfb.wait() as sc_blk:
                # First column: mask → store masked → reduce → init max
                with s_dfb.wait() as s, m_dfb.wait() as m:
                    with o_dfb.reserve() as out:
                        out.store(s + m)
                    with masked_dfb.reserve() as msk:
                        msk.store(s + m)
                with masked_dfb.wait() as msk_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                with tmp_dfb.wait() as reduced:
                    with acc_dfb.reserve() as acc:
                        acc.store(reduced)

                # Remaining columns
                for _ in range(Nt - 1):
                    with s_dfb.wait() as s, m_dfb.wait() as m:
                        with o_dfb.reserve() as out:
                            out.store(s + m)
                        with masked_dfb.reserve() as msk:
                            msk.store(s + m)
                    with masked_dfb.wait() as msk_blk:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_max(msk_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as reduced, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(ttl.math.max(prev, reduced))

            # Write final max
            with acc_dfb.wait() as final_max:
                with mx_dfb.reserve() as mx:
                    mx.store(final_max)

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            for col in range(Nt):
                with o_dfb.wait() as blk:
                    tx = ttl.copy(blk, masked_out[row, col])
                    tx.wait()
            with mx_dfb.wait() as blk:
                tx = ttl.copy(blk, max_out[row, 0])
                tx.wait()


@ttl.kernel(grid=(1, 1))
def fused_exp_sum_kernel(masked, row_max, scaler, exp_out, sum_out):
    """Compute exp(masked - max) and row-wise sum.

    masked: [Mt, Nt], row_max: [Mt, 1], scaler: [1, 1] (ones)
    exp_out: [Mt, Nt], sum_out: [Mt, 1]
    """
    Mt = masked.shape[0] // TILE
    Nt = masked.shape[1] // TILE

    m_dfb = ttl.make_dataflow_buffer_like(masked, shape=(1, 1), buffer_factor=2)
    mx_dfb = ttl.make_dataflow_buffer_like(row_max, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    e_dfb = ttl.make_dataflow_buffer_like(exp_out, shape=(1, 1), buffer_factor=2)
    # Compute-local: exp value for reduce input
    exp_local_dfb = ttl.make_dataflow_buffer_like(exp_out, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(sum_out, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(sum_out, shape=(1, 1), buffer_factor=2)
    sm_dfb = ttl.make_dataflow_buffer_like(sum_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            with mx_dfb.reserve() as blk:
                tx = ttl.copy(row_max[row, 0], blk)
                tx.wait()
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()
            for col in range(Nt):
                with m_dfb.reserve() as blk:
                    tx = ttl.copy(masked[row, col], blk)
                    tx.wait()

    # Compute-local: broadcast max for subtraction
    mx_bc_dfb = ttl.make_dataflow_buffer_like(row_max, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            with mx_dfb.wait() as max_blk, sc_dfb.wait() as sc_blk:
                # Scalar broadcast: max[0,0] has the row-0 max,
                # broadcast to all positions for subtraction
                with mx_bc_dfb.reserve() as mx_bc:
                    mx_bc.store(ttl.math.broadcast(max_blk, mx_bc, dims=[0, 1]))

            with mx_bc_dfb.wait() as max_bc:
                # First column: exp → output + local CB → reduce → init sum
                with m_dfb.wait() as masked_blk:
                    with e_dfb.reserve() as exp_tile:
                        exp_tile.store(ttl.math.exp(masked_blk - max_bc))
                    with exp_local_dfb.reserve() as el:
                        el.store(ttl.math.exp(masked_blk - max_bc))
                with exp_local_dfb.wait() as el_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(ttl.math.reduce_sum(el_blk, sc_blk, tmp, dims=[1]))
                with tmp_dfb.wait() as reduced:
                    with acc_dfb.reserve() as acc:
                        acc.store(reduced)

                # Remaining columns
                for _ in range(Nt - 1):
                    with m_dfb.wait() as masked_blk:
                        with e_dfb.reserve() as exp_tile:
                            exp_tile.store(ttl.math.exp(masked_blk - max_bc))
                        with exp_local_dfb.reserve() as el:
                            el.store(ttl.math.exp(masked_blk - max_bc))
                    with exp_local_dfb.wait() as el_blk:
                        with tmp_dfb.reserve() as tmp:
                            tmp.store(ttl.math.reduce_sum(el_blk, sc_blk, tmp, dims=[1]))
                    with tmp_dfb.wait() as reduced, acc_dfb.wait() as prev:
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + reduced)

            with acc_dfb.wait() as final_sum:
                with sm_dfb.reserve() as sm:
                    sm.store(final_sum)

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            for col in range(Nt):
                with e_dfb.wait() as blk:
                    tx = ttl.copy(blk, exp_out[row, col])
                    tx.wait()
            with sm_dfb.wait() as blk:
                tx = ttl.copy(blk, sum_out[row, 0])
                tx.wait()


@ttl.kernel(grid=(1, 1))
def normalize_kernel(exp_scores, row_sum, Y):
    """Y = exp_scores * recip(row_sum). row_sum has values in row 0."""
    Mt = exp_scores.shape[0] // TILE
    Nt = exp_scores.shape[1] // TILE

    e_dfb = ttl.make_dataflow_buffer_like(exp_scores, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(row_sum, shape=(1, 1), buffer_factor=2)
    s_bc_dfb = ttl.make_dataflow_buffer_like(row_sum, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for row in range(Mt):
            with s_dfb.reserve() as blk:
                tx = ttl.copy(row_sum[row, 0], blk)
                tx.wait()
            for col in range(Nt):
                with e_dfb.reserve() as blk:
                    tx = ttl.copy(exp_scores[row, col], blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(Mt):
            # Scalar broadcast: sum[0,0] has the row-0 sum,
            # broadcast to all positions for division
            with s_dfb.wait() as sum_blk:
                with s_bc_dfb.reserve() as s_bc:
                    s_bc.store(ttl.math.broadcast(sum_blk, s_bc, dims=[0, 1]))
            with s_bc_dfb.wait() as sum_bc:
                for _ in range(Nt):
                    with e_dfb.wait() as exp_blk, y_dfb.reserve() as out:
                        out.store(exp_blk * ttl.math.recip(sum_bc))

    @ttl.datamovement()
    def write():
        for row in range(Mt):
            for col in range(Nt):
                with y_dfb.wait() as blk:
                    tx = ttl.copy(blk, Y[row, col])
                    tx.wait()


def device_softmax(scores_dev, mask_dev, scaler_dev, device):
    """Full device-side softmax. No host transfers.

    Args:
        scores_dev: [rows, cols] on device
        mask_dev: [rows, cols] on device (0 for valid, -inf for masked)
        scaler_dev: [32, 32] ones tile on device
        device: TTNN device

    Returns:
        weights_dev: [rows, cols] on device
    """
    rows = scores_dev.shape[0]
    cols = scores_dev.shape[1]

    def alloc(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    masked = alloc((rows, cols))
    row_max = alloc((rows, TILE))
    exp_out = alloc((rows, cols))
    row_sum = alloc((rows, TILE))
    weights = alloc((rows, cols))

    fused_mask_max_kernel(scores_dev, mask_dev, scaler_dev, masked, row_max)
    fused_exp_sum_kernel(masked, row_max, scaler_dev, exp_out, row_sum)
    normalize_kernel(exp_out, row_sum, weights)

    return weights


# =========================================================================
# Test
# =========================================================================
def _to_device(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_softmax_decode(device):
    """Test decode-size softmax: [32, 512], only row 0 matters."""
    rows, cols = TILE, 512
    print(f"  softmax [{rows}x{cols}] decode...", end="", flush=True)

    scores_t = torch.randn(rows, cols, dtype=torch.bfloat16) * 0.5
    mask_t = torch.full((rows, cols), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :200] = 0.0  # attend to first 200 positions

    scores = _to_device(scores_t, device)
    mask = _to_device(mask_t, device)
    scaler = _to_device(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)

    result_dev = device_softmax(scores, mask, scaler, device)
    result = ttnn.to_torch(result_dev)

    expected = torch.nn.functional.softmax(
        (scores_t.float() + mask_t.float()), dim=-1
    ).bfloat16()

    # Check row 0 only (decode)
    score = torch.corrcoef(
        torch.stack([result[0].float(), expected[0].float()])
    )[0, 1].item()
    print(f" PCC={score:.6f}", end="")
    assert score > 0.98, f" FAIL"
    print(" PASS")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        print("Device softmax tests:")
        test_softmax_decode(device)
        print("All softmax tests passed!")
    finally:
        ttnn.close_device(device)
