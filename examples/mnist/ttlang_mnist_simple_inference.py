# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal MNIST inference with TT-Lang."""

import torch
import numpy as np
import ttnn
import ttl

# Dimensions
BATCH = 32
INPUT_DIM = 800
HIDDEN_DIM = 1024
OUTPUT_DIM = 32
NUM_CHUNKS = 8
CHUNK_SIZE = 128
BATCH_TILES = 1
INPUT_TILES = 25
CHUNK_TILES = 4
OUTPUT_TILES = 1


@ttl.kernel(grid=(NUM_CHUNKS, 1))
def layer1_kernel(x, w1, bias1, hidden_out):
    x_cb = ttl.make_circular_buffer_like(x, shape=(BATCH_TILES, INPUT_TILES), buffer_factor=1)
    w1_cb = ttl.make_circular_buffer_like(w1, shape=(INPUT_TILES, CHUNK_TILES), buffer_factor=1)
    bias1_cb = ttl.make_circular_buffer_like(bias1, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=1)
    hidden_mm_cb = ttl.make_circular_buffer_like(hidden_out, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=2)
    hidden_cb = ttl.make_circular_buffer_like(hidden_out, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv, w1_cb.wait() as w1v:
            with hidden_mm_cb.reserve() as hmm:
                hmm.store(ttl.math.matmul(xv, w1v, hmm))
        with hidden_mm_cb.wait() as hmmv, bias1_cb.wait() as b1v:
            with hidden_cb.reserve() as h:
                h.store(ttl.math.relu(hmmv + b1v))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:BATCH_TILES, 0:INPUT_TILES], blk)
            tx.wait()
        col_start = core_x * CHUNK_TILES
        col_end = col_start + CHUNK_TILES
        with w1_cb.reserve() as blk:
            tx = ttl.copy(w1[0:INPUT_TILES, col_start:col_end], blk)
            tx.wait()
        with bias1_cb.reserve() as blk:
            tx = ttl.copy(bias1[0:BATCH_TILES, col_start:col_end], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        col_start = core_x * CHUNK_TILES
        col_end = col_start + CHUNK_TILES
        with hidden_cb.wait() as blk:
            tx = ttl.copy(blk, hidden_out[0:BATCH_TILES, col_start:col_end])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def layer2_kernel(hidden, w2, bias2, scaler, out):
    HIDDEN_TILES = 32
    hidden_cb = ttl.make_circular_buffer_like(hidden, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=2)
    w2_cb = ttl.make_circular_buffer_like(w2, shape=(CHUNK_TILES, OUTPUT_TILES), buffer_factor=2)
    acc_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    part_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    bias2_cb = ttl.make_circular_buffer_like(bias2, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=1)
    logits_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    max_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    max_bcast_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    exp_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    sum_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    sum_bcast_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        with hidden_cb.wait() as hc, w2_cb.wait() as wc:
            with acc_cb.reserve() as acc:
                acc.store(ttl.math.matmul(hc, wc, acc))
        for _ in range(NUM_CHUNKS - 1):
            with hidden_cb.wait() as hc, w2_cb.wait() as wc:
                with part_cb.reserve() as part:
                    part.store(ttl.math.matmul(hc, wc, part))
            with part_cb.wait() as pv, acc_cb.wait() as av:
                with acc_cb.reserve() as new_acc:
                    new_acc.store(av + pv)
        with acc_cb.wait() as accv, bias2_cb.wait() as b2v:
            with logits_cb.reserve() as lg:
                lg.store(accv + b2v)
        with logits_cb.wait() as lgv, scaler_cb.wait() as sc:
            with max_cb.reserve() as mx:
                mx.store(ttl.math.reduce_max(lgv, sc, mx, dims=[0]))
            with max_cb.wait() as mxv, max_bcast_cb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, mxb, dims=[1]))
            with max_bcast_cb.wait() as mxbv:
                with exp_cb.reserve() as ex:
                    ex.store(ttl.math.exp(lgv - mxbv))
                with exp_cb.wait() as exv, sum_cb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, sc, sm, dims=[0]))
                with sum_cb.wait() as smv, sum_bcast_cb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv, smb, dims=[1]))
                with sum_bcast_cb.wait() as smbv, out_cb.reserve() as o:
                    o.store(ttl.math.exp(lgv - mxbv) / smbv)

    @ttl.datamovement()
    def dm_read():
        for i in range(NUM_CHUNKS):
            col_start = i * CHUNK_TILES
            col_end = col_start + CHUNK_TILES
            with hidden_cb.reserve() as blk:
                tx = ttl.copy(hidden[0:BATCH_TILES, col_start:col_end], blk)
                tx.wait()
            with w2_cb.reserve() as blk:
                tx = ttl.copy(w2[col_start:col_end, 0:OUTPUT_TILES], blk)
                tx.wait()
        with bias2_cb.reserve() as blk:
            tx = ttl.copy(bias2[0:BATCH_TILES, 0:OUTPUT_TILES], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0:BATCH_TILES, 0:OUTPUT_TILES])
            tx.wait()


def to_ttnn(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    # Paths (absolute for VM)
    base = "/home/zcarver.linux"

    # Load weights
    weights = torch.load(f"{base}/mnist_weights.pt", weights_only=True)
    w1_orig, b1_orig, w2_orig, b2_orig = weights['w1'], weights['b1'], weights['w2'], weights['b2']

    # Pad weights
    w1 = torch.zeros(INPUT_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
    w1[:784, :] = w1_orig.to(torch.bfloat16)
    b1 = b1_orig.unsqueeze(0).expand(BATCH, -1).contiguous().to(torch.bfloat16)
    w2 = torch.zeros(HIDDEN_DIM, OUTPUT_DIM, dtype=torch.bfloat16)
    w2[:, :10] = w2_orig.to(torch.bfloat16)
    b2_full = torch.zeros(OUTPUT_DIM, dtype=torch.bfloat16)
    b2_full[:10] = b2_orig.to(torch.bfloat16)
    b2 = b2_full.unsqueeze(0).expand(BATCH, -1).contiguous()

    # Load test data
    X_test = np.fromfile(f"{base}/data/X_test.bin", dtype=np.float32).reshape(-1, 784)
    y_test = np.fromfile(f"{base}/data/y_test.bin", dtype=np.int32)
    X_test = (X_test - 0.1307) / 0.3081  # Normalize

    print("=== TT-Lang MNIST Inference ===")
    device = ttnn.open_device(device_id=0)

    scaler = torch.ones(32, 32, dtype=torch.bfloat16)
    hidden_buf = torch.zeros(BATCH, HIDDEN_DIM, dtype=torch.bfloat16)
    out_buf = torch.zeros(BATCH, OUTPUT_DIM, dtype=torch.bfloat16)

    # Convert weights to TTNN
    w1_tt = to_ttnn(w1, device)
    b1_tt = to_ttnn(b1, device)
    w2_tt = to_ttnn(w2, device)
    b2_tt = to_ttnn(b2, device)
    scaler_tt = to_ttnn(scaler, device)

    correct, total = 0, 0
    num_batches = 10

    for i in range(num_batches):
        # Pad input
        x_np = X_test[i*BATCH:(i+1)*BATCH]
        x = torch.zeros(BATCH, INPUT_DIM, dtype=torch.bfloat16)
        x[:, :784] = torch.from_numpy(x_np).to(torch.bfloat16)

        x_tt = to_ttnn(x, device)
        hidden_tt = to_ttnn(hidden_buf, device)
        out_tt = to_ttnn(out_buf, device)

        # Run kernels
        layer1_kernel(x_tt, w1_tt, b1_tt, hidden_tt)
        layer2_kernel(hidden_tt, w2_tt, b2_tt, scaler_tt, out_tt)

        # Get predictions
        result = ttnn.to_torch(out_tt).float()[:, :10]
        preds = result.argmax(dim=1).numpy()
        labels = y_test[i*BATCH:(i+1)*BATCH]

        correct += (preds == labels).sum()
        total += BATCH

        if i == 0:
            print(f"Predictions: {preds[:8]}")
            print(f"Labels:      {labels[:8]}")

    print(f"\nAccuracy: {100*correct/total:.2f}% ({correct}/{total})")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
