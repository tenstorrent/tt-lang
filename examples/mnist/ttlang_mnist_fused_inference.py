# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MNIST inference with fully fused single-kernel (pipes).

NOTE: This hits a simulator limitation with reduce_max after pipes.
      May work on real hardware.

Optimizations:
- Single kernel (no DRAM round-trip for hidden layer)
- 8 cores compute layer1 + layer2 partial in parallel
- Pipes gather partial results to core 0
- Core 0 accumulates, adds bias, does softmax
"""

import torch
import numpy as np
import ttnn
import ttl

# Dimensions
BATCH = 32
INPUT_DIM = 800
HIDDEN_DIM = 1024
OUTPUT_DIM = 32
NUM_CORES = 8
CHUNK_SIZE = 128
BATCH_TILES = 1
INPUT_TILES = 25
CHUNK_TILES = 4
OUTPUT_TILES = 1


@ttl.kernel(grid=(NUM_CORES, 1))
def mnist_fused_kernel(x, w1, bias1, w2, bias2, scaler, out):
    """
    Fully fused MNIST forward - single kernel, no DRAM for hidden.

    Each core:
    1. Computes hidden_chunk = relu(X @ W1_chunk + bias1_chunk)
    2. Computes partial = hidden_chunk @ W2_chunk
    3. Workers send partials to core 0 via pipes
    4. Core 0 accumulates, adds bias2, does softmax
    """
    # Pipes: workers -> coordinator
    pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))
    pipe4 = ttl.Pipe(src=(4, 0), dst=(0, 0))
    pipe5 = ttl.Pipe(src=(5, 0), dst=(0, 0))
    pipe6 = ttl.Pipe(src=(6, 0), dst=(0, 0))
    pipe7 = ttl.Pipe(src=(7, 0), dst=(0, 0))

    # Layer 1 CBs
    x_cb = ttl.make_circular_buffer_like(x, shape=(BATCH_TILES, INPUT_TILES), buffer_factor=1)
    w1_cb = ttl.make_circular_buffer_like(w1, shape=(INPUT_TILES, CHUNK_TILES), buffer_factor=1)
    bias1_cb = ttl.make_circular_buffer_like(bias1, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=1)
    hidden_mm_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=2)
    hidden_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, CHUNK_TILES), buffer_factor=2)

    # Layer 2 CBs
    w2_cb = ttl.make_circular_buffer_like(w2, shape=(CHUNK_TILES, OUTPUT_TILES), buffer_factor=1)
    partial_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)
    gather_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=8)
    acc_cb = ttl.make_circular_buffer_like(out, shape=(BATCH_TILES, OUTPUT_TILES), buffer_factor=2)

    # Coordinator-only CBs
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
        core_x, core_y = ttl.core(dims=2)

        # Layer 1: all cores compute their hidden chunk
        with x_cb.wait() as xv, w1_cb.wait() as w1v:
            with hidden_mm_cb.reserve() as hmm:
                result = ttl.math.matmul(xv, w1v, hmm)
                hmm.store(result)

        with hidden_mm_cb.wait() as hmmv, bias1_cb.wait() as b1v:
            with hidden_cb.reserve() as h:
                h.store(ttl.math.relu(hmmv + b1v))

        # Layer 2: all cores compute partial output
        with hidden_cb.wait() as hc, w2_cb.wait() as wc:
            with partial_cb.reserve() as part:
                result = ttl.math.matmul(hc, wc, part)
                part.store(result)

        if core_x == 0:
            # Coordinator: init accumulator with own partial
            with partial_cb.wait() as p0, acc_cb.reserve() as acc:
                acc.store(ttl.math.abs(p0))

            # Receive and accumulate from 7 workers
            for _ in range(NUM_CORES - 1):
                with gather_cb.wait() as g, acc_cb.wait() as av:
                    with acc_cb.reserve() as new_acc:
                        new_acc.store(av + g)

            # Add bias2
            with acc_cb.wait() as accv, bias2_cb.wait() as b2v:
                with logits_cb.reserve() as lg:
                    lg.store(accv + b2v)

            # Softmax
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
        else:
            # Workers: copy partial to gather_cb for pipe
            with partial_cb.wait() as pv, gather_cb.reserve() as g:
                g.store(ttl.math.abs(pv))

    @ttl.datamovement()
    def dm_read():
        core_x, core_y = ttl.core(dims=2)

        # Coordinator loads bias2 and scaler first
        if core_x == 0:
            with bias2_cb.reserve() as blk:
                tx = ttl.copy(bias2[0:BATCH_TILES, 0:OUTPUT_TILES], blk)
                tx.wait()
            with scaler_cb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk)
                tx.wait()

        # All cores load X
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0:BATCH_TILES, 0:INPUT_TILES], blk)
            tx.wait()

        # Each core loads its chunks
        col_start = core_x * CHUNK_TILES
        col_end = col_start + CHUNK_TILES

        with w1_cb.reserve() as blk:
            tx = ttl.copy(w1[0:INPUT_TILES, col_start:col_end], blk)
            tx.wait()

        with bias1_cb.reserve() as blk:
            tx = ttl.copy(bias1[0:BATCH_TILES, col_start:col_end], blk)
            tx.wait()

        with w2_cb.reserve() as blk:
            tx = ttl.copy(w2[col_start:col_end, 0:OUTPUT_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, core_y = ttl.core(dims=2)

        # Workers send via pipes
        if core_x == 1:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe1)
                tx.wait()
        elif core_x == 2:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe2)
                tx.wait()
        elif core_x == 3:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe3)
                tx.wait()
        elif core_x == 4:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe4)
                tx.wait()
        elif core_x == 5:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe5)
                tx.wait()
        elif core_x == 6:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe6)
                tx.wait()
        elif core_x == 7:
            with gather_cb.wait() as blk:
                tx = ttl.copy(blk, pipe7)
                tx.wait()

        # Coordinator receives from all pipes then writes output
        if core_x == 0:
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe1, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe2, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe3, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe4, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe5, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe6, blk)
                tx.wait()
            with gather_cb.reserve() as blk:
                tx = ttl.copy(pipe7, blk)
                tx.wait()

            with out_cb.wait() as blk:
                tx = ttl.copy(blk, out[0:BATCH_TILES, 0:OUTPUT_TILES])
                tx.wait()


def to_ttnn(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
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
    X_test = (X_test - 0.1307) / 0.3081

    print("=== TT-Lang MNIST Inference (Fully Fused + Pipes) ===")
    print("NOTE: This version hits a simulator limitation.")
    print("      May work on real hardware.")
    print()

    device = ttnn.open_device(device_id=0)

    scaler = torch.ones(32, 32, dtype=torch.bfloat16)
    out_buf = torch.zeros(BATCH, OUTPUT_DIM, dtype=torch.bfloat16)

    # Convert weights to TTNN (only once)
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
        out_tt = to_ttnn(out_buf, device)

        # Run single fused kernel
        mnist_fused_kernel(x_tt, w1_tt, b1_tt, w2_tt, b2_tt, scaler_tt, out_tt)

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
