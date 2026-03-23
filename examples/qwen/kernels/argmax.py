# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-side parallel argmax for Qwen decode.

Multi-core reduce_max kernel finds per-core max values in parallel.
Host wrapper narrows down to winning tile range for final index lookup.

For decode: logits are [32, V_padded] bf16, only row 0 matters.
"""

import math
import time

import torch
import ttl
import ttnn

TILE = 32
GRID_Y = 11
GRID_X = 10


@ttl.kernel(grid=(GRID_Y, GRID_X))
def parallel_max_reduce_kernel(logits, scaler, max_out):
    """Parallel row-wise max reduction across tile columns.

    Each core processes a chunk of tile columns. For each tile,
    reduce_max(dims=[1]) gets per-row max, then element-wise max
    accumulates across tiles. Output: one tile per core.

    logits:  [32, V_padded]         — input logits (row 0 is decode output)
    scaler:  [32, 32]               — ones tile (required by reduce_max)
    max_out: [32, N_CORES * 32]     — per-core max values
    """
    Nt = logits.shape[1] // TILE  # total tile columns

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (Nt + num_cores - 1) // num_cores

    # Input CBs
    in_dfb = ttl.make_dataflow_buffer_like(logits, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)

    # Compute-local CBs for reduction
    tmp_dfb = ttl.make_dataflow_buffer_like(max_out, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(max_out, shape=(1, 1), buffer_factor=2)

    # Output CB
    out_dfb = ttl.make_dataflow_buffer_like(max_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk

        # Read scaler once
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

        # Read this core's chunk of tile columns (modulo wrap for excess)
        for tid in range(chunk):
            col = (tile_start + tid) % Nt
            with in_dfb.reserve() as blk:
                tx = ttl.copy(logits[0, col], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc_blk:
            # First tile: reduce_max → init accumulator
            with in_dfb.wait() as in_blk:
                with tmp_dfb.reserve() as tmp:
                    tmp.store(
                        ttl.math.reduce_max(in_blk, sc_blk, tmp, dims=[1])
                    )
            with tmp_dfb.wait() as reduced:
                with acc_dfb.reserve() as acc:
                    acc.store(reduced)

            # Remaining tiles: reduce_max → element-wise max with accumulator
            for _ in range(chunk - 1):
                with in_dfb.wait() as in_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(
                            ttl.math.reduce_max(in_blk, sc_blk, tmp, dims=[1])
                        )
                with tmp_dfb.wait() as reduced, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(ttl.math.max(prev, reduced))

        # Move to output CB
        with acc_dfb.wait() as final:
            with out_dfb.reserve() as out:
                out.store(final)

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, max_out[0, nid])
            tx.wait()


class DeviceArgmax:
    """Persistent device-side argmax with pre-allocated buffers."""

    def __init__(self, device, vocab_size=151936):
        self.device = device
        self.vocab_size = vocab_size
        self.vocab_padded = math.ceil(vocab_size / TILE) * TILE
        self.Nt = self.vocab_padded // TILE  # tile columns
        self.num_cores = GRID_Y * GRID_X  # 110
        self.chunk = (self.Nt + self.num_cores - 1) // self.num_cores

        # Pre-allocate scaler (ones tile)
        scaler_t = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        self.scaler_dev = ttnn.from_torch(
            scaler_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-allocate output: [32, num_cores * 32]
        out_cols = self.num_cores * TILE
        max_out_t = torch.zeros(TILE, out_cols, dtype=torch.bfloat16)
        self.max_out_dev = ttnn.from_torch(
            max_out_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernel
        dummy = torch.randn(TILE, self.vocab_padded, dtype=torch.bfloat16)
        dummy_dev = ttnn.from_torch(
            dummy,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        parallel_max_reduce_kernel(dummy_dev, self.scaler_dev, self.max_out_dev)
        ttnn.deallocate(dummy_dev)

    def __call__(self, logits_dev):
        """Find argmax of row 0 of logits_dev on device.

        Args:
            logits_dev: [32, vocab_padded] bf16 TILE tensor on device

        Returns:
            int: token index (argmax of row 0)
        """
        # Phase 1: parallel max reduction on device
        parallel_max_reduce_kernel(logits_dev, self.scaler_dev, self.max_out_dev)

        # Phase 2: read small output, find winning core
        max_host = ttnn.to_torch(self.max_out_dev).float()
        # Each core's max is at row 0, col core_id*32 (first element of its tile)
        core_maxes = max_host[0, ::TILE].numpy()[:self.num_cores]
        winning_core = int(core_maxes.argmax())

        # Phase 3: read back only the winning core's tile range
        tile_start = winning_core * self.chunk
        tile_end = min(tile_start + self.chunk, self.Nt)
        col_start = tile_start * TILE
        col_end = min(tile_end * TILE, self.vocab_size)

        winning_slice = ttnn.to_torch(
            logits_dev[0:1, col_start:col_end]
        ).float()
        local_idx = winning_slice[0, :col_end - col_start].argmax().item()

        return col_start + local_idx


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    VOCAB = 151936

    print("Opening device...")
    device = ttnn.open_device(device_id=0)

    try:
        print("Initializing DeviceArgmax (compiles kernel)...")
        t0 = time.perf_counter()
        dargmax = DeviceArgmax(device, vocab_size=VOCAB)
        print(f"  Init took {time.perf_counter() - t0:.2f}s")

        # ---- Correctness ----
        print(f"\n{'='*60}")
        print("Correctness tests")
        print(f"{'='*60}")

        VOCAB_PAD = math.ceil(VOCAB / TILE) * TILE
        mismatches = 0
        n_trials = 20
        for i in range(n_trials):
            t = torch.randn(TILE, VOCAB_PAD, dtype=torch.bfloat16)
            # Plant a known max at a random position in row 0
            known_idx = torch.randint(0, VOCAB, (1,)).item()
            t[0, known_idx] = 100.0

            dt = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            dev_idx = dargmax(dt)
            ref_idx = t[0, :VOCAB].float().argmax().item()
            ttnn.deallocate(dt)

            ok = "OK" if dev_idx == ref_idx else "MISMATCH"
            if dev_idx != ref_idx:
                mismatches += 1
                print(f"  Trial {i}: planted={known_idx} dev={dev_idx} ref={ref_idx} — {ok}")
            else:
                print(f"  Trial {i}: idx={dev_idx} — {ok}")

        if mismatches == 0:
            print(f"  PASS: all {n_trials} trials correct")
        else:
            print(f"  {mismatches}/{n_trials} mismatches")

        # ---- Timing ----
        print(f"\n{'='*60}")
        print("Timing (50 iterations)")
        print(f"{'='*60}")

        t = torch.randn(TILE, VOCAB_PAD, dtype=torch.bfloat16)
        dt = ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup
        for _ in range(5):
            _ = dargmax(dt)

        # Device argmax
        times_dev = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = dargmax(dt)
            times_dev.append(time.perf_counter() - t0)

        # Host baseline: full readback + argmax
        times_host = []
        for _ in range(50):
            t0 = time.perf_counter()
            h = ttnn.to_torch(dt).float()
            _ = h[0, :VOCAB].argmax().item()
            times_host.append(time.perf_counter() - t0)

        ttnn.deallocate(dt)

        avg_d = sum(times_dev) / len(times_dev) * 1e3
        min_d = min(times_dev) * 1e3
        avg_h = sum(times_host) / len(times_host) * 1e3
        min_h = min(times_host) * 1e3

        print(f"  Device argmax:              avg={avg_d:.2f}ms  min={min_d:.2f}ms")
        print(f"  Host readback + argmax:     avg={avg_h:.2f}ms  min={min_h:.2f}ms")
        print(f"  Speedup: {avg_h/avg_d:.2f}x (avg), {min_h/min_d:.2f}x (min)")

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)
        print("Done.")
