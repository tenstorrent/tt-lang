# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-side parallel argmax for Qwen decode.

Three-kernel approach — all heavy work stays on device:
  1. parallel_max_reduce_kernel: 110-core reduce_max across tile columns
  2. global_max_reduce_kernel: single-core reduce to find global max
  3. (future) parallel_index_find_kernel: parallel element scan for index

Host wrapper does only a tiny readback for the final result.

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


@ttl.kernel(grid=(1, 1))
def global_max_reduce_kernel(per_core_max, scaler, global_max):
    """Reduce per-core max values to a single global max + broadcast.

    Reads N tiles (one per core from kernel 1), accumulates element-wise
    max, then broadcasts the scalar max to fill the output tile.

    per_core_max: [32, N_CORES * 32] — one tile per core from kernel 1
    scaler:       [32, 32]           — ones tile (for reduce_max)
    global_max:   [32, 32]           — output: global max broadcast to all positions
    """
    Nt = per_core_max.shape[1] // TILE  # number of core tiles

    in_dfb = ttl.make_dataflow_buffer_like(per_core_max, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)
    bc_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        for col in range(Nt):
            with in_dfb.reserve() as blk:
                tx = ttl.copy(per_core_max[0, col], blk)
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

            # Remaining tiles
            for _ in range(Nt - 1):
                with in_dfb.wait() as in_blk:
                    with tmp_dfb.reserve() as tmp:
                        tmp.store(
                            ttl.math.reduce_max(in_blk, sc_blk, tmp, dims=[1])
                        )
                with tmp_dfb.wait() as reduced, acc_dfb.wait() as prev:
                    with acc_dfb.reserve() as acc:
                        acc.store(ttl.math.max(prev, reduced))

        # Broadcast scalar max to fill entire tile
        with acc_dfb.wait() as final:
            with bc_dfb.reserve() as bc:
                bc.store(ttl.math.broadcast(final, bc, dims=[0, 1]))
        with bc_dfb.wait() as bc_blk:
            with out_dfb.reserve() as out:
                out.store(bc_blk)

    @ttl.datamovement()
    def write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, global_max[0, 0])
            tx.wait()


@ttl.kernel(grid=(GRID_Y, GRID_X))
def parallel_index_find_kernel(logits, global_max, index_out):
    """Parallel element scan: find the column index of the global max value.

    Each core scans its chunk of tile columns, comparing row-0 elements
    against the global max value.  The DM write thread does element-level
    reads and comparison — no compute thread work needed.

    logits:     [32, V_padded]      — input logits
    global_max: [32, 32]            — broadcast global max (from kernel 2)
    index_out:  [32, N_CORES * 32]  — per-core result index
    """
    Nt = logits.shape[1] // TILE

    y_size, x_size = ttl.grid_size(dims=2)
    num_cores = y_size * x_size
    chunk = (Nt + num_cores - 1) // num_cores

    in_dfb = ttl.make_dataflow_buffer_like(logits, shape=(1, 1), buffer_factor=2)
    # Compute-local CB: passes tiles from compute to write thread
    pass_dfb = ttl.make_dataflow_buffer_like(logits, shape=(1, 1), buffer_factor=2)
    mx_dfb = ttl.make_dataflow_buffer_like(global_max, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(index_out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk

        with mx_dfb.reserve() as blk:
            tx = ttl.copy(global_max[0, 0], blk)
            tx.wait()
        for tid in range(chunk):
            col = (tile_start + tid) % Nt
            with in_dfb.reserve() as blk:
                tx = ttl.copy(logits[0, col], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        # Pass mx_dfb through and copy in_dfb → pass_dfb for the write thread
        with mx_dfb.wait() as blk:
            pass
        for _ in range(chunk):
            with in_dfb.wait() as in_blk:
                with pass_dfb.reserve() as out_blk:
                    out_blk.store(in_blk)
        for _ in range(1):
            with out_dfb.reserve() as oblk:
                pass

    @ttl.datamovement()
    def write():
        node_y, node_x = ttl.node(dims=2)
        nid = node_y * x_size + node_x
        tile_start = nid * chunk

        # Read global max value (scalar from position [0,0])
        with mx_dfb.wait() as mx_blk:
            max_val = ttl.element_read(mx_blk, 0, 0)

        # Scan this core's tiles for the max value
        best_lo = 0
        best_hi = 0
        for tid in range(chunk):
            col = (tile_start + tid) % Nt
            with pass_dfb.wait() as blk:
                for c in range(32):
                    val = ttl.element_read(blk, 0, c)
                    if val == max_val:
                        # Store tile col and local col separately to
                        # avoid uint16 overflow in element_write
                        best_lo = c
                        best_hi = col

        # Write found index as two uint16 values: [0,0]=low, [0,1]=high
        with out_dfb.reserve() as oblk:
            ttl.element_write(oblk, 0, 0, best_lo)
            ttl.element_write(oblk, 0, 1, best_hi)
            tx = ttl.copy(oblk, index_out[0, nid])
            tx.wait()
            oblk.pop()


class DeviceArgmax:
    """Fully on-device argmax — three kernels, one tiny host readback.

    Pipeline:
      1. parallel_max_reduce_kernel: 110-core reduce_max → per-core max values
      2. global_max_reduce_kernel: 1-core reduce → global max + broadcast
      3. parallel_index_find_kernel: 110-core element scan → per-core found index
      4. Host: read per-core indices (tiny), pick the valid one
    """

    def __init__(self, device, vocab_size=151936):
        self.device = device
        self.vocab_size = vocab_size
        self.vocab_padded = math.ceil(vocab_size / TILE) * TILE
        self.Nt = self.vocab_padded // TILE
        self.num_cores = GRID_Y * GRID_X
        self.chunk = (self.Nt + self.num_cores - 1) // self.num_cores

        def _alloc(shape):
            t = torch.zeros(*shape, dtype=torch.bfloat16)
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def _host(shape):
            t = torch.zeros(*shape, dtype=torch.bfloat16)
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        out_cols = self.num_cores * TILE
        self.scaler_dev = _alloc((TILE, TILE))
        # Fill scaler with ones
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.ones(TILE, TILE, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT),
            self.scaler_dev)
        self.max_out_dev = _alloc((TILE, out_cols))
        self.global_max_dev = _alloc((TILE, TILE))
        self.index_out_dev = _alloc((TILE, out_cols))
        self._index_out_host = _host((TILE, out_cols))

        # Warmup: compile all three kernels
        dummy_dev = _alloc((TILE, self.vocab_padded))
        parallel_max_reduce_kernel(dummy_dev, self.scaler_dev, self.max_out_dev)
        global_max_reduce_kernel(self.max_out_dev, self.scaler_dev, self.global_max_dev)
        parallel_index_find_kernel(dummy_dev, self.global_max_dev, self.index_out_dev)
        ttnn.deallocate(dummy_dev)

    def __call__(self, logits_dev):
        """Find argmax of row 0 of logits_dev on device.

        Returns: int token index
        """
        # All three kernels run on device back-to-back
        parallel_max_reduce_kernel(logits_dev, self.scaler_dev, self.max_out_dev)
        global_max_reduce_kernel(self.max_out_dev, self.scaler_dev, self.global_max_dev)
        parallel_index_find_kernel(logits_dev, self.global_max_dev, self.index_out_dev)

        # Tiny readback: per-core indices (bf16-encoded in tile [0,0] positions)
        ttnn.copy_device_to_host_tensor(self.index_out_dev, self._index_out_host)
        # element_write stores local_col at [0,0] and tile_col at [0,1]
        # per core to avoid uint16 overflow.
        idx_bf16 = self._index_out_host.to_torch().to(torch.bfloat16)
        idx_raw = idx_bf16.view(torch.int16).to(torch.int64)
        local_cols = idx_raw[0, ::TILE].numpy()[:self.num_cores] & 0xFFFF
        tile_cols = idx_raw[0, 1::TILE].numpy()[:self.num_cores] & 0xFFFF
        # Reconstruct global indices
        core_indices = tile_cols * TILE + local_cols
        # Filter: valid results have tile_col > 0 or local_col > 0
        valid_mask = (tile_cols > 0) | (local_cols > 0)
        if valid_mask.any():
            return int(core_indices[valid_mask].min())
        return 0


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
