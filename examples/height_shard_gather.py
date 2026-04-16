# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Height-sharded gather: demonstrates mixed local / remote L1 accesses.

A (64, 64)-element tensor is height-sharded across 2 cores so each core holds
one tile-row (32 x 64 elements) in L1.  Both cores run a "gather" that reads
the full tensor in a single ttl.copy call:

  core 0: inp[0:2, 0:2]  -- tile-row 0 is its own L1 (local),
                             tile-row 1 lives on core 1 (remote)
  core 1: inp[0:2, 0:2]  -- tile-row 0 lives on core 0 (remote),
                             tile-row 1 is its own L1 (local)

Every dm_read copy event therefore carries a 50/50 local_l1 / remote_l1 split
in the trace.  The gathered data passes through compute and is written to a
DRAM output tensor for verification.

Usage::

    ttlang-sim examples/height_shard_gather.py --trace /tmp/gather.jsonl
    ttlang-sim-stats /tmp/gather.jsonl

Expected Tensor Access Statistics (locality columns)::

    inp   2 reads  ...  Local L1 R: 4   Remote L1 R: 4   DRAM R: 0
    out   2 writes ...  Local L1 W: 0   Remote L1 W: 0   DRAM W: 8
"""

import torch
import ttl
import ttnn

TILE = 32
CORES = 2  # number of cores / tile-rows
COLS = 2  # tile-columns wide


@ttl.operation(grid=(CORES, 1))
def height_shard_gather(inp, out):
    # Two DFBs: one to receive the gathered input, one for the compute output.
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(CORES, COLS))
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(CORES, COLS))

    @ttl.compute()
    def compute():
        # In a real kernel: apply a cross-shard reduction or normalization.
        # Here we pass the gathered data through unchanged.
        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
            out_blk.store(in_blk)

    @ttl.datamovement()
    def dm_read():
        # Gather the full tensor. Each core reads all CORES tile-rows in one
        # copy: its own tile-row is served from local L1, the peer's from
        # remote L1.  The resulting trace event shows a 50/50 locality split.
        with in_dfb.reserve() as blk:
            tx = ttl.copy(inp[0:CORES, 0:COLS], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        # Write the gathered result back to DRAM. Both cores write the same
        # content (the full input), so the output is well-defined.
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:CORES, 0:COLS])
            tx.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        inp_mc = ttnn.create_sharded_memory_config(
            shape=(CORES * TILE, COLS * TILE),
            core_grid=ttnn.CoreGrid(y=1, x=CORES),
            strategy=ttnn.ShardStrategy.HEIGHT,
        )
        inp = ttnn.from_torch(
            torch.arange(CORES * TILE * COLS * TILE, dtype=torch.float32)
            .reshape(CORES * TILE, COLS * TILE)
            .to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=inp_mc,
        )
        out = ttnn.empty(
            (CORES * TILE, COLS * TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        height_shard_gather(inp, out)

        inp_torch = ttnn.to_torch(inp)
        out_torch = ttnn.to_torch(out)
        if torch.allclose(inp_torch.float(), out_torch.float()):
            print("height_shard_gather: PASS (output matches input)")
        else:
            max_err = (inp_torch.float() - out_torch.float()).abs().max().item()
            print(f"height_shard_gather: FAIL (max abs error = {max_err:.6f})")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
