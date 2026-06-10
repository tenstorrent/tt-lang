# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Dynamic indexed DRAM streaming: read a slot index from a tensor and use it
as a slice offset. This is the gather primitive behind routed-expert weight
streaming and KV-cache appends at a runtime position.
"""

import torch
import ttnn
import ttl

TILE = 32
SLOT_TILES = 2
NUM_SLOTS = 4


@ttl.operation(grid=(1, 1))
def gather_slot_kernel(idx_t, w, out):
    idx_dfb = ttl.make_dataflow_buffer_like(idx_t, shape=(1, 1), block_count=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(SLOT_TILES, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with idx_dfb.reserve() as blk:
            tx = ttl.copy(idx_t[0, 0], blk)
            tx.wait()
        idx_blk = idx_dfb.wait()
        idx = ttl.read_index(idx_blk, 0, 0)
        with w_dfb.reserve() as wblk:
            tx = ttl.copy(w[idx * SLOT_TILES : (idx + 1) * SLOT_TILES, 0:1], wblk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        wblk = w_dfb.wait()
        tx = ttl.copy(wblk, out[0:SLOT_TILES, 0:1])
        tx.wait()


# CHECK-LABEL: func.func @dm_read
# CHECK: ttl.raw_element_read
# CHECK: arith.fptosi
# CHECK: arith.index_cast
# CHECK: ttl.copy

# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()

device = ttnn.open_device(device_id=0)

idx_t = ttnn.from_torch(
    torch.full((TILE, TILE), 2.0, dtype=torch.float32),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
w = ttnn.from_torch(
    torch.randn(NUM_SLOTS * SLOT_TILES * TILE, TILE, dtype=torch.float32),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
out = ttnn.from_torch(
    torch.zeros(SLOT_TILES * TILE, TILE, dtype=torch.float32),
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

gather_slot_kernel(idx_t, w, out)

got = ttnn.to_torch(out)
want = ttnn.to_torch(w)[2 * SLOT_TILES * TILE : 3 * SLOT_TILES * TILE]
assert torch.equal(got, want), "indexed gather mismatch"
print("PASS")
ttnn.close_device(device)
