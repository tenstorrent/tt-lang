# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Multi-tile matmul with K-dimension accumulation.

Tests [1,2] @ [2,1] = [1,1] where the K dimension has 2 tiles.
The compiler must generate a K-loop that accumulates in DST.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl

# CHECK-LABEL: func.func @compute_fn
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}
# CHECK: ttl.matmul

# K-loop should generate matmul_tiles inside an scf.for
# CHECK-CPP: // compute_fn
# CHECK-CPP: void kernel_main()
# CHECK-CPP: mm_init(
# CHECK-CPP: matmul_tiles(
# CHECK-CPP: pack_tile


@ttl.kernel(grid=(1, 1))
def matmul_k_kernel(a, b, c):
    # A: [M=1, K=2], B: [K=2, N=1], C: [M=1, N=1]
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 2), buffer_factor=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), buffer_factor=2)
    c_cb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as av, b_cb.wait() as bv, c_cb.reserve() as out:
            out.store(ttl.math.matmul(av, bv))

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:2], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0:2, 0:1], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_cb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()


device = ttnn.open_device(device_id=0)
torch = __import__("torch")
a = ttnn.from_torch(
    torch.randn(32, 64, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
b = ttnn.from_torch(
    torch.randn(64, 32, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
c = ttnn.from_torch(
    torch.zeros(32, 32, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
matmul_k_kernel(a, b, c)
ttnn.close_device(device)
