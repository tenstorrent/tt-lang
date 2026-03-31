# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple matmul kernel - verifies matmul lowers to correct TTL ops and C++ code.

Tests single-tile matmul: [1,1] @ [1,1] = [1,1] using the @ operator.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl

# CHECK-LABEL: func.func @matmul_compute
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}
# CHECK: %[[A_CB:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[B_CB:.+]] = ttl.bind_cb{cb_index = 1
# CHECK: %[[C_CB:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: ttl.cb_wait %[[A_CB]]
# CHECK: ttl.cb_wait %[[B_CB]]
# CHECK: ttl.cb_reserve %[[C_CB]]
# CHECK: ttl.matmul
# CHECK: ttl.store
# CHECK: ttl.cb_push %[[C_CB]]
# CHECK: ttl.cb_pop %[[B_CB]]
# CHECK: ttl.cb_pop %[[A_CB]]

# CHECK-CPP: // matmul_compute
# CHECK-CPP: void kernel_main()
# CHECK-CPP: mm_init(
# CHECK-CPP: matmul_tiles(
# CHECK-CPP: pack_tile


@ttl.kernel(grid=(1, 1))
def matmul_kernel(a, b, c):
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def matmul_compute():
        with a_cb.wait() as av, b_cb.wait() as bv, c_cb.reserve() as out:
            out.store(ttl.math.matmul(av, bv))

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_cb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()


device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
b = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
c = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
matmul_kernel(a, b, c)
ttnn.close_device(device)
