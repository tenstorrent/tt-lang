# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Large-trip tensor recurrence with forced in-DST accumulation.

The generated compute kernel must keep the accumulator in DST across all loop
iterations: acquire once, copy the initial value once, update the same DST slot
inside the loop, pack once, and release once.
"""

import torch

import ttnn
import ttl
from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

TILE = 32
N_ITERS = 32


@ttl.operation(grid=(1, 1))
def tensor_dst_acc(initial, delta, out):
    initial_dfb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
    delta_dfb = ttl.make_dataflow_buffer_like(delta, shape=(1, 1), block_count=N_ITERS)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with initial_dfb.wait() as acc:
            for _ in range(N_ITERS):
                with delta_dfb.wait() as delta_blk:
                    acc = acc + delta_blk

            with out_dfb.reserve() as out_blk:
                out_blk.store(acc)

    @ttl.datamovement()
    def reader():
        with initial_dfb.reserve() as initial_blk:
            ttl.copy(initial[0:1, 0:1], initial_blk).wait()
        for _ in range(N_ITERS):
            with delta_dfb.reserve() as delta_blk:
                ttl.copy(delta[0:1, 0:1], delta_blk).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:1, 0:1]).wait()


# CHECK-CPP:      === compute kernel written to {{.*}} ===
# CHECK-CPP:      void kernel_main()
# CHECK-CPP-NOT:  llk_pack_reconfig_l1_acc
# CHECK-CPP:      tile_regs_acquire();
# CHECK-CPP:      copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP:      copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP:      binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(get_compile_time_arg_val(1));
# CHECK-CPP:      for (size_t
# CHECK-CPP:        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(get_compile_time_arg_val(1),
# CHECK-CPP-NOT:  llk_pack_reconfig_l1_acc
# CHECK-CPP:      pack_tile<true>(
# CHECK-CPP:      tile_regs_release();
# CHECK-CPP-NOT:  llk_pack_reconfig_l1_acc

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        initial_torch = torch.full((TILE, TILE), 4.0, dtype=torch.bfloat16)
        delta_torch = torch.full((TILE, TILE), 0.25, dtype=torch.bfloat16)
        out_torch = torch.zeros((TILE, TILE), dtype=torch.bfloat16)
        expected = initial_torch.float() + N_ITERS * delta_torch.float()

        initial = to_dram(initial_torch, device)
        delta = to_dram(delta_torch, device)
        out = to_dram(out_torch, device)

        tensor_dst_acc(initial, delta, out, options="--ttl-accumulation-strategy=dst")
        result = ttnn.to_torch(out).float()
        assert_allclose(result, expected, rtol=5e-2, atol=1.0)
        print("PASS")
    finally:
        ttnn.close_device(device)
