# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Matmul K accumulation with the full K block resident in DST.

The output block fits DST capacity, so the generated compute kernel keeps the
partial products in DST for the complete K loop and packs the final output
block once.
"""

import torch
import ttl
import ttnn
from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
K_TILES = 8
N_TILES = 2


@ttl.operation(grid=(1, 1))
def matmul_dst_resident_acc(input_a, input_b, output):
    input_a_dfb = ttl.make_dataflow_buffer_like(
        input_a, shape=(1, K_TILES), block_count=2
    )
    input_b_dfb = ttl.make_dataflow_buffer_like(
        input_b, shape=(K_TILES, N_TILES), block_count=2
    )
    output_dfb = ttl.make_dataflow_buffer_like(
        output, shape=(1, N_TILES), block_count=2
    )

    @ttl.compute()
    def compute():
        with input_a_dfb.wait() as input_a_blk, input_b_dfb.wait() as input_b_blk:
            with output_dfb.reserve() as output_blk:
                output_blk.store(input_a_blk @ input_b_blk)

    @ttl.datamovement()
    def reader():
        with input_a_dfb.reserve() as input_a_blk:
            ttl.copy(input_a[0:1, 0:K_TILES], input_a_blk).wait()
        with input_b_dfb.reserve() as input_b_blk:
            ttl.copy(input_b[0:K_TILES, 0:N_TILES], input_b_blk).wait()

    @ttl.datamovement()
    def writer():
        with output_dfb.wait() as output_blk:
            ttl.copy(output_blk, output[0:1, 0:N_TILES]).wait()


# CHECK-CPP-LABEL: === compute kernel written to {{.*}} ===
# CHECK-CPP:       void kernel_main()
# CHECK-CPP-NOT:   llk_pack_reconfig_l1_acc
# CHECK-CPP:       tile_regs_acquire();
# CHECK-CPP-NOT:   pack_tile
# CHECK-CPP-NOT:   tile_regs_release
# CHECK-CPP:       for (size_t
# CHECK-CPP-NOT:   wait_front
# CHECK-CPP-NOT:   reserve_back
# CHECK-CPP-NOT:   push_back
# CHECK-CPP-NOT:   pop_front
# CHECK-CPP-NOT:   pack_tile
# CHECK-CPP:         matmul_block(
# CHECK-CPP-NOT:   wait_front
# CHECK-CPP-NOT:   reserve_back
# CHECK-CPP-NOT:   push_back
# CHECK-CPP-NOT:   pop_front
# CHECK-CPP-NOT:   pack_tile
# CHECK-CPP-NOT:   tile_regs_release
# CHECK-CPP:       tile_regs_commit();
# CHECK-CPP-NEXT:  tile_regs_wait();
# CHECK-CPP-NEXT:  pack_tile_block(
# CHECK-CPP-NEXT:  tile_regs_release();
# CHECK-CPP-NOT:   llk_pack_reconfig_l1_acc

# CHECK-RESULT: PASS


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        input_a_torch = torch.randn((TILE, K_TILES * TILE), dtype=torch.bfloat16)
        input_b_torch = torch.randn(
            (K_TILES * TILE, N_TILES * TILE), dtype=torch.bfloat16
        )
        output_torch = torch.zeros((TILE, N_TILES * TILE), dtype=torch.bfloat16)

        input_a = to_dram(input_a_torch, device)
        input_b = to_dram(input_b_torch, device)
        output = to_dram(output_torch, device)

        matmul_dst_resident_acc(input_a, input_b, output)

        expected = (input_a_torch.float() @ input_b_torch.float()).float()
        result = ttnn.to_torch(output).float()
        assert_pcc(expected, result, threshold=0.999)
        print("PASS")
    finally:
        ttnn.close_device(device)
