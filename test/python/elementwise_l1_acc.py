# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Elementwise L1 packer accumulation: `out_blk += a_blk + b_blk` across K.

Pins the generated compute kernel structure for the non-matmul `+=` lowering:
- `llk_pack_reconfig_l1_acc(DISABLE)` before the K loop
- FPU binary `add_tiles` + `pack_tile` inside the loop body
- first-iteration `if (iv == lb)` guard containing
  `llk_pack_reconfig_l1_acc(ENABLE)`
- `cb_push_back` followed by `llk_pack_reconfig_l1_acc(DISABLE)` after the loop

`matmul_l1_acc_multinode.py` pins the equivalent pattern for the matmul RHS.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

from utils.correctness import assert_pcc

TILE = 32


@ttl.operation(grid=(1, 1))
def elementwise_add_acc(a, b, out):
    Kt = a.shape[0] // TILE
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        out_blk = out_dfb.reserve()
        for _ in range(Kt):
            a_blk = a_dfb.wait()
            b_blk = b_dfb.wait()
            out_blk += a_blk + b_blk
            a_blk.pop()
            b_blk.pop()
        out_blk.push()

    @ttl.datamovement()
    def reader():
        for kt in range(Kt):
            with a_dfb.reserve() as blk:
                ttl.copy(a[kt : kt + 1, 0:1], blk).wait()
            with b_dfb.reserve() as blk:
                ttl.copy(b[kt : kt + 1, 0:1], blk).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0:1, 0:1]).wait()


# =============================================================================
# C++ output: L1 packer accumulation pattern around the elementwise add loop.
# =============================================================================

# CHECK-CPP-DAG:  int32_t [[ENABLE:v[0-9]+]] = 1;
# CHECK-CPP-DAG:  int32_t [[DISABLE:v[0-9]+]] = 0;
# CHECK-CPP:      PACK((llk_pack_reconfig_l1_acc([[DISABLE]])));
# CHECK-CPP:      for
# CHECK-CPP:        add_tiles(
# CHECK-CPP:        pack_tile
# CHECK-CPP:        if (
# CHECK-CPP-NEXT:   PACK((llk_pack_reconfig_l1_acc([[ENABLE]])));
# CHECK-CPP:      cb_push_back(
# CHECK-CPP:      PACK((llk_pack_reconfig_l1_acc([[DISABLE]])));

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        Kt = 4
        a_torch = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
        b_torch = torch.randn(Kt * TILE, TILE, dtype=torch.bfloat16)
        golden = (a_torch.float() + b_torch.float()).reshape(Kt, TILE, TILE).sum(dim=0)

        a_dev = ttnn.from_torch(
            a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b_dev = ttnn.from_torch(
            b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out_dev = ttnn.from_torch(
            torch.zeros(TILE, TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        elementwise_add_acc(a_dev, b_dev, out_dev)

        result = ttnn.to_torch(out_dev).float()
        try:
            assert_pcc(golden, result)
            print("PASS")
        except AssertionError as e:
            print(f"FAIL: {e}")
    finally:
        ttnn.close_device(device)
