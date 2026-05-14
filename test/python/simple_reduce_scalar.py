# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Simple reduce kernel with numeric scalar scalers.

Verifies numeric reduce scalers are materialized as 1x1 fill tensors before
being passed to ttl.reduce for scalar, COL, and ROW reductions over a multi-tile
input. The test then executes the kernel and checks the result against torch
golden tensors.
"""

import torch
import ttnn
import ttl
from ttlang_test_utils import assert_allclose


@ttl.operation(grid=(1, 1))
def reduce_scalar_kernel(inp, scalar_out, col_out, row_out):
    scalar_inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    col_inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), block_count=2)
    row_inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 1), block_count=2)
    scalar_out_dfb = ttl.make_dataflow_buffer_like(
        scalar_out, shape=(1, 1), block_count=2
    )
    col_out_dfb = ttl.make_dataflow_buffer_like(col_out, shape=(1, 2), block_count=2)
    row_out_dfb = ttl.make_dataflow_buffer_like(row_out, shape=(2, 1), block_count=2)

    @ttl.compute()
    def reduce_compute():
        with (
            scalar_inp_dfb.wait() as scalar_block,
            col_inp_dfb.wait() as col_block,
            row_inp_dfb.wait() as row_block,
            scalar_out_dfb.reserve() as scalar_out_block,
            col_out_dfb.reserve() as col_out_block,
            row_out_dfb.reserve() as row_out_block,
        ):
            scalar_out_block.store(ttl.math.reduce_sum(scalar_block, 0.5, dims=[0, 1]))
            col_out_block.store(ttl.math.reduce_sum(col_block, 1.25, dims=[0]))
            row_out_block.store(ttl.math.reduce_sum(row_block, -0.25, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with scalar_inp_dfb.reserve() as block:
            tx = ttl.copy(inp[0, 0], block)
            tx.wait()
        with col_inp_dfb.reserve() as block:
            tx = ttl.copy(inp[0, 0:2], block)
            tx.wait()
        with row_inp_dfb.reserve() as block:
            tx = ttl.copy(inp[0:2, 0], block)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with scalar_out_dfb.wait() as block:
            tx = ttl.copy(block, scalar_out[0, 0])
            tx.wait()
        with col_out_dfb.wait() as block:
            tx = ttl.copy(block, col_out[0, 0:2])
            tx.wait()
        with row_out_dfb.wait() as block:
            tx = ttl.copy(block, row_out[0:2, 0])
            tx.wait()


# CHECK-LABEL: func.func @reduce_compute
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}
# CHECK: %[[SCALAR_IN:.*]] = ttl.attach_cb
# CHECK: %[[COL_IN:.*]] = ttl.attach_cb
# CHECK: %[[ROW_IN:.*]] = ttl.attach_cb
# CHECK: %[[SCALAR_FILL:.*]] = ttl.fill 5.000000e-01 : tensor<1x1x!ttcore.tile<32x32, bf16>>
# CHECK: ttl.reduce %[[SCALAR_IN]], %[[SCALAR_FILL]] 0 : i32 [0, 1]
# CHECK: %[[COL_FILL:.*]] = ttl.fill 1.250000e+00 : tensor<1x1x!ttcore.tile<32x32, bf16>>
# CHECK: ttl.reduce %[[COL_IN]], %[[COL_FILL]] 0 : i32 [0]
# CHECK: %[[ROW_FILL:.*]] = ttl.fill -2.500000e-01 : tensor<1x1x!ttcore.tile<32x32, bf16>>
# CHECK: ttl.reduce %[[ROW_IN]], %[[ROW_FILL]] 0 : i32 [1]


device = ttnn.open_device(device_id=0)
try:
    inp_torch = torch.ones(64, 64, dtype=torch.bfloat16)
    scalar_out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)
    col_out_torch = torch.zeros(32, 64, dtype=torch.bfloat16)
    row_out_torch = torch.zeros(64, 32, dtype=torch.bfloat16)

    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    scalar_out = ttnn.from_torch(
        scalar_out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    col_out = ttnn.from_torch(
        col_out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    row_out = ttnn.from_torch(
        row_out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    reduce_scalar_kernel(inp, scalar_out, col_out, row_out)
    ttnn.synchronize_device(device)

    # REDUCE_SCALAR writes valid data only at output position [0, 0]; the
    # remaining DST positions hold undefined contents that get packed to L1
    # but are not part of the reduce result.
    scalar_result = ttnn.to_torch(scalar_out).float()
    scalar_expected = torch.tensor(
        inp_torch[:32, :32].float().sum().item() * 0.5, dtype=torch.float32
    )
    assert_allclose(scalar_result[0, 0], scalar_expected, rtol=0.01, atol=0.01)

    col_result = ttnn.to_torch(col_out).float()
    col_expected = inp_torch[:32, :].float().sum(dim=0) * 1.25
    assert_allclose(col_result[0, :], col_expected, rtol=0.01, atol=0.01)

    row_result = ttnn.to_torch(row_out).float()
    row_expected = inp_torch[:, :32].float().sum(dim=1) * -0.25
    assert_allclose(row_result[:, 0], row_expected, rtol=0.01, atol=0.01)
finally:
    ttnn.close_device(device)
