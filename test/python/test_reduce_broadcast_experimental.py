# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# EXPERIMENTAL: Test if we can broadcast directly in reduce by changing indexing maps
# This tests if the hardware tile_reduce_sum already produces broadcast values

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_reduce_broadcast_inline(A, ones, out):
    A_accessor = TensorAccessor(A)
    ones_accessor = TensorAccessor(ones)

    @compute()
    async def compute_reduce(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        from ttmlir.ir import *
        from ttmlir.dialects import d2m, linalg

        a = A_cb.wait()
        ones_val = ones_cb.wait()
        o = out_cb.reserve()

        # EXPERIMENTAL: Create reduce with broadcast output map
        # Instead of: input=(d0,d1), scaler=(0,0), output=(d0,0), iters=[parallel, reduction]
        # Try: input=(d0,d1), scaler=(0,0), output=(d0,d1), iters=[parallel, reduction]
        # This asks: "Does the hardware already broadcast within the tile?"

        ctx = a.type.context
        rank = 2

        # Standard maps
        identity_map = AffineMap.get_identity(rank, ctx)
        zero_map = AffineMap.get(2, 0, [AffineConstantExpr.get(0, ctx), AffineConstantExpr.get(0, ctx)], ctx)

        # EXPERIMENTAL: Use identity output map instead of collapsed map
        output_map = identity_map  # Write to all positions!

        out_type = RankedTensorType.get(
            list(a.type.shape), a.type.element_type, a.type.encoding
        )
        empty = d2m.empty(out_type)

        affine_maps_attr = ArrayAttr.get([
            AffineMapAttr.get(identity_map),  # Input: all positions
            AffineMapAttr.get(zero_map),      # Scaler: (0,0)
            AffineMapAttr.get(output_map)     # EXPERIMENTAL: Output to all positions
        ])

        # Keep reduction iterator type
        iter_types_attr = ArrayAttr.get([
            Attribute.parse("#linalg.iterator_type<parallel>", ctx),
            Attribute.parse("#linalg.iterator_type<reduction>", ctx)
        ])

        generic_op = linalg.GenericOp(
            result_tensors=[out_type],
            inputs=[a, ones_val],
            outputs=[empty],
            indexing_maps=affine_maps_attr,
            iterator_types=iter_types_attr,
        )

        block_arg_types = [a.type.element_type, ones_val.type.element_type, empty.type.element_type]
        block = generic_op.regions[0].blocks.append(*block_arg_types)

        with InsertionPoint(block):
            reduce_dim_attr = Attribute.parse("#d2m<reduce_dim R>", ctx)
            tile_result = d2m.tile_reduce_sum(
                a.type.element_type,
                block.arguments[0],
                block.arguments[1],
                block.arguments[2],
                reduce_dim_attr
            )
            linalg.YieldOp([tile_result])

        result = generic_op.result

        o.store(result)
        A_cb.pop()
        ones_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_loader(A_cb: CircularBuffer, ones_cb: CircularBuffer, out_cb: CircularBuffer):
        a_shard = A_cb.reserve()
        tx = dma(A_accessor[0, 0], a_shard)
        tx.wait()
        A_cb.push()

        ones_shard = ones_cb.reserve()
        tx = dma(ones_accessor[0, 0], ones_shard)
        tx.wait()
        ones_cb.push()

    return Program(compute_reduce, dm_loader)(A, ones, out)


# CHECK: func.func @test_reduce_broadcast_inline
# CHECK: "d2m.tile_reduce_sum"

# Test: Does tile_reduce_sum with broadcast output work?
A = torch.full((32, 32), 2.0)
ones = torch.ones((32, 32))
out = torch.zeros(32, 32)

print("=== EXPERIMENTAL TEST ===")
print(f"Testing if reduce can broadcast directly via output map")
print(f"Input: 32x32 tensor, all values = 2.0")
print(f"Expected if broadcast works: 64.0 at ALL positions")
print(f"Expected if broadcast fails: 64.0 only at column 0")

test_reduce_broadcast_inline(A, ones, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===

print(f"Output inspection:")
print(f"  out[0, 0] = {out[0, 0].item():.1f}")
print(f"  out[0, 15] = {out[0, 15].item():.1f}")
print(f"  out[0, 31] = {out[0, 31].item():.1f}")

row_0_all_same = torch.allclose(out[0, :], torch.full((32,), 64.0), rtol=0.01)

print(f"\nResult:")
if row_0_all_same:
    print(f"✅ BROADCAST WORKED: Hardware tile_reduce_sum broadcasts within tile!")
    print(f"   All columns have 64.0 - we can use this approach!")
    # CHECK-OUTPUT: BROADCAST WORKED
else:
    print(f"❌ BROADCAST FAILED: Only column 0 has 64.0")
    print(f"   Hardware doesn't broadcast - we need separate broadcast op")
    print(f"   Values: [{out[0, 0].item():.1f}, {out[0, 1].item():.1f}, {out[0, 2].item():.1f}, ...]")
