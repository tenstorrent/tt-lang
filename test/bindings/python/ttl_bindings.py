# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

# Tests for TTL dialect Python bindings, using both ttmlir and ttlang modules

from ttlang.dialects import ttl
from ttmlir.dialects import ttcore
from ttmlir.ir import *

with Context() as ctx, Location.unknown():
    ttl.ensure_dialects_registered(ctx)

    print("=== Test SliceAttr Creation ===")
    # Test SliceAttr.get() with basic values (positional)
    slice_attr = ttl.SliceAttr.get(ctx, 0, 8, 1)

    # CHECK: #ttl.slice<start = 0, stop = 8, step = 1>
    print(slice_attr)

    print("\n=== Test SliceAttr with Different Values (keyword args) ===")
    slice_attr2 = ttl.SliceAttr.get(ctx, start=10, stop=20, step=2)

    # CHECK: #ttl.slice<start = 10, stop = 20, step = 2>
    print(slice_attr2)

    print("\n=== Test SliceAttr with Negative Step (mixed args) ===")
    slice_attr3 = ttl.SliceAttr.get(ctx, start=15, stop=0, step=-1)

    # CHECK: #ttl.slice<start = 15, stop = 0, step = -1>
    print(slice_attr3)

    print("\n=== Test Integration with ttmlir Types ===")
    # Create a ttcore TileType to verify ttlang works alongside ttmlir
    tile_type = ttcore.ir.TileType.get(ctx, 32, 32, 2)  # 32x32 BFloat16 tile

    # CHECK: !ttcore.tile<32x32, bf16>
    print(tile_type)

    # Create a tensor type with tiles
    tensor_type = RankedTensorType.get([2, 4], tile_type)

    # CHECK: tensor<2x4x!ttcore.tile<32x32, bf16>>
    print(tensor_type)

    print("\n=== Test Multiple SliceAttrs ===")
    # Create multiple slice attributes to test they can coexist
    slices = [
        ttl.SliceAttr.get(ctx, 0, 8, 1),
        ttl.SliceAttr.get(ctx, start=8, stop=16, step=1),
        ttl.SliceAttr.get(ctx, 16, stop=24, step=1),
    ]

    # CHECK: #ttl.slice<start = 0, stop = 8, step = 1>
    print(slices[0])
    # CHECK: #ttl.slice<start = 8, stop = 16, step = 1>
    print(slices[1])
    # CHECK: #ttl.slice<start = 16, stop = 24, step = 1>
    print(slices[2])
