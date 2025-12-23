# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s | FileCheck %s
#
# Ensure ttl, ttcore, and ttnn dialects all work in the shared tt-mlir context.

from ttmlir import ir as tmlir_ir
from ttmlir.dialects import ttcore, ttnn
from ttlang.dialects import ttl

with tmlir_ir.Context() as ctx, tmlir_ir.Location.unknown():
    ttl.ensure_dialects_registered(ctx)

    tile = ttcore.ir.TileType.get(ctx, 32, 32, 2)
    sl = ttl.SliceAttr.get(ctx, 0, 4, 1)
    # Build a TTNN layout attr to ensure the dialect is usable.
    # shape=[1,1], grid=[1,1], core=[0,0], start/end offsets zeroed.
    mesh = ttnn.ir.MeshShapeAttr.get(ctx, 1, 1)
    core = ttnn.ir.CoreCoordAttr.get(ctx, 0, 0)

    # CHECK: !ttcore.tile<32x32, bf16>
    print(tile)
    # CHECK: #ttl.slice<start = 0, stop = 4, step = 1>
    print(sl)
    # CHECK: #ttnn<mesh_shape 1x1>
    print(mesh)
    # CHECK: #ttnn.core_coord<0, 0>
    print(core)
