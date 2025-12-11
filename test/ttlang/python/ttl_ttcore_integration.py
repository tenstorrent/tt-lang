# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s | FileCheck %s
#
# This lit-style test exercises ttl and ttcore in the same tt-mlir context.

from ttmlir import ir as tmlir_ir
from ttmlir.dialects import ttcore
from ttlang.dialects import ttl

with tmlir_ir.Context() as ctx, tmlir_ir.Location.unknown():
    ttl.ensure_dialects_registered(ctx)
    tile = ttcore.ir.TileType.get(ctx, 32, 32, 2)
    sl = ttl.SliceAttr.get(ctx, 0, 8, 2)

    # CHECK: !ttcore.tile<32x32, bf16>
    print(tile)
    # CHECK: #ttl.slice<start = 0, stop = 8, step = 2>
    print(sl)
