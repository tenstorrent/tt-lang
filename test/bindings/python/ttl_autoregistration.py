# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s | FileCheck %s
#
# Verify ttl site-init registers dialects used by frontend-created IR.

from ttl.dialects import ttl
from ttl import ir as ttlang_ir

with ttlang_ir.Context() as ctx, ttlang_ir.Location.unknown():
    ttl.ensure_dialects_registered(ctx)
    _ = ctx.dialects["tensor"]
    s = ttl.SliceAttr.get(ctx, 0, 4, 1)
    # CHECK: #ttl.slice<start = 0, stop = 4, step = 1>
    print(s)
