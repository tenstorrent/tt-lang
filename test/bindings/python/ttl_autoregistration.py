# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s | FileCheck %s
#
# Verify ttl site-init registers the dialect so SliceAttr works without
# calling ensure_dialects_registered explicitly.

from ttmlir import ir as tmlir_ir
from ttlang.dialects import ttl

with tmlir_ir.Context() as ctx, tmlir_ir.Location.unknown():
    ttl.ensure_dialects_registered(ctx)
    s = ttl.SliceAttr.get(ctx, 0, 4, 1)
    # CHECK: #ttl.slice<start = 0, stop = 4, step = 1>
    print(s)
