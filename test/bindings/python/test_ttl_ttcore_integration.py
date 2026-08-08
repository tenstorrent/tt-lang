# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Pytest for the ttlang python bindings -- TTL, TTCore, and TTKernel dialects
# together.

import pytest

from ttl import ir as ttlang_ir
from ttl import passes as ttlang_passes
from ttl.dialects import ttl as ttl_dialect
from ttl.dialects import ttcore
from ttl.dialects import ttkernel


def test_ttl_ttcore_and_ttkernel_same_context():
    with ttlang_ir.Context() as ctx, ttlang_ir.Location.unknown():
        ttl_dialect.ensure_dialects_registered(ctx)
        tile = ttcore.ir.TileType.get(ctx, 32, 32, 2)
        memref = ttlang_ir.MemRefType.get([2], tile)
        cb_type = ttkernel.ir.CBType.get(ctx, memref)
        thread_attr = ttkernel.ir.ThreadTypeAttr.get(ctx, "compute")
        sl = ttl_dialect.SliceAttr.get(ctx, 0, 8, 2)

        assert str(tile) == "!ttcore.tile<32x32, bf16>"
        assert str(cb_type) == "!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>"
        assert str(thread_attr) == "#ttkernel.thread<compute>"
        assert str(sl) == "#ttl.slice<start = 0, stop = 8, step = 2>"
        assert hasattr(ttlang_passes, "get_ttkernel_names")


def test_external_template_argument_validation():
    with ttlang_ir.Context() as ctx, ttlang_ir.Location.unknown():
        ttl_dialect.ensure_dialects_registered(ctx)
        with pytest.raises(ValueError, match="invalid external template argument"):
            ttl_dialect.ir.ExternalTemplateArgAttr.get(
                ctx, ttl_dialect.ir.ExternalTemplateArgKind.Boolean, 5
            )
