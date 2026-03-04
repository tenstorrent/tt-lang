# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Pytest for the ttlang python bindings — TTL and TTCore dialects together.

from ttl.dialects import ttl
from ttl import ir as ttlang_ir
from ttl.dialects import ttcore


def test_ttl_and_ttcore_same_context():
    with ttlang_ir.Context() as ctx, ttlang_ir.Location.unknown():
        ttl.ensure_dialects_registered(ctx)
        tile = ttcore.ir.TileType.get(ctx, 32, 32, 2)
        sl = ttl.SliceAttr.get(ctx, 0, 8, 2)

        assert str(tile) == "!ttcore.tile<32x32, bf16>"
        assert str(sl) == "#ttl.slice<start = 0, stop = 8, step = 2>"
