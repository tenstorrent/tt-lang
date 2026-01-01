# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .._mlir_libs import _ttlang


def register_dialects(registry):
    """Called by MLIR site initialization to add TTL dialects to the registry."""
    _ttlang.register_dialects(registry)


def context_init_hook(ctx):
    """Optional post-init hook to force-load key tt-lang dialects."""
    # Ensure the dialect is loaded so attributes/types are usable immediately.
    _ = ctx.dialects["ttl"]
