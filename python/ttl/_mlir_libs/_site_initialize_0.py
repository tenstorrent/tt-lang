# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Site initialization for tt-mlir dialects (minimal build).
# Registers TTCore, TTKernel, TTMetal dialects.
# Numbered _0 so it loads before _1 (TTL dialect).

from . import _ttmlir


def register_dialects(registry):
    """Called by MLIR site initialization to add tt-mlir dialects to the registry."""
    _ttmlir.register_dialects(registry)
