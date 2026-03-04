# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Site initialization for tt-mlir dialects (minimal build).
# This replaces tt-mlir's _site_initialize_0.py and registers only the
# dialects we build: TTCore, TTKernel, TTMetal.

from .._mlir_libs._ttmlir import register_dialects


def register_dialects(registry):
    """Called by MLIR site initialization to add tt-mlir dialects."""
    from .._mlir_libs._ttmlir import register_dialects as _register

    _register(registry)
