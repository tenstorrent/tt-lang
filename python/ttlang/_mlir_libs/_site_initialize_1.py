# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .._mlir_libs import _ttlang


def register_dialects(registry):
    """Called by MLIR site initialization to add TTL dialects to the registry."""
    _ttlang.register_dialects(registry)
