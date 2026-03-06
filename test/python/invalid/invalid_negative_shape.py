# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: tensor shape dimensions must be positive.

Negative dimensions (e.g., -1) are common in Python but not supported
in tt-lang. This test verifies the validation catches them.
"""

from unittest.mock import MagicMock
from ttl._src.ttl_ast import _build_tensor_type


# CHECK: ValueError: All shape dimensions must be positive, got shape (-1, 32)
mock_tensor = MagicMock()
mock_tensor.shape = (-1, 32)
_build_tensor_type(None, mock_tensor, grid=[1, 1], tiled=True, memory_space="L1")
