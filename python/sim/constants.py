# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the dfbsim module.
"""

from .typedefs import Shape

# Private tile size - use TILE_SHAPE in external code
_TILE_SIZE = 32  # Standard tile dimensions (32x32)
# TODO: Should this be a user defined option?
TILE_SHAPE: Shape = (_TILE_SIZE, _TILE_SIZE)  # Standard tile shape (32x32)
