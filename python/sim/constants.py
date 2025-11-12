# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the cbsim module.
"""

MAX_CBS = 32  # Fixed pool of circular buffers
# TODO: A better constant here would be the actual shape of the core grid
#       And it would be even better if it was a used defined variable
MAX_CORES = 4  # Maximum number of cores (assuming 2x2 grid for simplicity)
# Private tile size - use TILE_SHAPE in external code
_TILE_SIZE = 32  # Standard tile dimensions (32x32)
# TODO: Should this be a user defined option?
TILE_SHAPE = (_TILE_SIZE, _TILE_SIZE)  # Standard tile shape (32x32)
