# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the cbsim module.
"""

MAX_CBS = 32  # Fixed pool of circular buffers
TILE_SIZE = 32  # Standard tile dimensions (32x32)
TILE_SHAPE = (TILE_SIZE, TILE_SIZE)  # Standard tile shape (32x32)
