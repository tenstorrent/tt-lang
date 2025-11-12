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
# TODO: Is there a reason to expose TILE_SIZE instead of exposing only TILE_SHAPE?
#       Should TILE_SIZE in isolation ever be useful?
#       Do we ever plan on making tile shapes that have different height and width (or other dimensions)?
#       Should this be a used defined option?
TILE_SIZE = 32  # Standard tile dimensions (32x32)
TILE_SHAPE = (TILE_SIZE, TILE_SIZE)  # Standard tile shape (32x32)
