# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Constants used throughout the DSL."""

DEFAULT_TILE_SIZE = 32
DEFAULT_TILE_SHAPE = [32, 32]
DEFAULT_GRID = (1, 1)
SUPPORTED_MEMORY_SPACES = frozenset(["L1", "DRAM"])
SUPPORTED_KERNEL_TYPES = frozenset(["compute", "datamovement", "noc"])
