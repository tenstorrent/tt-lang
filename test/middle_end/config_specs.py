# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration specifications for middle-end tests.

Defines test configurations including tile shapes, data types,
memory layouts, and buffering options.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch


class MemoryLayout(Enum):
    """Memory layout types matching ttnn.TensorMemoryLayout."""

    INTERLEAVED = "interleaved"
    HEIGHT_SHARDED = "height_sharded"
    WIDTH_SHARDED = "width_sharded"
    BLOCK_SHARDED = "block_sharded"


class BufferType(Enum):
    """Buffer type matching ttnn.BufferType."""

    DRAM = "dram"
    L1 = "l1"


@dataclass(frozen=True)
class TestConfig:
    """Complete test configuration."""

    # Tile dimensions (fixed for Tenstorrent hardware).
    tile_h: int = 32
    tile_w: int = 32

    # Grid shape in tiles (rows, cols).
    grid_shape: Tuple[int, int] = (2, 2)

    # Data type.
    dtype: torch.dtype = torch.bfloat16

    # Buffer factor: 1=single buffer, 2=double buffer.
    buffer_factor: int = 1

    # Memory configuration.
    memory_layout: MemoryLayout = MemoryLayout.INTERLEAVED
    buffer_type: BufferType = BufferType.DRAM

    @property
    def num_tiles(self) -> int:
        """Total number of tiles in the grid."""
        return self.grid_shape[0] * self.grid_shape[1]

    @property
    def tensor_shape(self) -> Tuple[int, int]:
        """Tensor shape in elements (height, width)."""
        return (self.grid_shape[0] * self.tile_h, self.grid_shape[1] * self.tile_w)

    def __str__(self) -> str:
        layout = "db" if self.buffer_factor == 2 else "sb"
        dtype_str = str(self.dtype).split(".")[-1]
        return f"{self.grid_shape[0]}x{self.grid_shape[1]}_{dtype_str}_{layout}"


# Standard test configurations.
CONFIGS = [
    # Basic single-buffer configurations.
    TestConfig(grid_shape=(1, 1)),  # Single tile.
    TestConfig(grid_shape=(2, 2)),  # 2x2 grid.
    # Double-buffered configurations.
    TestConfig(grid_shape=(2, 2), buffer_factor=2),
    # Different data types.
    TestConfig(grid_shape=(2, 2), dtype=torch.float32),
    # Larger grids.
    TestConfig(grid_shape=(4, 4)),
]

# Minimal configuration for quick smoke tests.
SMOKE_CONFIGS = [
    TestConfig(grid_shape=(1, 1)),
]

# Extended configurations for thorough testing.
EXTENDED_CONFIGS = CONFIGS + [
    TestConfig(grid_shape=(8, 8)),
    TestConfig(grid_shape=(2, 2), memory_layout=MemoryLayout.HEIGHT_SHARDED),
]

