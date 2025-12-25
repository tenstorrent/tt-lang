# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration specifications for E2E tests.

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


# Minimal grid shapes (fast tests).
MINIMAL_GRIDS = [(1, 1)]

# Small grid shapes (typical unit tests).
SMALL_GRIDS = [(1, 1), (2, 2), (1, 4), (4, 1)]

# Medium grid shapes (integration tests).
MEDIUM_GRIDS = [(4, 4), (2, 8), (8, 2)]

# Large grid shapes (stress tests).
LARGE_GRIDS = [(8, 8), (4, 16), (16, 4)]


def make_config(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.bfloat16,
    buffer_factor: int = 1,
    memory_layout: MemoryLayout = MemoryLayout.INTERLEAVED,
    buffer_type: BufferType = BufferType.DRAM,
) -> TestConfig:
    """Helper to create TestConfig with defaults."""
    return TestConfig(
        grid_shape=grid_shape,
        dtype=dtype,
        buffer_factor=buffer_factor,
        memory_layout=memory_layout,
        buffer_type=buffer_type,
    )


# Minimal configuration for quick smoke tests.
SMOKE_CONFIGS = [make_config((1, 1))]

# Standard test configurations.
CONFIGS = [
    make_config((1, 1)),  # Single tile.
    make_config((2, 2)),  # 2x2 grid.
    make_config((2, 2), buffer_factor=2),  # Double-buffered.
    make_config((2, 2), dtype=torch.float32),  # Different dtype.
    make_config((4, 4)),  # Larger grid.
]
