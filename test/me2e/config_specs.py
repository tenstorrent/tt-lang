# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration specifications for declarative E2E tests.

Defines TestConfig dataclass and CONFIGS registry for test configurations.
This enables declarative testing where configurations are specified as data.
"""

from dataclasses import dataclass

import torch

from .config import E2EConfig, MemoryLayout


@dataclass(frozen=True)
class TestConfig:
    """
    Complete test configuration for declarative E2E tests.

    This dataclass encapsulates all configuration parameters needed to run a test
    case in the declarative E2E test framework. It enables test configurations to
    be specified as data rather than code, allowing automatic test generation
    through pytest parametrization.

    TestConfig is designed to match the plan specification format, using block_h/block_w
    for grid dimensions and num_tiles for total tile count. It can be converted to
    E2EConfig (which uses grid_shape) via the to_e2e_config() method for use with
    existing infrastructure.

    All configs in CONFIGS are automatically parametrized in test_compute_ops.py,
    creating a test case for each (op, config) combination.

    Attributes:
        tile_h: Height of each tile in elements. Fixed at 32 for Tenstorrent hardware.
            Used to compute tensor dimensions from grid shape.

        tile_w: Width of each tile in elements. Fixed at 32 for Tenstorrent hardware.
            Used to compute tensor dimensions from grid shape.

        block_h: Grid height in number of tiles (rows). Determines how many tile rows
            are processed. Default is 8, giving an 8x8 grid (64 tiles total).

        block_w: Grid width in number of tiles (cols). Determines how many tile columns
            are processed. Default is 8, giving an 8x8 grid (64 tiles total).

        dtype: PyTorch data type for test tensors. Default is torch.bfloat16, which
            matches typical hardware execution. Can also be torch.float32 for
            higher precision testing.

        num_tiles: Total number of tiles in the grid (block_h * block_w). Used for
            documentation and validation. Default is 64 (8x8 grid). Note: This is
            a derived value - the actual grid shape is determined by block_h and block_w.

        buffer_factor: Circular buffer factor for double buffering.
            - 1: Single buffering (default)
            - 2: Double buffering (overlaps data movement with compute)
            Double buffering can improve performance but uses more L1 memory.

        memory_layout: Memory layout type for tensor storage. Must be a MemoryLayout
            enum value. Default is INTERLEAVED. Other options include HEIGHT_SHARDED,
            WIDTH_SHARDED, and BLOCK_SHARDED for distributed memory configurations.

    Examples:
        Basic configuration (8x8 grid, single buffering):
        >>> config = TestConfig()
        >>> config.block_h, config.block_w
        (8, 8)

        Large grid with double buffering:
        >>> config = TestConfig(
        ...     block_h=8,
        ...     block_w=16,
        ...     buffer_factor=2
        ... )
        >>> config.num_tiles
        128

        Configuration with sharded memory layout:
        >>> config = TestConfig(
        ...     memory_layout=MemoryLayout.HEIGHT_SHARDED
        ... )
    """

    tile_h: int = 32
    tile_w: int = 32
    block_h: int = 8
    block_w: int = 8
    dtype: torch.dtype = torch.bfloat16
    num_tiles: int = 64
    buffer_factor: int = 1
    memory_layout: MemoryLayout = MemoryLayout.INTERLEAVED

    def to_e2e_config(self) -> E2EConfig:
        """
        Convert to E2EConfig for use with existing infrastructure.

        TestConfig uses block_h/block_w for grid dimensions, while E2EConfig uses
        grid_shape tuple. This method performs the conversion, mapping:
        - block_h, block_w -> grid_shape
        - Other fields are passed through unchanged

        Returns:
            E2EConfig instance with equivalent configuration parameters.
        """
        return E2EConfig(
            grid_shape=(self.block_h, self.block_w),
            dtype=self.dtype,
            buffer_factor=self.buffer_factor,
            memory_layout=self.memory_layout,
        )


CONFIGS = [
    # Single tile config.
    TestConfig(num_tiles=1, block_h=1, block_w=1),  # 1x1 grid (single tile)
    # Multi-tile configs with loop generation.
    TestConfig(num_tiles=4, block_h=2, block_w=2),  # 2x2 grid (4 tiles)
    # TODO(#123): Enable 8x8 config once tile index lowering is fixed.
    # Currently fails with high ULP errors - tensor_slice indices don't correctly
    # map to tile offsets in the C++ lowering for grids larger than 2x2.
    # TestConfig(num_tiles=64, block_h=8, block_w=8),  # 8x8 grid (64 tiles)
    # TODO: Double buffering and sharded memory require additional work.
    # TestConfig(num_tiles=64, buffer_factor=2),  # Double buffering
    # TestConfig(num_tiles=64, memory_layout=MemoryLayout.HEIGHT_SHARDED),
]
