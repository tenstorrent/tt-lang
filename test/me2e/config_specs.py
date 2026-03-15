# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration specifications for declarative ME2E tests.

Defines TestConfig dataclass and CONFIGS registry for test configurations.
This enables declarative testing where configurations are specified as data.
"""

from dataclasses import dataclass

import torch

from .config import E2EConfig, MemoryLayout

# Expected failures keyed by partial parameter match (config/dtype/op level).
# For per-op xfails, see ops/XFAILS.py (uses fully-qualified test IDs).
#
# Each key is a tuple of (config_str, dtype_str, op_name). A None element
# matches all values for that position. Trailing None elements can be omitted.
# Examples:
#   ("8x8_bfloat16_buf2_interleaved", "float32"):  all ops, one config+dtype
#   ("8x8_bfloat16_buf2_interleaved", "float32", "add"):  single combination
#   ("8x8_bfloat16_buf2_interleaved",):  all dtypes and ops for that config
XFAILS = {
    ("8x8_bfloat16_buf2_interleaved", "float32"): (
        "f32 8x8 produces ~19M ULP delta; tile index lowering inaccurate for large grids"
    ),
}


@dataclass(frozen=True)
class TestConfig:
    """
    Complete test configuration for declarative ME2E tests.

    This dataclass encapsulates all configuration parameters needed to run a test
    case in the declarative ME2E test framework. It enables test configurations to
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
            - 1: Single buffering
            - 2: Double buffering (default, overlaps data movement with compute)
            Double buffering can improve performance but uses more L1 memory.

        memory_layout: Memory layout type for tensor storage. Must be a MemoryLayout
            enum value. Default is INTERLEAVED. Other options include HEIGHT_SHARDED,
            WIDTH_SHARDED, and BLOCK_SHARDED for distributed memory configurations.

        maximize_dst: Enable DST subblocking and operation scheduling passes.
            When False, uses basic loop lowering without subblocking. Default is True.

        enable_fpu_binary_ops: Enable FPU binary op detection for add/sub/mul.
            When False, all binary ops use the copy_tile + SFPU path. Default is True.

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
    buffer_factor: int = 2
    memory_layout: MemoryLayout = MemoryLayout.INTERLEAVED

    # Pipeline options.
    maximize_dst: bool = True
    enable_fpu_binary_ops: bool = True

    def __str__(self) -> str:
        """
        Compact string representation for test output.

        Format: block_h x block_w_dtype_buf{buffer_factor}_layout[_nodst][_sfpu]
        Examples:
            - 2x2_bf16_buf2_interleaved (2x2 grid, bfloat16, double buffer, interleaved)
            - 8x8_f32_buf2_interleaved (8x8 grid, float32, double buffer, interleaved)
            - 2x2_bf16_buf2_interleaved_nodst (maximize_dst disabled)
            - 2x2_bf16_buf2_interleaved_sfpu (FPU binary ops disabled)
        """
        # Short dtype name (bf16, f32, etc.)
        dtype_str = str(self.dtype).split(".")[-1]

        # Buffer factor (always explicit)
        buffer_str = f"_buf{self.buffer_factor}"

        # Layout indicator (always explicit, using enum value)
        layout_str = f"_{self.memory_layout.value}"

        # Pipeline mode suffix.
        pipeline_str = ""
        if not self.maximize_dst:
            pipeline_str += "_nodst"
        if not self.enable_fpu_binary_ops:
            pipeline_str += "_sfpu"

        return f"{self.block_h}x{self.block_w}_{dtype_str}{buffer_str}{layout_str}{pipeline_str}"

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
    # Maximize-DST disabled: no subblocking or scheduling (basic loop lowering).
    TestConfig(num_tiles=4, block_h=2, block_w=2, maximize_dst=False),
    # SFPU path: FPU binary detection disabled (all binary ops use copy_tile + SFPU).
    TestConfig(num_tiles=4, block_h=2, block_w=2, enable_fpu_binary_ops=False),
    # Both disabled: basic loop lowering with SFPU binary path.
    TestConfig(
        num_tiles=4,
        block_h=2,
        block_w=2,
        maximize_dst=False,
        enable_fpu_binary_ops=False,
    ),
    # 8x8 grid (f32 xfailed via XFAILS below).
    TestConfig(num_tiles=64, block_h=8, block_w=8),
    # TODO: Sharded memory layout requires additional work.
    # TestConfig(num_tiles=64, memory_layout=MemoryLayout.HEIGHT_SHARDED),
]
