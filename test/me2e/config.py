# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration specifications for ME2E tests.

Defines test configurations including tile shapes, data types,
memory layouts, and buffering options.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch

ME2E_MAXIMUM_ULP_THRESHOLDS = {
    torch.float32: 2**14,
    torch.float16: 2**8,
    torch.bfloat16: 2**4,
}


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
class E2EConfig:
    """Complete test configuration for ME2E tests."""

    # Tile dimensions (fixed for Tenstorrent hardware).
    tile_h: int = 32
    tile_w: int = 32

    # Grid shape in tiles (rows, cols).
    grid_shape: Tuple[int, int] = (2, 2)

    # Data type.
    dtype: torch.dtype = torch.bfloat16

    # Buffer factor: 1=single buffer, 2=double buffer (default).
    buffer_factor: int = 2

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


# Minimal configuration for quick smoke tests.
SMOKE_CONFIGS = [E2EConfig(grid_shape=(1, 1))]

# Standard dtypes for testing.
TEST_DTYPES = [torch.bfloat16, torch.float32]

# Mapping from torch dtype to MLIR dtype string.
DTYPE_TO_MLIR = {
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}


def get_test_dtypes():
    """Get list of dtypes for pytest parametrization."""
    return list(DTYPE_TO_MLIR.keys())


def get_dtype_ids():
    """Get list of dtype IDs for pytest parametrization."""
    return [str(dt).split(".")[-1] for dt in DTYPE_TO_MLIR.keys()]


def get_maximum_ulp_threshold(dtype: torch.dtype) -> int:
    if golden.dtype in ME2E_MAXIMUM_ULP_THRESHOLDS:
        if ulp_threshold is None:
            ulp_threshold = ME2E_MAXIMUM_ULP_THRESHOLDS[dtype]
    else:
        raise ValueError(f"Unsupported dtype for ULP comparison: {dtype}")
    return ulp_threshold
