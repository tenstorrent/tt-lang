# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layout creation and manipulation utilities for tensor distribution across cores."""

from typing import List, Optional
from dataclasses import dataclass

from ttmlir.ir import *
from ttmlir.dialects import ttcore, d2m

from .constants import DEFAULT_TILE_SHAPE, SUPPORTED_MEMORY_SPACES


@dataclass(frozen=True)
class MetalLayoutConfig:
    """Immutable configuration for metal layout creation."""

    logical_shape: List[int]
    grid: List[int]
    tiled: bool = True
    memory_space: str = "L1"
    sharded: bool = True


@dataclass(frozen=True)
class StreamLayoutConfig:
    """Immutable configuration for stream layout creation."""

    logical_shape: List[int]
    grid: List[int]
    tiled: bool
    memory_space: str


def compute_device_shape(
    layout,
    grid: List[int],
    logical_shape: List[int],
    tile_shape: Optional[List[int]] = None,
) -> List[int]:
    """
    Compute device shape from layout attributes and grid configuration.

    The device shape combines grid dimensions with shard dimensions to represent
    how a tensor is distributed across cores. For a 2D tensor with grid [2, 2],
    the device shape will be [2, 2, shard_y, shard_x] where shard dimensions
    represent tiles per core.

    Args:
        layout: MetalLayoutAttr containing layout information
        grid: Grid dimensions [grid_y, grid_x]
        logical_shape: Logical tensor shape in elements
        tile_shape: Tile dimensions (default [32, 32])

    Returns:
        Device shape as list of integers

    Raises:
        RuntimeError: If layout cannot be downcast to MetalLayoutAttr
    """
    if tile_shape is None:
        tile_shape = DEFAULT_TILE_SHAPE

    logical_rank = len(logical_shape)
    if len(grid) == 2 and logical_rank == 2:
        grid_shape = list(grid)
    else:
        grid_shape = list(grid) + [1] * (logical_rank - len(grid))

    typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
    if typed_layout is None:
        raise RuntimeError("Failed to downcast MetalLayoutAttr")

    return typed_layout.getDeviceShape(grid_shape, tile_shape)


def create_metal_layout(ctx, config: MetalLayoutConfig) -> "ttcore.MetalLayoutAttr":
    """
    Create a MetalLayoutAttr from configuration.

    Args:
        ctx: MLIR context
        config: Immutable configuration containing layout parameters

    Returns:
        ttcore.MetalLayoutAttr with computed device shape

    Raises:
        ValueError: If memory_space is invalid or logical dimensions are not
                   divisible by grid dimensions
    """
    if config.memory_space == "L1":
        mem_space = ttcore.MemorySpace.DeviceL1
    elif config.memory_space == "DRAM":
        mem_space = ttcore.MemorySpace.DeviceDRAM
    else:
        raise ValueError(
            f"Invalid memory_space: {config.memory_space}. Must be 'L1' or 'DRAM'"
        )

    if config.sharded:
        memory_layout = ttcore.TensorMemoryLayout.Sharded
    else:
        memory_layout = ttcore.TensorMemoryLayout.Interleaved

    for i in range(len(config.logical_shape)):
        if config.logical_shape[i] % config.grid[i] != 0:
            raise ValueError(
                f"Logical dimension {i} ({config.logical_shape[i]}) must be evenly divisible by grid dimension {i} ({config.grid[i]})"
            )

    layout = ttcore.ir.MetalLayoutAttr.get(
        ctx,
        config.logical_shape,
        config.grid,
        int(ttcore.OOBVal.Undef),
        int(mem_space),
        int(ttcore.TensorMemoryLayout.Sharded),
    )

    return layout


def create_stream_layout_for_input(ctx, input_arg, config: StreamLayoutConfig):
    """
    Create a stream_layout op for the given input argument.

    Stream layouts create placeholder storage buffers for input tensors. The
    d2m-allocate pass creates new L1 allocations and uses the stream as a data
    source via stream_layout ops.

    Storage uses MetalLayoutAttr WITHOUT index_map (becomes ShardLayoutAttr after
    bufferization). Result uses MetalLayoutAttr WITH identity index_map (becomes
    ViewLayoutAttr after bufferization).

    Args:
        ctx: MLIR context
        input_arg: Function argument to wrap with stream layout
        config: Immutable configuration containing logical_shape, grid, tiled, memory_space

    Returns:
        Stream layout operation result

    Raises:
        RuntimeError: If input argument lacks MetalLayoutAttr encoding
    """
    input_type = input_arg.type

    input_tensor_type = RankedTensorType(input_type)
    device_shape = list(input_tensor_type.shape)
    element_type = input_tensor_type.element_type
    encoding = input_tensor_type.encoding

    metal_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(encoding)
    if metal_layout is None:
        raise RuntimeError("Input argument must have MetalLayoutAttr encoding")

    storage_layout = create_metal_layout(
        ctx,
        MetalLayoutConfig(
            logical_shape=config.logical_shape,
            grid=config.grid,
            tiled=config.tiled,
            memory_space=config.memory_space,
        ),
    )
    storage_type = RankedTensorType.get(device_shape, element_type, storage_layout)
    storage = d2m.EmptyOp(storage_type)

    rank = len(device_shape)
    identity_map = AffineMap.get_identity(rank, ctx)

    result_layout = ttcore.ir.MetalLayoutAttr.get(
        ctx,
        config.logical_shape,
        config.grid,
        int(ttcore.OOBVal.Undef),
        int(
            ttcore.MemorySpace.DeviceL1
            if config.memory_space == "L1"
            else ttcore.MemorySpace.DeviceDRAM
        ),
        int(ttcore.TensorMemoryLayout.Sharded),
        identity_map,
    )
    result_type = RankedTensorType.get(device_shape, element_type, result_layout)

    stream = d2m.StreamLayoutOp(result_type, input_arg, storage.result)
    return stream.result
