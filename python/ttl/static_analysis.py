# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only entry points for validating TT-Lang operations.

This module deliberately reuses the compiler frontend and validation pipeline.
It stops before TTKernel/EmitC lowering, runtime artifact generation, and all
device access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

from .atom import _operation_validator


@dataclass(frozen=True)
class _StaticShardSpec:
    shape: tuple[int, ...]
    shard_grid: Optional[tuple[int, ...]]
    orientation: Optional[str]
    core_ranges: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class _StaticNdShardSpec:
    shard_shape: tuple[int, ...]
    shard_grid: Optional[tuple[int, ...]]
    distribution: Optional[str]
    core_ranges: tuple[tuple[int, int, int, int], ...]
    num_cores: Optional[int]


@dataclass(frozen=True)
class _StaticMemoryConfig:
    buffer_type: str
    memory_layout: str
    shard_spec: Optional[_StaticShardSpec]
    nd_shard_spec: Optional[_StaticNdShardSpec]


@dataclass(frozen=True)
class _StaticTile:
    tile_shape: tuple[int, int]
    size_bytes: Optional[int]

    def get_tile_size(self, _dtype: Any) -> int:
        if self.size_bytes is None:
            raise ValueError("static tensor descriptor does not include tile byte size")
        return self.size_bytes


@dataclass(frozen=True)
class StaticTensorSpec:
    """Host-only tensor metadata consumed by compiler validation.

    The fields mirror the simulator and TTNN properties that can affect TTL IR,
    resource validation, or cache identity.  Defaults retain the original
    tiled/interleaved descriptor behavior for callers that only provide shape
    and dtype.
    """

    shape: tuple[int, ...]
    dtype: Any
    padded_shape: Optional[tuple[int, ...]] = None
    layout: str = "TILE"
    memory_space: str = "L1"
    memory_layout: str = "INTERLEAVED"
    tile_shape: Optional[tuple[int, int]] = (32, 32)
    tile_size_bytes: Optional[int] = None
    shard_shape: Optional[tuple[int, ...]] = None
    shard_grid: Optional[tuple[int, ...]] = None
    shard_orientation: Optional[str] = None
    shard_core_ranges: tuple[tuple[int, int, int, int], ...] = ()
    nd_shard_shape: Optional[tuple[int, ...]] = None
    nd_shard_grid: Optional[tuple[int, ...]] = None
    nd_shard_distribution: Optional[str] = None
    nd_shard_core_ranges: tuple[tuple[int, int, int, int], ...] = ()
    nd_shard_num_cores: Optional[int] = None
    mesh_shape: Optional[tuple[int, ...]] = None
    mesh_dims: Optional[tuple[Optional[int], ...]] = None
    _ttlang_static_tensor: ClassVar[bool] = True

    def __post_init__(self) -> None:
        tuple_fields = (
            "shape",
            "padded_shape",
            "tile_shape",
            "shard_shape",
            "shard_grid",
            "nd_shard_shape",
            "nd_shard_grid",
            "mesh_shape",
            "mesh_dims",
        )
        for name in tuple_fields:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        if self.padded_shape is None:
            object.__setattr__(self, "padded_shape", self.shape)
        else:
            object.__setattr__(
                self, "padded_shape", tuple(int(d) for d in self.padded_shape)
            )
        object.__setattr__(
            self,
            "shard_core_ranges",
            tuple(tuple(int(v) for v in region) for region in self.shard_core_ranges),
        )
        object.__setattr__(
            self,
            "nd_shard_core_ranges",
            tuple(
                tuple(int(v) for v in region) for region in self.nd_shard_core_ranges
            ),
        )

    def memory_config(self) -> _StaticMemoryConfig:
        shard_spec = None
        if self.shard_shape is not None:
            shard_spec = _StaticShardSpec(
                self.shard_shape,
                self.shard_grid,
                self.shard_orientation,
                self.shard_core_ranges,
            )
        nd_shard_spec = None
        if self.nd_shard_shape is not None:
            nd_shard_spec = _StaticNdShardSpec(
                self.nd_shard_shape,
                self.nd_shard_grid,
                self.nd_shard_distribution,
                self.nd_shard_core_ranges,
                self.nd_shard_num_cores,
            )
        return _StaticMemoryConfig(
            self.memory_space,
            self.memory_layout,
            shard_spec,
            nd_shard_spec,
        )

    def get_tile(self) -> _StaticTile:
        if self.tile_shape is None:
            raise ValueError("row-major tensor descriptor has no native tile")
        return _StaticTile(self.tile_shape, self.tile_size_bytes)

    def compiler_cache_key(self) -> tuple:
        """Return every represented tensor property as an immutable cache key."""
        return (
            self.shape,
            self.padded_shape,
            str(self.dtype),
            self.memory_space,
            self.memory_layout,
            self.layout,
            self.tile_shape,
            self.tile_size_bytes,
            self.shard_shape,
            self.shard_grid,
            self.shard_orientation,
            self.shard_core_ranges,
            self.nd_shard_shape,
            self.nd_shard_grid,
            self.nd_shard_distribution,
            self.nd_shard_core_ranges,
            self.nd_shard_num_cores,
            self.mesh_shape,
            self.mesh_dims,
        )


def build_operation_validator(
    function: Callable,
    *,
    grid,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    math_fidelity: Optional[str] = None,
    target_arch: Optional[str] = None,
) -> Callable:
    """Return a cached, host-only validator for ``function``.

    Calls accept :class:`StaticTensorSpec` objects in place of TTNN tensors.
    A successful call returns ``None``; compiler diagnostics are raised with
    the same source-aware errors as normal compilation.
    """
    return _operation_validator(
        grid=grid,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        math_fidelity=math_fidelity,
        target_arch=target_arch,
    )(function)


__all__ = ["StaticTensorSpec", "build_operation_validator"]
