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


# Increment when the public descriptor or validator contract changes in an
# incompatible way. The simulator checks this before constructing validators so
# mixed tt-lang installations fail with an actionable message.
COMPILER_VALIDATION_API_VERSION = 1

_VALID_LAYOUTS = {"TILE", "ROW_MAJOR"}
_VALID_MEMORY_SPACES = {"L1", "DRAM"}
_REGULAR_SHARDED_LAYOUTS = {
    "HEIGHT_SHARDED",
    "WIDTH_SHARDED",
    "BLOCK_SHARDED",
}
_VALID_MEMORY_LAYOUTS = {"INTERLEAVED", "ND_SHARDED", *_REGULAR_SHARDED_LAYOUTS}
_VALID_SHARD_ORIENTATIONS = {"ROW_MAJOR", "COL_MAJOR"}
_VALID_ND_DISTRIBUTIONS = {"ROUND_ROBIN_1D", "GRID_2D"}


def _positive_dimensions(name: str, value: tuple[int, ...]) -> None:
    if any(dimension <= 0 for dimension in value):
        raise ValueError(f"{name} dimensions must be positive, got {value}")


def _validate_core_ranges(
    name: str, ranges: tuple[tuple[int, int, int, int], ...]
) -> None:
    for region in ranges:
        if len(region) != 4:
            raise ValueError(
                f"{name} entries must be (start_x, start_y, end_x, end_y), "
                f"got {region}"
            )
        start_x, start_y, end_x, end_y = region
        if min(region) < 0 or start_x > end_x or start_y > end_y:
            raise ValueError(f"{name} contains an invalid core range {region}")


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
        dimension_fields = (
            "shape",
            "padded_shape",
            "tile_shape",
            "shard_shape",
            "shard_grid",
            "nd_shard_shape",
            "nd_shard_grid",
            "mesh_shape",
        )
        for name in dimension_fields:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, tuple(int(dimension) for dimension in value)
                )
        if self.mesh_dims is not None:
            object.__setattr__(
                self,
                "mesh_dims",
                tuple(
                    None if dimension is None else int(dimension)
                    for dimension in self.mesh_dims
                ),
            )
        if self.padded_shape is None:
            object.__setattr__(self, "padded_shape", self.shape)
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
        self._validate()

    def _validate(self) -> None:
        _positive_dimensions("shape", self.shape)
        assert self.padded_shape is not None
        _positive_dimensions("padded_shape", self.padded_shape)
        if len(self.padded_shape) != len(self.shape):
            raise ValueError(
                "padded_shape must have the same rank as shape, got "
                f"{self.padded_shape} and {self.shape}"
            )
        if any(
            padded < logical for padded, logical in zip(self.padded_shape, self.shape)
        ):
            raise ValueError(
                f"padded_shape {self.padded_shape} cannot be smaller than shape {self.shape}"
            )
        if self.layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(_VALID_LAYOUTS)}, got {self.layout!r}"
            )
        if self.memory_space not in _VALID_MEMORY_SPACES:
            raise ValueError(
                "memory_space must be one of "
                f"{sorted(_VALID_MEMORY_SPACES)}, got {self.memory_space!r}"
            )
        if self.memory_layout not in _VALID_MEMORY_LAYOUTS:
            raise ValueError(
                "memory_layout must be one of "
                f"{sorted(_VALID_MEMORY_LAYOUTS)}, got {self.memory_layout!r}"
            )

        if self.layout == "TILE":
            if self.tile_shape is None or len(self.tile_shape) != 2:
                raise ValueError(
                    f"TILE layout requires a two-dimensional tile_shape, got {self.tile_shape}"
                )
            _positive_dimensions("tile_shape", self.tile_shape)
        elif self.tile_shape is not None or self.tile_size_bytes is not None:
            raise ValueError(
                "ROW_MAJOR layout must not include tile_shape or tile_size_bytes"
            )
        if self.tile_size_bytes is not None and self.tile_size_bytes <= 0:
            raise ValueError(
                f"tile_size_bytes must be positive, got {self.tile_size_bytes}"
            )

        regular_shard_metadata = (
            self.shard_shape is not None
            or self.shard_grid is not None
            or self.shard_orientation is not None
            or bool(self.shard_core_ranges)
        )
        nd_shard_metadata = (
            self.nd_shard_shape is not None
            or self.nd_shard_grid is not None
            or self.nd_shard_distribution is not None
            or bool(self.nd_shard_core_ranges)
            or self.nd_shard_num_cores is not None
        )
        if self.memory_layout in _REGULAR_SHARDED_LAYOUTS:
            if self.shard_shape is None:
                raise ValueError(
                    f"{self.memory_layout} requires regular shard_shape metadata"
                )
        elif regular_shard_metadata:
            raise ValueError(
                f"{self.memory_layout} cannot include regular shard metadata"
            )
        if self.memory_layout == "ND_SHARDED":
            if self.nd_shard_shape is None:
                raise ValueError("ND_SHARDED requires nd_shard_shape metadata")
        elif nd_shard_metadata:
            raise ValueError(f"{self.memory_layout} cannot include ND shard metadata")
        if self.nd_shard_num_cores is not None and self.nd_shard_num_cores <= 0:
            raise ValueError(
                f"nd_shard_num_cores must be positive, got {self.nd_shard_num_cores}"
            )
        if (
            self.shard_orientation is not None
            and self.shard_orientation not in _VALID_SHARD_ORIENTATIONS
        ):
            raise ValueError(
                "shard_orientation must be one of "
                f"{sorted(_VALID_SHARD_ORIENTATIONS)}, got {self.shard_orientation!r}"
            )
        if (
            self.nd_shard_distribution is not None
            and self.nd_shard_distribution not in _VALID_ND_DISTRIBUTIONS
        ):
            raise ValueError(
                "nd_shard_distribution must be one of "
                f"{sorted(_VALID_ND_DISTRIBUTIONS)}, got "
                f"{self.nd_shard_distribution!r}"
            )

        for name in (
            "shard_shape",
            "shard_grid",
            "nd_shard_shape",
            "nd_shard_grid",
            "mesh_shape",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_dimensions(name, value)
        _validate_core_ranges("shard_core_ranges", self.shard_core_ranges)
        _validate_core_ranges("nd_shard_core_ranges", self.nd_shard_core_ranges)

        if self.mesh_dims is not None:
            if self.mesh_shape is None or len(self.mesh_dims) != len(self.mesh_shape):
                raise ValueError(
                    "mesh_dims requires mesh_shape with the same rank, got "
                    f"{self.mesh_dims} and {self.mesh_shape}"
                )
            for dimension in self.mesh_dims:
                if dimension is not None and not 0 <= dimension < len(self.shape):
                    raise ValueError(
                        f"mesh dimension {dimension} is outside tensor rank {len(self.shape)}"
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


__all__ = [
    "COMPILER_VALIDATION_API_VERSION",
    "StaticTensorSpec",
    "build_operation_validator",
]
