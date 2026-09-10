# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared tensor blocking and worker-grid configuration."""

from dataclasses import dataclass
from math import prod

MAX_DEVICE_COUNT = 32


@dataclass(frozen=True)
class AllGatherMinimalMatmulConfig:
    """Static tile and device decomposition for the fused operation."""

    mesh_shape: tuple[int, ...]
    m_tiles: int
    k_tiles_per_device: int
    n_tiles_per_device: int
    m_block_tiles: int = 1
    n_block_tiles: int = 1
    worker_grid: tuple[int, int] | None = None
    transpose: bool = False
    k_block_tiles: int = 1
    reuse_activation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        if not self.mesh_shape or any(extent <= 0 for extent in self.mesh_shape):
            raise ValueError("mesh_shape must contain positive extents")
        if self.device_count > MAX_DEVICE_COUNT:
            raise ValueError(f"all-gather supports at most {MAX_DEVICE_COUNT} devices")

        tile_fields = {
            "m_tiles": self.m_tiles,
            "k_tiles_per_device": self.k_tiles_per_device,
            "n_tiles_per_device": self.n_tiles_per_device,
            "m_block_tiles": self.m_block_tiles,
            "k_block_tiles": self.k_block_tiles,
            "n_block_tiles": self.n_block_tiles,
        }
        for field_name, value in tile_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        divisibility_constraints = (
            ("m_tiles", self.m_tiles, "m_block_tiles", self.m_block_tiles),
            (
                "k_tiles_per_device",
                self.k_tiles_per_device,
                "k_block_tiles",
                self.k_block_tiles,
            ),
            (
                "n_tiles_per_device",
                self.n_tiles_per_device,
                "n_block_tiles",
                self.n_block_tiles,
            ),
        )
        for total_name, total, block_name, block in divisibility_constraints:
            if total % block != 0:
                raise ValueError(f"{total_name} must be divisible by {block_name}")

        if self.reuse_activation and self.activation_block_count > 32:
            raise ValueError(
                "activation reuse requires at most 32 K blocks; increase "
                "k_block_tiles or disable reuse_activation"
            )

        if self.worker_grid is not None:
            object.__setattr__(self, "worker_grid", tuple(self.worker_grid))
            if len(self.worker_grid) != 2 or min(self.worker_grid) < 2:
                raise ValueError("worker_grid must have two extents of at least two")
        if (self.n_tiles_per_device // self.n_block_tiles) % self.n_workers:
            raise ValueError("N block count must be divisible by the N worker count")

        if min(self.grid) < 2:
            raise ValueError(
                "full-grid scheduling requires at least two M and two N workers"
            )

    @property
    def padded_m_tiles(self) -> int:
        round_tiles = self.m_block_tiles * self.m_workers
        return ((self.m_tiles + round_tiles - 1) // round_tiles) * round_tiles

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    @property
    def grid(self) -> tuple[int, int]:
        if self.worker_grid is not None:
            return self.worker_grid
        logical_grid = (
            self.n_tiles_per_device // self.n_block_tiles,
            self.m_tiles // self.m_block_tiles,
        )
        return logical_grid[::-1] if self.transpose else logical_grid

    @property
    def m_workers(self) -> int:
        return self.grid[0 if self.transpose else 1]

    @property
    def n_workers(self) -> int:
        return self.grid[1 if self.transpose else 0]

    @property
    def compute_k_tiles(self) -> int:
        return self.k_block_tiles

    @property
    def activation_block_count(self) -> int:
        if not self.reuse_activation:
            return 2
        return self.device_count * self.k_tiles_per_device // self.k_block_tiles
