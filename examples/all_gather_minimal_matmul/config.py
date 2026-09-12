# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static decomposition for column-parallel all-gather matmul."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class AllGatherMinimalMatmulConfig:
    mesh_shape: tuple[int, int]
    m_tiles: int
    k_tiles_per_device: int
    n_tiles_per_device: int
    compute_grid: tuple[int, int]
    m_block_tiles: int = 1
    k_block_tiles: int = 1
    n_block_tiles: int = 1
    reuse_activation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        object.__setattr__(self, "compute_grid", tuple(self.compute_grid))
        if len(self.mesh_shape) != 2 or min(self.mesh_shape) < 1:
            raise ValueError("mesh_shape must contain two positive extents")
        if self.device_count < 2:
            raise ValueError("all-gather requires at least two devices")
        if len(self.compute_grid) != 2 or min(self.compute_grid) < 1:
            raise ValueError("compute_grid must contain two positive extents")

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

        for total_name, block_name in (
            ("k_tiles_per_device", "k_block_tiles"),
            ("n_tiles_per_device", "n_block_tiles"),
        ):
            if getattr(self, total_name) % getattr(self, block_name):
                raise ValueError(f"{total_name} must be divisible by {block_name}")

        if self.n_tiles_per_device % (self.n_block_tiles * self.n_workers):
            raise ValueError("N block count must be divisible by compute_grid[1]")
        if self.reuse_activation and self.activation_block_count > 32:
            raise ValueError(
                "activation reuse requires at most 32 global K blocks; increase "
                "k_block_tiles or disable reuse_activation"
            )

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    @property
    def m_workers(self) -> int:
        return self.compute_grid[0]

    @property
    def n_workers(self) -> int:
        return self.compute_grid[1]

    @property
    def padded_m_tiles(self) -> int:
        tiles_per_round = self.m_block_tiles * self.m_workers
        return (
            (self.m_tiles + tiles_per_round - 1) // tiles_per_round
        ) * tiles_per_round

    @property
    def activation_block_count(self) -> int:
        if not self.reuse_activation:
            return 2
        return self.device_count * self.k_tiles_per_device // self.k_block_tiles
