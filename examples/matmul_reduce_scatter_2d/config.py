# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static decomposition for two-dimensional matmul reduce-scatter."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class MatmulReduceScatter2DConfig:
    mesh_shape: tuple[int, int]
    m_tiles: int
    k_tiles_per_group: int
    n_tiles_per_group: int
    compute_grid: tuple[int, int]
    m_block_tiles: int
    k_block_tiles: int
    n_block_tiles: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        object.__setattr__(self, "compute_grid", tuple(self.compute_grid))
        if len(self.mesh_shape) != 2 or min(self.mesh_shape) < 1:
            raise ValueError("mesh_shape must contain two positive extents")
        if self.k_group_count != 2:
            raise ValueError("2D matmul reduce-scatter currently requires two K groups")
        if len(self.compute_grid) != 2 or min(self.compute_grid) < 1:
            raise ValueError("compute_grid must contain two positive extents")

        tile_fields = {
            "m_tiles": self.m_tiles,
            "k_tiles_per_group": self.k_tiles_per_group,
            "n_tiles_per_group": self.n_tiles_per_group,
            "m_block_tiles": self.m_block_tiles,
            "k_block_tiles": self.k_block_tiles,
            "n_block_tiles": self.n_block_tiles,
        }
        for field_name, value in tile_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.k_tiles_per_group % self.k_block_tiles:
            raise ValueError("K tiles per group must be divisible by the K block")
        if self.n_tiles_per_group % (self.n_block_tiles * self.n_worker_count):
            raise ValueError(
                "N tiles per group must be divisible by the N block and N workers"
            )
        if self.output_m_tiles_per_group % self.m_block_tiles:
            raise ValueError("each output M shard must contain complete M blocks")

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    @property
    def k_group_count(self) -> int:
        return self.mesh_shape[0]

    @property
    def n_group_count(self) -> int:
        return self.mesh_shape[1]

    @property
    def m_worker_count(self) -> int:
        return self.compute_grid[0]

    @property
    def n_worker_count(self) -> int:
        return self.compute_grid[1]

    @property
    def padded_m_tiles(self) -> int:
        tiles_per_partition = (
            self.m_block_tiles * self.m_worker_count * self.k_group_count
        )
        return (
            (self.m_tiles + tiles_per_partition - 1) // tiles_per_partition
        ) * tiles_per_partition

    @property
    def output_m_tiles_per_group(self) -> int:
        return self.padded_m_tiles // self.k_group_count

    @property
    def m_round_count(self) -> int:
        return self.output_m_tiles_per_group // (
            self.m_block_tiles * self.m_worker_count
        )

    @property
    def n_round_count(self) -> int:
        return self.n_tiles_per_group // (self.n_block_tiles * self.n_worker_count)

    @property
    def output_block_count(self) -> int:
        return self.m_round_count * self.n_round_count
