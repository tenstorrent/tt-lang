# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-grid output decomposition without opening devices."""

import pytest

from examples.all_gather_minimal_matmul import AllGatherMinimalMatmulConfig


@pytest.mark.parametrize("transpose", [False, True])
def test_fixed_worker_grid(transpose):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 1),
        m_tiles=4,
        k_tiles_per_device=4,
        n_tiles_per_device=6,
        worker_grid=(2, 3) if transpose else (3, 2),
        transpose=transpose,
    )
    assert config.m_workers == 2
    assert config.n_workers == 3
    assert config.m_tiles // (config.m_workers * config.m_block_tiles) == 2
    assert config.n_tiles_per_device // (config.n_workers * config.n_block_tiles) == 2


@pytest.mark.parametrize("transpose", [False, True])
def test_inferred_worker_grid(transpose):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 1),
        m_tiles=4,
        k_tiles_per_device=4,
        n_tiles_per_device=6,
        transpose=transpose,
    )
    assert config.grid == ((4, 6) if transpose else (6, 4))


@pytest.mark.parametrize(
    "worker_grid,error",
    [
        ((0, 2), "two extents"),
        ((2,), "two extents"),
        ((3, 3), "M block count"),
        ((4, 2), "N block count"),
    ],
)
def test_invalid_worker_grid(worker_grid, error):
    with pytest.raises(ValueError, match=error):
        AllGatherMinimalMatmulConfig(
            mesh_shape=(2, 1),
            m_tiles=4,
            k_tiles_per_device=4,
            n_tiles_per_device=6,
            worker_grid=worker_grid,
        )


@pytest.mark.parametrize("k_block_tiles", [1, 2, 4])
def test_compute_k_block(k_block_tiles):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 1),
        m_tiles=2,
        k_tiles_per_device=4,
        n_tiles_per_device=2,
        k_block_tiles=k_block_tiles,
    )
    assert config.compute_k_tiles == k_block_tiles
    assert config.activation_block_count == 8 // k_block_tiles


@pytest.mark.parametrize("k_block_tiles", [0, -1, 3, 8])
def test_invalid_compute_k_block(k_block_tiles):
    with pytest.raises(ValueError, match="k_block_tiles"):
        AllGatherMinimalMatmulConfig(
            mesh_shape=(2, 1),
            m_tiles=2,
            k_tiles_per_device=4,
            n_tiles_per_device=2,
            k_block_tiles=k_block_tiles,
        )


def test_streaming_bounds_activation_capacity():
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 1),
        m_tiles=2,
        k_tiles_per_device=64,
        n_tiles_per_device=2,
        reuse_activation=False,
    )
    assert config.activation_block_count == 2


def test_reuse_capacity_limit():
    with pytest.raises(ValueError, match="at most 32 K blocks"):
        AllGatherMinimalMatmulConfig(
            mesh_shape=(2, 1),
            m_tiles=2,
            k_tiles_per_device=64,
            n_tiles_per_device=2,
        )
