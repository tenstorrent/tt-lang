# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-grid output decomposition without opening devices."""

import pytest

from examples.all_gather_minimal_matmul import AllGatherMinimalMatmulConfig
from examples.all_gather_minimal_matmul.dedicated_communication.operation import (
    make_all_gather_minimal_matmul_operation as make_dedicated_operation,
)


def test_single_device_configuration():
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(1, 1),
        m_tiles=4,
        k_tiles_per_device=4,
        n_tiles_per_device=6,
    )

    assert config.device_count == 1
    assert config.activation_block_count == 4


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


@pytest.mark.parametrize("m_tiles,padded_tiles", [(96, 104), (104, 104)])
def test_full_device_row_padding(m_tiles, padded_tiles):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=(2, 2),
        m_tiles=m_tiles,
        k_tiles_per_device=40,
        n_tiles_per_device=10,
        m_block_tiles=4,
        k_block_tiles=8,
        worker_grid=(13, 10),
        transpose=True,
        reuse_activation=False,
    )
    assert config.m_workers * config.n_workers == 130
    assert config.padded_m_tiles == padded_tiles


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


@pytest.mark.parametrize(
    "mesh_shape,worker_grid,transpose,algorithm,communication_workers,error",
    [
        ((1, 2), (4, 5), False, "ring", 2, "transposed multi-device ring"),
        ((2, 1), (5, 4), True, "all_to_all", 2, "transposed multi-device ring"),
        ((2, 1), (5, 4), True, "ring", 4, "at most 2"),
        ((4, 1), (5, 4), True, "ring", 5, "communication column"),
    ],
)
def test_dedicated_communication_rejects_unsupported_resources(
    mesh_shape,
    worker_grid,
    transpose,
    algorithm,
    communication_workers,
    error,
):
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=mesh_shape,
        m_tiles=10,
        k_tiles_per_device=4,
        n_tiles_per_device=4,
        worker_grid=worker_grid,
        transpose=transpose,
    )
    with pytest.raises(ValueError, match=error):
        make_dedicated_operation(
            config,
            all_gather_algorithm=algorithm,
            communication_worker_count=communication_workers,
        )
