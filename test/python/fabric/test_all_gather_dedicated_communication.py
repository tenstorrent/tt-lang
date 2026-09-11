# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dedicated-worker activation exchange with cached and streamed matmul."""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from benchmarks.all_gather_minimal_matmul.__main__ import (
    create_workloads,
    open_participant_mesh,
)
from examples.all_gather_minimal_matmul import AllGatherMinimalMatmulConfig

pytestmark = pytest.mark.multi_device


@pytest.fixture(
    params=[(2, 2), (4, 2), (4, 4)],
    ids=[
        "two-devices-two-workers",
        "four-devices-two-workers",
        "four-devices-four-workers",
    ],
)
def participants(request):
    device_count, communication_workers = request.param
    with open_participant_mesh((device_count, 1), "2d", "strict", 8192) as opened:
        yield opened[0], (device_count, 1), communication_workers


@pytest.mark.parametrize("worker_grid", [(5, 4), (12, 10)])
@pytest.mark.parametrize(
    "dtype,dest_fp32",
    [("bf16", False), ("bf16", True), ("fp32", True)],
    ids=["bf16-dst16", "bf16-dst32", "fp32"],
)
@pytest.mark.parametrize("reuse_activation", [False, True], ids=["stream", "cache"])
def test_dedicated_communication(
    participants, worker_grid, dtype, dest_fp32, reuse_activation
):
    mesh, mesh_shape, communication_workers = participants
    config = AllGatherMinimalMatmulConfig(
        mesh_shape=mesh_shape,
        m_tiles=2 * (worker_grid[0] + 1),
        k_tiles_per_device=4,
        n_tiles_per_device=2 * worker_grid[1],
        m_block_tiles=2,
        k_block_tiles=2,
        n_block_tiles=1,
        worker_grid=worker_grid,
        transpose=True,
        reuse_activation=reuse_activation,
    )
    workloads, validate = create_workloads(
        mesh,
        config,
        dtype,
        0,
        "ttlang",
        19,
        math_fidelity="HiFi4" if dtype == "fp32" else "HiFi2",
        fp32_dest_acc=dest_fp32,
        activation_all_gather="ring",
        dedicated_communication_workers=communication_workers,
        gather_output=worker_grid == (5, 4),
    )
    workload = workloads["ttlang"]
    for invocation in range(2):
        output = workload.run()
        ttnn.synchronize_device(mesh)
        validate("ttlang", output, workload.gathered)
        workload.cleanup(output)


def test_full_grid_local_distribution_workers():
    mesh_shape = (4, 1)
    worker_grid = (12, 10)
    with open_participant_mesh(mesh_shape, "2d", "strict", 8192) as opened:
        mesh = opened[0]
        config = AllGatherMinimalMatmulConfig(
            mesh_shape=mesh_shape,
            m_tiles=2 * worker_grid[0],
            k_tiles_per_device=4,
            n_tiles_per_device=2 * worker_grid[1],
            m_block_tiles=2,
            k_block_tiles=2,
            n_block_tiles=1,
            worker_grid=worker_grid,
            transpose=True,
            reuse_activation=False,
        )
        workloads, validate = create_workloads(
            mesh,
            config,
            "bf16",
            0,
            "ttlang",
            19,
            math_fidelity="HiFi2",
            fp32_dest_acc=True,
            activation_all_gather="ring",
            dedicated_communication_workers=10,
            gather_output=False,
        )
        workload = workloads["ttlang"]
        for _invocation in range(2):
            output = workload.run()
            ttnn.synchronize_device(mesh)
            validate("ttlang", output, workload.gathered)
            workload.cleanup(output)
