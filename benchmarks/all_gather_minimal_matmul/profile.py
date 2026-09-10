# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile Python dispatch overhead; device timings come from the benchmark."""

import argparse
import cProfile
import os
import pstats
from pathlib import Path

import ttnn

from benchmarks.all_gather_minimal_matmul.__main__ import (
    create_workloads,
    open_participant_mesh,
)
from examples.all_gather_minimal_matmul import AllGatherMinimalMatmulConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("ttlang", "ttmetal"), required=True
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.runs <= 0:
        parser.error("--runs must be positive")
    with open_participant_mesh() as (
        mesh,
        mesh_shape,
        cluster_axis,
        _discovered,
        _fabric_config,
    ):
        config = AllGatherMinimalMatmulConfig(
            mesh_shape=mesh_shape,
            m_tiles=2,
            k_tiles_per_device=1,
            n_tiles_per_device=4,
        )
        workloads, validate = create_workloads(
            mesh, config, "bf16", cluster_axis, arguments.implementation, 0
        )
        run, gathered, cleanup = workloads[arguments.implementation]
        for _iteration in range(3):
            output = run()
            ttnn.synchronize_device(mesh)
            validate(arguments.implementation, output, gathered)
            cleanup(output)
        profiler = cProfile.Profile()
        profiler.enable()
        for _iteration in range(arguments.runs):
            output = run()
            cleanup(output)
        ttnn.synchronize_device(mesh)
        profiler.disable()
        profiler.dump_stats(str(arguments.output))
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(40)
        if os.getenv("TT_METAL_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(mesh)


if __name__ == "__main__":
    main()
