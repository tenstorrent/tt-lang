# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device time of the TT-Lang ring all-gather and TT-Metal all-gather.

Both gather the same K-sharded BF16 tensor across every discovered device,
ordered as a ring. Device time is TT-Metal's first-kernel-start to
last-kernel-end interval of the slowest device, which is when the collective
completes. Run with `TT_METAL_DEVICE_PROFILER=1`,
`TT_METAL_PROFILER_MID_RUN_DUMP=1`, and `TT_METAL_PROFILER_DIR` set.
"""

import argparse
import os
import statistics
from pathlib import Path

import torch
import ttnn

from benchmarks.common import write_csv
from benchmarks.device_timing import latest_kernel_duration, read_device_profile
from examples.multidevice_ring_all_gather import (
    TILE_SIZE,
    expected_destination,
    make_ring_all_gather_operation,
    open_ring_mesh,
)
from ttlang_test_utils import to_dram

FABRIC_CONFIGS = {
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "1d-ring": ttnn.FabricConfig.FABRIC_1D_RING,
}

CSV_FIELDS = (
    "implementation",
    "fabric_config",
    "device_count",
    "m_tiles",
    "k_shard_tiles",
    "lanes",
    "chunk_shape",
    "warmup",
    "runs",
    "samples_us",
    "median_us",
    "received_gb_per_s_per_device",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("ttlang", "ttmetal"), required=True
    )
    parser.add_argument("--m-tiles", type=int, default=128)
    parser.add_argument("--k-shard-tiles", type=int, default=48)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--chunk-shape", type=int, nargs=2, default=(8, 12))
    parser.add_argument("--fabric-config", choices=tuple(FABRIC_CONFIGS), default="2d")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--csv", type=Path, default=Path("/tmp/all_gather.csv"))
    return parser.parse_args()


def make_ttlang_workload(mesh_device, full, arguments, device_count):
    m = arguments.m_tiles * TILE_SIZE
    k_shard = arguments.k_shard_tiles * TILE_SIZE
    source = to_dram(
        full, mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)
    )
    destination = to_dram(
        torch.zeros(device_count * m, (device_count - 1) * k_shard, dtype=full.dtype),
        mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    operation = make_ring_all_gather_operation(
        device_count,
        m_tiles=arguments.m_tiles,
        k_shard_tiles=arguments.k_shard_tiles,
        lanes=arguments.lanes,
        chunk_shape=tuple(arguments.chunk_shape),
    )

    def run():
        operation(source, destination)

    def correct():
        result = ttnn.to_torch(
            destination, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )
        return all(
            torch.equal(
                result[device * m : (device + 1) * m],
                expected_destination(full, device, device_count, k_shard),
            )
            for device in range(device_count)
        )

    return run, correct


def make_ttmetal_workload(mesh_device, full, arguments, device_count):
    m = arguments.m_tiles * TILE_SIZE
    source = to_dram(
        full, mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)
    )
    destination = to_dram(
        torch.zeros_like(full),
        mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run():
        ttnn.all_gather(source, dim=1, cluster_axis=0, output_tensor=destination)

    def correct():
        result = ttnn.to_torch(
            destination, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )
        return all(
            torch.equal(result[device * m : (device + 1) * m], full)
            for device in range(device_count)
        )

    return run, correct


def main():
    arguments = parse_args()
    if (
        os.environ.get("TT_METAL_DEVICE_PROFILER") != "1"
        or os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") != "1"
        or not os.environ.get("TT_METAL_PROFILER_DIR")
    ):
        raise SystemExit(
            "set TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1, "
            "and TT_METAL_PROFILER_DIR"
        )
    with open_ring_mesh(FABRIC_CONFIGS[arguments.fabric_config]) as mesh_device:
        device_count = mesh_device.get_num_devices()
        full = torch.randn(
            arguments.m_tiles * TILE_SIZE,
            device_count * arguments.k_shard_tiles * TILE_SIZE,
            dtype=torch.bfloat16,
        )
        make_workload = (
            make_ttlang_workload
            if arguments.implementation == "ttlang"
            else make_ttmetal_workload
        )
        run, correct = make_workload(mesh_device, full, arguments, device_count)
        for _ in range(arguments.warmup):
            run()
            ttnn.synchronize_device(mesh_device)
        if not correct():
            raise SystemExit("gathered output does not match the input shards")
        samples = []
        for _ in range(arguments.runs):
            run()
            ttnn.synchronize_device(mesh_device)
            duration = latest_kernel_duration(
                read_device_profile(mesh_device), mesh_device.get_device_ids()
            )
            samples.append(duration["us"])
        if not correct():
            raise SystemExit("gathered output does not match the input shards")
    received_bytes = (
        (device_count - 1) * (full.numel() // device_count) * full.element_size()
    )
    median_us = statistics.median(samples)
    row = {
        "implementation": arguments.implementation,
        "fabric_config": arguments.fabric_config,
        "device_count": device_count,
        "m_tiles": arguments.m_tiles,
        "k_shard_tiles": arguments.k_shard_tiles,
        "lanes": arguments.lanes,
        "chunk_shape": "x".join(map(str, arguments.chunk_shape)),
        "warmup": arguments.warmup,
        "runs": arguments.runs,
        "samples_us": ";".join(f"{sample:.2f}" for sample in samples),
        "median_us": f"{median_us:.2f}",
        "received_gb_per_s_per_device": f"{received_bytes / median_us / 1e3:.1f}",
    }
    write_csv(arguments.csv, CSV_FIELDS, row)
    print(row)


if __name__ == "__main__":
    main()
