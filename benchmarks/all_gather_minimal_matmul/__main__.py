# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare TT-Lang and TT-Metal all-gather matmul on a local fabric pair."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import ttnn

from benchmarks.device_timing import latest_kernel_duration, read_device_profile
from benchmarks.provenance import collect_provenance
from examples.all_gather_minimal_matmul import (
    AllGatherMinimalMatmulConfig,
    make_all_gather_minimal_matmul_operation,
)
from ttlang_test_utils import get_fabric_mesh_shape, to_dram
from utils.correctness import assert_allclose, assert_pcc

REFERENCE_REVISION = "f69f924c6b4f38daa0a6f25716731f36c573dc0e"
REFERENCE_ROOT = f"https://github.com/tenstorrent/tt-metal/blob/{REFERENCE_REVISION}"
MATH_FIDELITIES = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("both", "ttlang", "ttmetal"), default="both"
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--math-fidelity", choices=("HiFi2", "HiFi4"))
    parser.add_argument(
        "--fp32-dest-acc", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--native-channel-buffers", type=positive_int, default=24)
    parser.add_argument("--device-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--m-tiles", type=positive_int, default=2)
    parser.add_argument("--k-tiles-per-device", type=positive_int, default=1)
    parser.add_argument("--n-tiles-per-device", type=positive_int, default=4)
    parser.add_argument("--m-block-tiles", type=positive_int, default=1)
    parser.add_argument(
        "--k-block-tiles",
        type=positive_int,
        default=1,
        help="compute and fabric K block in tiles",
    )
    parser.add_argument(
        "--reuse-activation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--n-block-tiles", type=positive_int, default=1)
    parser.add_argument("--worker-grid", type=positive_int, nargs=2, metavar=("X", "Y"))
    parser.add_argument(
        "--transpose", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json", type=Path, default=Path("/tmp/all_gather_minimal_matmul_perf.json")
    )
    arguments = parser.parse_args()
    if arguments.math_fidelity is None:
        arguments.math_fidelity = "HiFi2" if arguments.dtype == "bf16" else "HiFi4"
    return arguments


def native_subblock(config):
    return tuple(
        2 if extent % 2 == 0 else 1
        for extent in (config.m_block_tiles, config.n_block_tiles)
    )


@contextmanager
def open_participant_mesh():
    parent_mesh = None
    participant_mesh = None
    try:
        discovered_shape = get_fabric_mesh_shape(
            fabric_config=ttnn.FabricConfig.FABRIC_1D
        )
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
        )
        participant_axis = next(
            (axis for axis, extent in enumerate(discovered_shape) if extent >= 2), None
        )
        if participant_axis is None:
            raise RuntimeError("benchmark requires a connected two-device fabric")
        participant_shape = tuple(
            2 if axis == participant_axis else 1
            for axis in range(len(discovered_shape))
        )
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(discovered_shape))
        participant_mesh = (
            parent_mesh
            if participant_shape == tuple(discovered_shape)
            else parent_mesh.create_submesh(ttnn.MeshShape(participant_shape))
        )
        yield participant_mesh, participant_shape, participant_axis, discovered_shape
    finally:
        if participant_mesh is not None and participant_mesh is not parent_mesh:
            ttnn.close_mesh_device(participant_mesh)
        if parent_mesh is not None:
            ttnn.close_mesh_device(parent_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def create_workloads(
    mesh,
    config,
    dtype,
    cluster_axis,
    implementation,
    seed,
    *,
    math_fidelity="HiFi2",
    fp32_dest_acc=True,
    native_channel_buffers=24,
):
    torch.manual_seed(seed)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    m_elements = config.m_tiles * 32
    k_elements = config.device_count * config.k_tiles_per_device * 32
    n_elements = config.device_count * config.n_tiles_per_device * 32
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype) / k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    expected = activation.float() @ weight.float() + bias.float()
    shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=1)
    replicate_mapper = ttnn.ReplicateTensorToMesh(mesh)
    activation_device = to_dram(activation, mesh, mesh_mapper=shard_mapper)
    weight_device = to_dram(weight, mesh, mesh_mapper=shard_mapper)
    bias_device = to_dram(bias, mesh, mesh_mapper=shard_mapper)

    workloads = {}
    for name in (
        ("ttlang", "ttmetal") if implementation == "both" else (implementation,)
    ):
        gathered = None
        if name == "ttlang":
            output = to_dram(
                torch.zeros((m_elements, n_elements), dtype=torch_dtype),
                mesh,
                mesh_mapper=shard_mapper,
            )
            operation = make_all_gather_minimal_matmul_operation(
                config, math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc
            )

            def run_ttlang(operation=operation, output=output):
                operation(activation_device, weight_device, bias_device, output)
                return output

            workloads[name] = (run_ttlang, gathered, lambda result: None)
        else:
            gathered = to_dram(
                torch.zeros_like(activation), mesh, mesh_mapper=replicate_mapper
            )
            full_grid = mesh.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1),
                    )
                }
            )
            semaphores = [
                ttnn.create_global_semaphore(mesh, cores, 0) for _index in range(2)
            ]
            native_config = ttnn.MinimalMatmulConfig(
                M_block_size=config.m_block_tiles,
                K_block_size=config.compute_k_tiles,
                N_block_size=config.n_block_tiles,
                subblock_h=native_subblock(config)[0],
                subblock_w=native_subblock(config)[1],
                compute_with_storage_grid_size=ttnn.CoreCoord(*config.grid),
            )
            compute_config = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=MATH_FIDELITIES[math_fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc,
                packer_l1_acc=True,
            )

            def run_ttmetal(gathered=gathered, semaphores=semaphores):
                result = ttnn.experimental.all_gather_minimal_matmul_async(
                    activation_device,
                    weight_device,
                    bias_tensor=bias_device,
                    config=native_config,
                    compute_kernel_config=compute_config,
                    persistent_output_buffer=gathered,
                    multi_device_global_semaphore=semaphores,
                    topology=ttnn.Topology.Linear,
                    cluster_axis=cluster_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    force_transpose=config.transpose,
                    num_links=1,
                    num_workers_per_link=config.m_workers,
                    num_buffers_per_channel=native_channel_buffers,
                )
                if len(result) != 1:
                    raise RuntimeError(f"expected one native output, got {len(result)}")
                return result[0]

            workloads[name] = (run_ttmetal, gathered, ttnn.deallocate)

    def validate(name, output, gathered):
        output_torch = ttnn.to_torch(
            output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)
        ).float()
        if name == "ttmetal":
            gathered_torch = ttnn.to_torch(
                gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float()
            expected_gather = activation.float().repeat(config.device_count, 1)
            # Native compute reads its local shard directly; only remote shards
            # are written into the persistent gather scratch buffer.
            local_k = config.k_tiles_per_device * 32
            for device_index in range(config.device_count):
                expected_gather[
                    device_index * m_elements : (device_index + 1) * m_elements,
                    device_index * local_k : (device_index + 1) * local_k,
                ] = 0
            assert_allclose(gathered_torch, expected_gather, rtol=0, atol=0)
        assert_pcc(expected, output_torch, threshold=0.999 if dtype == "fp32" else 0.99)
        # The FPU truncates FP32 sources to TF32 even with FP32 destinations.
        tolerance = 0.005 if dtype == "fp32" else 0.05
        assert_allclose(
            output_torch,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
        absolute_error = (output_torch - expected).abs()
        return {
            "max_abs_error": absolute_error.max().item(),
            "mean_abs_error": absolute_error.mean().item(),
            "rtol": tolerance,
            "atol": tolerance,
            "pcc_threshold": 0.999 if dtype == "fp32" else 0.99,
        }

    return workloads, validate


def benchmark(workloads, validate, mesh, arguments):
    correctness = {name: [] for name in workloads}
    for name, (run, gathered, cleanup) in workloads.items():
        print(f"Compile and validate {name}", flush=True)
        for _iteration in range(arguments.warmup):
            output = run()
            ttnn.synchronize_device(mesh)
            correctness[name].append(validate(name, output, gathered))
            cleanup(output)

    samples = {name: [] for name in workloads}
    previous_run_ids = {}
    for sample_index in range(arguments.samples):
        for name, (run, gathered, cleanup) in workloads.items():
            output = run()
            ttnn.synchronize_device(mesh)
            # Flush before validation so the newest program is the operation,
            # not a conversion or setup program used by correctness checking.
            profiler_log = read_device_profile(mesh)
            duration = latest_kernel_duration(
                profiler_log,
                mesh.get_device_ids(),
                aggregation=arguments.device_aggregation,
            )
            for device_id, device in duration["per_device"].items():
                run_id = device["run_host_id"]
                if run_id <= previous_run_ids.get(device_id, -1):
                    raise ValueError(f"stale profiler data for device {device_id}")
                previous_run_ids[device_id] = run_id
            correctness[name].append(validate(name, output, gathered))
            cleanup(output)
            samples[name].append(duration)
            print(
                f"{name} sample {sample_index + 1}: {duration['cycles']} cycles, "
                f"{duration['us']:.3f} us device kernel",
                flush=True,
            )
    measurements = {
        name: {
            "samples": values,
            "median_cycles": statistics.median(value["cycles"] for value in values),
            "median_us": statistics.median(value["us"] for value in values),
            "min_us": min(value["us"] for value in values),
            "max_us": max(value["us"] for value in values),
        }
        for name, values in samples.items()
    }
    measurements["correctness"] = correctness
    return measurements


def run_isolated_variants(arguments):
    reports = {}
    variants = (
        ("ttlang", "ttmetal")
        if arguments.implementation == "both"
        else (arguments.implementation,)
    )
    for name in variants:
        # Profiler configuration is read at import time. Separate processes also
        # keep native and generated program logs independent without deleting logs.
        directory = Path(
            tempfile.mkdtemp(
                prefix=f"{arguments.json.stem}.{name}.",
                dir=arguments.json.resolve().parent,
            )
        )
        output_file = directory / "result.json"
        environment = {
            **os.environ,
            "TT_METAL_DEVICE_PROFILER": "1",
            "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
            "TT_METAL_PROFILER_DIR": str(directory),
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.all_gather_minimal_matmul",
                *sys.argv[1:],
                "--worker",
                "--implementation",
                name,
                "--json",
                str(output_file),
            ],
            env=environment,
            check=True,
            timeout=180,
        )
        reports[name] = json.loads(output_file.read_text())
    combined = {
        "timing": f"ttmetal_device_kernel_duration_{arguments.device_aggregation}_across_participating_devices",
        "variants": reports,
    }
    if len(reports) == 2:
        combined["ttlang_over_ttmetal"] = (
            reports["ttlang"]["measurements"]["ttlang"]["median_us"]
            / reports["ttmetal"]["measurements"]["ttmetal"]["median_us"]
        )
        print(f"TT-Lang/native device ratio: {combined['ttlang_over_ttmetal']:.3f}")
    arguments.json.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"Results: {arguments.json}", flush=True)


def main():
    arguments = parse_args()
    for variable in (
        "TTLANG_COMPILE_ONLY",
        "TTLANG_AUTO_PROFILE",
        "TTLANG_PERF_DUMP",
        "TTLANG_SIGNPOST_PROFILE",
        "TT_METAL_PROFILER_ACCUMULATE",
    ):
        if os.getenv(variable, "0") not in ("", "0"):
            raise ValueError(f"unset {variable} before timing")
    if not arguments.worker:
        run_isolated_variants(arguments)
        return
    if arguments.implementation == "both":
        raise ValueError("profiler workers must run exactly one implementation")
    if os.getenv("TT_METAL_DEVICE_PROFILER") != "1":
        raise ValueError("profiler worker requires TT_METAL_DEVICE_PROFILER=1")
    from tracy import device_post_proc_config, process_device_log
    from benchmarks.provenance import file_sha256

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timing": f"ttmetal_device_kernel_duration_{arguments.device_aggregation}_across_participating_devices",
        "profiler_directory": os.environ["TT_METAL_PROFILER_DIR"],
        "profiler_source_sha256": {
            module.__file__: file_sha256(module.__file__)
            for module in (device_post_proc_config, process_device_log)
        },
        "provenance": collect_provenance(
            [
                __file__,
                Path(__file__).resolve().parents[1] / "device_timing.py",
                Path(__file__).resolve().parents[2]
                / "examples/all_gather_minimal_matmul/operation.py",
            ]
        ),
        "arguments": {**vars(arguments), "json": str(arguments.json)},
        "methodology_reference": {
            "sweep": f"{REFERENCE_ROOT}/models/tt_dit/utils/sweep_mm_block_sizes.py",
            "test": f"{REFERENCE_ROOT}/models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py",
        },
    }
    with open_participant_mesh() as (mesh, mesh_shape, cluster_axis, discovered_shape):
        config = AllGatherMinimalMatmulConfig(
            mesh_shape=mesh_shape,
            m_tiles=arguments.m_tiles,
            k_tiles_per_device=arguments.k_tiles_per_device,
            n_tiles_per_device=arguments.n_tiles_per_device,
            m_block_tiles=arguments.m_block_tiles,
            k_block_tiles=arguments.k_block_tiles,
            reuse_activation=(
                arguments.reuse_activation and arguments.implementation == "ttlang"
            ),
            n_block_tiles=arguments.n_block_tiles,
            worker_grid=arguments.worker_grid,
            transpose=arguments.transpose,
        )
        if not config.transpose and config.m_tiles > config.n_tiles_per_device:
            raise ValueError(
                "matched non-transposed scheduling requires M <= per-device N"
            )
        if arguments.implementation != "ttlang" and config.n_workers < 4:
            raise ValueError(
                "native non-transposed all-gather matmul requires at least four "
                "N workers: its interior receiver range is [1, grid.x - 3]"
            )
        report.update(
            config=asdict(config),
            discovered_mesh=list(discovered_shape),
            cluster_axis=cluster_axis,
            device_ids=list(mesh.get_device_ids()),
            arch=str(mesh.arch()),
            compute_grid=config.grid,
            math_fidelity=arguments.math_fidelity,
            fabric_config="FABRIC_1D",
            fp32_dest_acc_en=arguments.fp32_dest_acc,
            native_channel_buffers=arguments.native_channel_buffers,
            device_aggregation=arguments.device_aggregation,
            layout="TILE",
            memory="DRAM",
            per_device_matmul={
                "m": config.m_tiles * 32,
                "gathered_k": config.k_tiles_per_device * config.device_count * 32,
                "n": config.n_tiles_per_device * 32,
                "local_k": config.k_tiles_per_device * 32,
            },
            native_config={
                "links": 1,
                "workers_per_link": config.m_workers,
                "compute_k_tiles": config.compute_k_tiles,
                "subblock": list(native_subblock(config)),
                "channel_buffers": arguments.native_channel_buffers,
                "packer_l1_acc": True,
            },
            execution="ordinary_launch",
            fabric_reliability="RELAXED_INIT",
            fabric_router_payload="installed_runtime_default",
        )
        workloads, validate = create_workloads(
            mesh,
            config,
            arguments.dtype,
            cluster_axis,
            arguments.implementation,
            arguments.seed,
            math_fidelity=arguments.math_fidelity,
            fp32_dest_acc=arguments.fp32_dest_acc,
            native_channel_buffers=arguments.native_channel_buffers,
        )
        report["measurements"] = benchmark(workloads, validate, mesh, arguments)
    arguments.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["measurements"], indent=2), flush=True)
    print(f"Results: {arguments.json}", flush=True)


if __name__ == "__main__":
    main()
