# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare TT-Lang and TT-Metal column-parallel all-gather matmul."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import prod
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

REFERENCE_REVISION = "ea042c4ad6237678103cd7cbceb346e060f0f9a3"
REFERENCE_ROOT = f"https://github.com/tenstorrent/tt-metal/blob/{REFERENCE_REVISION}"
MATH_FIDELITIES = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


@dataclass(frozen=True)
class CommonConfig:
    mesh_shape: tuple[int, int]
    m_tiles: int
    k_tiles_per_device: int
    n_tiles: int
    dtype: str
    math_fidelity: str
    fp32_dest_acc: bool
    seed: int

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    @property
    def n_tiles_per_device(self) -> int:
        if self.n_tiles % self.device_count:
            raise ValueError("complete N tile count must be divisible by device count")
        return self.n_tiles // self.device_count


@dataclass(frozen=True)
class TTLangConfig:
    compute_grid: tuple[int, int]
    communication_workers: int
    m_block_tiles: int
    k_block_tiles: int
    n_block_tiles: int
    reuse_activation: bool


@dataclass(frozen=True)
class NativeConfig:
    compute_grid: tuple[int, int]
    m_block_tiles: int
    k_block_tiles: int
    n_block_tiles: int
    subblock: tuple[int, int]
    num_links: int
    workers_per_link: int
    channel_buffers: int
    chunks: int
    math_approx_mode: bool


@dataclass
class Workload:
    run: Callable
    validate: Callable
    cleanup: Callable


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def mesh_shape(value: str) -> tuple[int, int]:
    try:
        result = tuple(int(extent) for extent in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("mesh shape must be ROWSxCOLS") from error
    if len(result) != 2 or min(result) != 1 or max(result) < 2:
        raise argparse.ArgumentTypeError(
            "mesh shape must be a device line, such as 4x1"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("both", "ttlang", "ttmetal"), default="both"
    )
    parser.add_argument("--mesh-shape", type=mesh_shape, default=(4, 1))
    parser.add_argument("--m-tiles", type=positive_int, default=296)
    parser.add_argument("--k-tiles-per-device", type=positive_int, default=40)
    parser.add_argument("--n-tiles", type=positive_int, default=480)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--math-fidelity", choices=("HiFi2", "HiFi4"), default="HiFi2")
    parser.add_argument(
        "--fp32-dest-acc", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--ttlang-compute-grid", type=positive_int, nargs=2, default=(12, 10)
    )
    parser.add_argument("--ttlang-communication-workers", type=positive_int, default=4)
    parser.add_argument("--ttlang-m-block-tiles", type=positive_int, default=5)
    parser.add_argument("--ttlang-k-block-tiles", type=positive_int, default=10)
    parser.add_argument("--ttlang-n-block-tiles", type=positive_int, default=12)
    parser.add_argument(
        "--ttlang-reuse-activation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--native-compute-grid", type=positive_int, nargs=2, default=(12, 9)
    )
    parser.add_argument("--native-m-block-tiles", type=positive_int, default=7)
    parser.add_argument("--native-k-block-tiles", type=positive_int, default=5)
    parser.add_argument("--native-n-block-tiles", type=positive_int, default=16)
    parser.add_argument("--native-subblock", type=positive_int, nargs=2, default=(1, 2))
    parser.add_argument("--native-num-links", type=positive_int, default=2)
    parser.add_argument("--native-workers-per-link", type=positive_int, default=6)
    parser.add_argument("--native-channel-buffers", type=positive_int, default=24)
    parser.add_argument("--native-chunks", type=positive_int, default=3)
    parser.add_argument(
        "--native-math-approx-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--fabric-router-payload", type=positive_int, default=8192)
    parser.add_argument("--device-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-timeout", type=positive_int, default=900)
    parser.add_argument(
        "--json", type=Path, default=Path("/tmp/all-gather-minimal-matmul.json")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def make_configs(arguments):
    common = CommonConfig(
        mesh_shape=arguments.mesh_shape,
        m_tiles=arguments.m_tiles,
        k_tiles_per_device=arguments.k_tiles_per_device,
        n_tiles=arguments.n_tiles,
        dtype=arguments.dtype,
        math_fidelity=arguments.math_fidelity,
        fp32_dest_acc=arguments.fp32_dest_acc,
        seed=arguments.seed,
    )
    ttlang = TTLangConfig(
        compute_grid=tuple(arguments.ttlang_compute_grid),
        communication_workers=arguments.ttlang_communication_workers,
        m_block_tiles=arguments.ttlang_m_block_tiles,
        k_block_tiles=arguments.ttlang_k_block_tiles,
        n_block_tiles=arguments.ttlang_n_block_tiles,
        reuse_activation=arguments.ttlang_reuse_activation,
    )
    native = NativeConfig(
        compute_grid=tuple(arguments.native_compute_grid),
        m_block_tiles=arguments.native_m_block_tiles,
        k_block_tiles=arguments.native_k_block_tiles,
        n_block_tiles=arguments.native_n_block_tiles,
        subblock=tuple(arguments.native_subblock),
        num_links=arguments.native_num_links,
        workers_per_link=arguments.native_workers_per_link,
        channel_buffers=arguments.native_channel_buffers,
        chunks=arguments.native_chunks,
        math_approx_mode=arguments.native_math_approx_mode,
    )
    if common.dtype == "fp32" and not common.fp32_dest_acc:
        raise ValueError("FP32 inputs require FP32 destination accumulation")
    return common, ttlang, native


def fabric_router_config(max_payload_size: int):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@contextmanager
def open_participant_mesh(common, implementation, router_payload):
    fabric_config = (
        ttnn.FabricConfig.FABRIC_2D
        if implementation == "ttlang"
        else ttnn.FabricConfig.FABRIC_1D_RING
    )
    reliability = ttnn.FabricReliabilityMode.STRICT_INIT
    router_config = fabric_router_config(router_payload)
    discovered_shape = get_fabric_mesh_shape(
        fabric_config=fabric_config,
        reliability_mode=reliability,
        router_config=router_config,
    )
    ttnn.set_fabric_config(
        fabric_config, reliability_mode=reliability, router_config=router_config
    )
    parent_mesh = None
    participant_mesh = None
    try:
        if common.device_count > prod(discovered_shape):
            raise ValueError(
                f"requested mesh {common.mesh_shape} exceeds discovered mesh "
                f"{discovered_shape}"
            )
        parent_shape = tuple(discovered_shape)
        if any(
            requested > available
            for requested, available in zip(
                common.mesh_shape, discovered_shape, strict=True
            )
        ):
            parent_shape = (
                (prod(discovered_shape), 1)
                if common.mesh_shape[0] > 1
                else (1, prod(discovered_shape))
            )
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(discovered_shape))
        if parent_shape != tuple(discovered_shape):
            parent_mesh.reshape(ttnn.MeshShape(parent_shape))
        participant_mesh = (
            parent_mesh
            if common.mesh_shape == parent_shape
            else parent_mesh.create_submesh(ttnn.MeshShape(common.mesh_shape))
        )
        cluster_axis = 0 if common.mesh_shape[0] > 1 else 1
        yield participant_mesh, cluster_axis, discovered_shape, fabric_config
    finally:
        if participant_mesh is not None and participant_mesh is not parent_mesh:
            ttnn.close_mesh_device(participant_mesh)
        if parent_mesh is not None:
            ttnn.close_mesh_device(parent_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def make_inputs(mesh, common, padded_m_tiles=None):
    torch.manual_seed(common.seed)
    torch_dtype = torch.bfloat16 if common.dtype == "bf16" else torch.float32
    m_elements = common.m_tiles * 32
    k_elements = common.device_count * common.k_tiles_per_device * 32
    n_elements = common.n_tiles * 32
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype) / k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    expected = activation.float() @ weight.float() + bias.float()
    activation_storage = activation
    if padded_m_tiles is not None and padded_m_tiles != common.m_tiles:
        activation_storage = torch.nn.functional.pad(
            activation, (0, 0, 0, padded_m_tiles * 32 - m_elements)
        )
    shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=1)
    return (
        activation,
        expected,
        to_dram(activation_storage, mesh, mesh_mapper=shard_mapper),
        to_dram(weight, mesh, mesh_mapper=shard_mapper),
        to_dram(bias, mesh, mesh_mapper=shard_mapper),
        shard_mapper,
    )


def validate_output(actual, expected, dtype):
    assert_pcc(expected, actual, threshold=0.99 if dtype == "bf16" else 0.999)
    tolerance = 0.05 if dtype == "bf16" else 0.005
    assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    absolute_error = (actual - expected).abs()
    return {
        "max_abs_error": absolute_error.max().item(),
        "mean_abs_error": absolute_error.mean().item(),
        "rtol": tolerance,
        "atol": tolerance,
    }


def create_ttlang_workload(mesh, common, ttlang):
    operation_config = AllGatherMinimalMatmulConfig(
        mesh_shape=common.mesh_shape,
        m_tiles=common.m_tiles,
        k_tiles_per_device=common.k_tiles_per_device,
        n_tiles_per_device=common.n_tiles_per_device,
        compute_grid=ttlang.compute_grid,
        m_block_tiles=ttlang.m_block_tiles,
        k_block_tiles=ttlang.k_block_tiles,
        n_block_tiles=ttlang.n_block_tiles,
        reuse_activation=ttlang.reuse_activation,
    )
    _, expected, activation, weight, bias, shard_mapper = make_inputs(
        mesh, common, operation_config.padded_m_tiles
    )
    torch_dtype = torch.bfloat16 if common.dtype == "bf16" else torch.float32
    output = to_dram(
        torch.zeros((common.m_tiles * 32, common.n_tiles * 32), dtype=torch_dtype),
        mesh,
        mesh_mapper=shard_mapper,
    )
    operation = make_all_gather_minimal_matmul_operation(
        operation_config,
        math_fidelity=common.math_fidelity,
        fp32_dest_acc_en=common.fp32_dest_acc,
        communication_worker_count=ttlang.communication_workers,
    )

    def run():
        operation(activation, weight, bias, output)
        return output

    def validate(result):
        actual = ttnn.to_torch(
            result, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)
        ).float()
        return validate_output(actual, expected, common.dtype)

    return Workload(run, validate, lambda _result: None), operation_config


def create_native_workload(mesh, cluster_axis, common, native):
    operation_config = {
        "mesh_shape": common.mesh_shape,
        "m_tiles": common.m_tiles,
        "k_tiles_per_device": common.k_tiles_per_device,
        "n_tiles_per_device": common.n_tiles_per_device,
        **asdict(native),
    }
    activation_host, expected, activation, weight, bias, _ = make_inputs(mesh, common)
    gathered = to_dram(
        torch.zeros_like(activation_host),
        mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    full_grid = mesh.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1),
            )
        }
    )
    semaphores = [
        ttnn.create_global_semaphore(mesh, all_cores, 0) for _index in range(2)
    ]
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=native.m_block_tiles,
        K_block_size=native.k_block_tiles,
        N_block_size=native.n_block_tiles,
        subblock_h=native.subblock[0],
        subblock_w=native.subblock[1],
        compute_with_storage_grid_size=ttnn.CoreCoord(*native.compute_grid),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=MATH_FIDELITIES[common.math_fidelity],
        math_approx_mode=native.math_approx_mode,
        fp32_dest_acc_en=common.fp32_dest_acc,
        packer_l1_acc=True,
    )

    def run():
        outputs = ttnn.experimental.all_gather_minimal_matmul_async(
            activation,
            weight,
            bias_tensor=bias,
            config=matmul_config,
            compute_kernel_config=compute_config,
            persistent_output_buffer=gathered,
            multi_device_global_semaphore=semaphores,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            force_transpose=True,
            num_links=native.num_links,
            num_workers_per_link=native.workers_per_link,
            num_buffers_per_channel=native.channel_buffers,
            chunks=native.chunks,
        )
        if len(outputs) != native.chunks:
            raise RuntimeError(
                f"expected {native.chunks} outputs, received {len(outputs)}"
            )
        return outputs

    def validate(outputs):
        chunk_tensors = [
            ttnn.to_torch(
                output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float()
            for output in outputs
        ]
        m_elements = common.m_tiles * 32
        device_outputs = [
            torch.cat(
                [
                    chunk[device_index * m_elements : (device_index + 1) * m_elements]
                    for chunk in chunk_tensors
                ],
                dim=1,
            )
            for device_index in range(common.device_count)
        ]
        return validate_output(torch.cat(device_outputs, dim=1), expected, common.dtype)

    def cleanup(outputs):
        for output in outputs:
            ttnn.deallocate(output)

    return Workload(run, validate, cleanup), operation_config


def measure(workload, mesh, arguments):
    correctness = []
    for _iteration in range(arguments.warmup):
        output = workload.run()
        ttnn.synchronize_device(mesh)
        correctness.append(workload.validate(output))
        workload.cleanup(output)

    samples = []
    previous_run_ids = {}
    for sample_index in range(arguments.samples):
        output = workload.run()
        ttnn.synchronize_device(mesh)
        profiler_log = read_device_profile(mesh)
        duration = latest_kernel_duration(
            profiler_log,
            mesh.get_device_ids(),
            aggregation=arguments.device_aggregation,
        )
        correctness.append(workload.validate(output))
        workload.cleanup(output)
        for device_id, device in duration["per_device"].items():
            run_id = device["run_host_id"]
            if run_id <= previous_run_ids.get(device_id, -1):
                raise ValueError(f"stale profiler data for device {device_id}")
            previous_run_ids[device_id] = run_id
        samples.append(duration)
        print(
            f"sample {sample_index + 1}: {duration['us']:.3f} us device interval",
            flush=True,
        )
    return {
        "samples": samples,
        "median_us": statistics.median(sample["us"] for sample in samples),
        "min_us": min(sample["us"] for sample in samples),
        "max_us": max(sample["us"] for sample in samples),
        "correctness": correctness,
    }


def run_worker(arguments):
    if arguments.implementation == "both":
        raise ValueError("a profiler worker runs exactly one implementation")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        raise ValueError("profiler worker requires TT_METAL_DEVICE_PROFILER=1")
    common, ttlang, native = make_configs(arguments)
    with open_participant_mesh(
        common, arguments.implementation, arguments.fabric_router_payload
    ) as (mesh, cluster_axis, discovered_shape, fabric_config):
        if arguments.implementation == "ttlang":
            workload, operation_config = create_ttlang_workload(mesh, common, ttlang)
            implementation_config = asdict(ttlang)
        else:
            workload, operation_config = create_native_workload(
                mesh, cluster_axis, common, native
            )
            implementation_config = asdict(native)
        measurements = measure(workload, mesh, arguments)

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "implementation": arguments.implementation,
            "common_config": asdict(common),
            "implementation_config": implementation_config,
            "operation_config": (
                asdict(operation_config)
                if hasattr(operation_config, "__dataclass_fields__")
                else operation_config
            ),
            "device_ids": list(mesh.get_device_ids()),
            "discovered_mesh": list(discovered_shape),
            "arch": str(mesh.arch()),
            "fabric_config": str(fabric_config),
            "fabric_router_payload": arguments.fabric_router_payload,
            "device_aggregation": arguments.device_aggregation,
            "measurements": measurements,
            "provenance": collect_provenance(
                [
                    __file__,
                    Path(__file__).resolve().parents[1] / "device_timing.py",
                    Path(__file__).resolve().parents[2]
                    / "examples/all_gather_minimal_matmul/operation.py",
                ]
            ),
            "references": {
                "operation": f"{REFERENCE_ROOT}/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async",
                "sweep": f"{REFERENCE_ROOT}/models/tt_dit/utils/sweep_mm_block_sizes.py",
            },
        }
    arguments.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(measurements, indent=2), flush=True)


def run_isolated_workers(arguments):
    implementations = (
        ("ttlang", "ttmetal")
        if arguments.implementation == "both"
        else (arguments.implementation,)
    )
    reports = {}
    for implementation in implementations:
        directory = Path(
            tempfile.mkdtemp(
                prefix=f"{arguments.json.stem}.{implementation}.",
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
                implementation,
                "--json",
                str(output_file),
            ],
            env=environment,
            check=True,
            timeout=arguments.worker_timeout,
        )
        reports[implementation] = json.loads(output_file.read_text())

    combined = {
        "timing": (
            "ttmetal_device_kernel_duration_"
            f"{arguments.device_aggregation}_across_participating_devices"
        ),
        "variants": reports,
    }
    if len(reports) == 2:
        ratio = (
            reports["ttlang"]["measurements"]["median_us"]
            / reports["ttmetal"]["measurements"]["median_us"]
        )
        combined["ttlang_over_ttmetal"] = ratio
        print(f"TT-Lang/native device ratio: {ratio:.3f}", flush=True)
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
    if arguments.worker:
        run_worker(arguments)
    else:
        run_isolated_workers(arguments)


if __name__ == "__main__":
    main()
