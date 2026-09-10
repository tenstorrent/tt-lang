# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare TT-Lang and TT-Metal all-gather matmul on a fabric-connected device line."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
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
from examples.all_gather_minimal_matmul.collectives import make_column_all_gather
from examples.all_gather_minimal_matmul.replicated.operation import (
    make_replicated_all_gather_matmul_operation,
)
from ttlang_test_utils import get_fabric_mesh_shape, to_dram
from ttl.kernel_runner import prepare_device_invocation
from utils.correctness import assert_allclose, assert_pcc

REFERENCE_REVISION = "f69f924c6b4f38daa0a6f25716731f36c573dc0e"
REFERENCE_ROOT = f"https://github.com/tenstorrent/tt-metal/blob/{REFERENCE_REVISION}"
MATH_FIDELITIES = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
FABRIC_CONFIGS = {
    "1d": ttnn.FabricConfig.FABRIC_1D,
    "1d-ring": ttnn.FabricConfig.FABRIC_1D_RING,
    "2d": ttnn.FabricConfig.FABRIC_2D,
}
FABRIC_RELIABILITY_MODES = {
    "relaxed": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "strict": ttnn.FabricReliabilityMode.STRICT_INIT,
}


@dataclass(frozen=True)
class BenchmarkCase:
    """Tensor dimensions and implementation-independent compute parameters."""

    mesh_shape: tuple[int, int]
    m_tiles: int
    k_tiles_per_device: int
    n_tiles_per_device: int
    m_block_tiles: int
    k_block_tiles: int
    n_block_tiles: int
    worker_grid: tuple[int, int] | None
    transpose: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        if self.worker_grid is not None:
            object.__setattr__(self, "worker_grid", tuple(self.worker_grid))

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    @property
    def grid(self) -> tuple[int, int]:
        if self.worker_grid is not None:
            return self.worker_grid
        logical_grid = (
            (self.n_tiles_per_device + self.n_block_tiles - 1) // self.n_block_tiles,
            (self.m_tiles + self.m_block_tiles - 1) // self.m_block_tiles,
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

    def make_ttlang_config(
        self, reuse_activation: bool
    ) -> AllGatherMinimalMatmulConfig:
        return AllGatherMinimalMatmulConfig(
            **asdict(self), reuse_activation=reuse_activation
        )


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def participant_shape(value):
    try:
        extents = tuple(int(extent) for extent in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("mesh shape must be ROWSxCOLS") from error
    if len(extents) != 2 or min(extents) != 1 or max(extents) < 2:
        raise argparse.ArgumentTypeError(
            "mesh shape must be a device line, such as 2x1 or 1x4; "
            "native all-gather operates along one axis"
        )
    return extents


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("both", "ttlang", "ttmetal"), default="both"
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--gather-output", action="store_true")
    parser.add_argument("--collective-only", choices=("activation", "output"))
    parser.add_argument(
        "--activation-all-gather", choices=("all_to_all", "ring"), default="all_to_all"
    )
    parser.add_argument(
        "--output-all-gather", choices=("all_to_all", "ring"), default="all_to_all"
    )
    parser.add_argument("--output-gather-block-tiles", type=positive_int)
    parser.add_argument("--output-gather-m-block-tiles", type=positive_int, default=1)
    parser.add_argument("--output-gather-workers", type=positive_int, default=2)
    parser.add_argument(
        "--variant", choices=("n_sharded", "replicated"), default="n_sharded"
    )
    parser.add_argument("--trace", action="store_true", help="time device trace replay")
    parser.add_argument(
        "--n-tiles",
        type=positive_int,
        help="complete output width in tiles; required with --gather-output",
    )
    parser.add_argument("--math-fidelity", choices=("HiFi2", "HiFi4"))
    parser.add_argument(
        "--fp32-dest-acc", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--native-channel-buffers", type=positive_int, default=24)
    parser.add_argument("--native-num-links", type=positive_int, default=1)
    parser.add_argument("--native-workers-per-link", type=positive_int)
    parser.add_argument(
        "--native-subblock",
        type=positive_int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--device-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument(
        "--mesh-shape",
        type=participant_shape,
        help="participant line ROWSxCOLS; default: two devices on a discovered axis",
    )
    parser.add_argument("--worker-timeout", type=positive_int, default=180)
    parser.add_argument(
        "--fabric-config",
        choices=("auto", *FABRIC_CONFIGS),
        default="auto",
        help="fabric routing; auto selects 1d for two devices and 1d-ring otherwise",
    )
    parser.add_argument(
        "--fabric-reliability",
        choices=FABRIC_RELIABILITY_MODES,
        default="relaxed",
    )
    parser.add_argument("--fabric-router-payload", type=positive_int)
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


def validate_native_subblock(config, subblock_shape, fp32_dest_acc):
    subblock_shape = tuple(subblock_shape)
    destination_capacity = 4 if fp32_dest_acc else 8
    if (
        len(subblock_shape) != 2
        or min(subblock_shape) <= 0
        or config.m_block_tiles % subblock_shape[0]
        or config.n_block_tiles % subblock_shape[1]
        or prod(subblock_shape) > destination_capacity
    ):
        raise ValueError(
            "native subblock must divide the output block and fit destination registers"
        )
    return subblock_shape


def mesh_selection(discovered_shape, requested_shape=None):
    """Keep an existing device line or reshape the parent before taking a submesh."""
    discovered_shape = tuple(discovered_shape)
    if len(discovered_shape) != 2 or prod(discovered_shape) < 2:
        raise ValueError(
            f"requires a connected two-dimensional mesh: {discovered_shape}"
        )
    if requested_shape is None:
        participant_axis = next(
            axis for axis, extent in enumerate(discovered_shape) if extent >= 2
        )
        requested_shape = tuple(
            2 if axis == participant_axis else 1 for axis in range(2)
        )
    else:
        requested_shape = participant_shape("x".join(map(str, requested_shape)))
        participant_axis = next(
            axis for axis, extent in enumerate(requested_shape) if extent > 1
        )
    if prod(requested_shape) > prod(discovered_shape):
        raise ValueError(
            f"requested mesh {requested_shape} exceeds discovered mesh {discovered_shape}"
        )
    parent_shape = discovered_shape
    if any(
        requested > available
        for requested, available in zip(requested_shape, discovered_shape)
    ):
        parent_shape = tuple(
            prod(discovered_shape) if axis == participant_axis else 1
            for axis in range(2)
        )
    return requested_shape, participant_axis, parent_shape


def select_fabric_config(requested_shape, fabric_config):
    if fabric_config != "auto":
        return fabric_config
    requested_device_count = 2 if requested_shape is None else prod(requested_shape)
    return "1d" if requested_device_count == 2 else "1d-ring"


def topology_for_fabric_config(fabric_config):
    return ttnn.Topology.Ring if fabric_config == "1d-ring" else ttnn.Topology.Linear


def create_fabric_router_config(max_payload_size):
    router_config = ttnn._ttnn.fabric.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = max_payload_size
    return router_config


@contextmanager
def open_participant_mesh(
    requested_shape=None,
    fabric_config="auto",
    fabric_reliability="relaxed",
    fabric_router_payload=None,
    trace_region_size=0,
):
    parent_mesh = None
    participant_mesh = None
    try:
        selected_fabric_name = select_fabric_config(requested_shape, fabric_config)
        selected_fabric = FABRIC_CONFIGS[selected_fabric_name]
        selected_reliability = FABRIC_RELIABILITY_MODES[fabric_reliability]
        router_config = (
            create_fabric_router_config(fabric_router_payload)
            if fabric_router_payload is not None
            else None
        )
        discovered_shape = get_fabric_mesh_shape(
            fabric_config=selected_fabric,
            reliability_mode=selected_reliability,
            router_config=router_config,
        )
        fabric_options = {"reliability_mode": selected_reliability}
        if router_config is not None:
            fabric_options["router_config"] = router_config
        ttnn.set_fabric_config(selected_fabric, **fabric_options)
        mesh_shape, participant_axis, parent_shape = mesh_selection(
            discovered_shape, requested_shape
        )
        parent_mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(discovered_shape), trace_region_size=trace_region_size
        )
        if parent_shape != tuple(discovered_shape):
            parent_mesh.reshape(ttnn.MeshShape(parent_shape))
        participant_mesh = (
            parent_mesh
            if mesh_shape == parent_shape
            else parent_mesh.create_submesh(ttnn.MeshShape(mesh_shape))
        )
        if len(participant_mesh.get_device_ids()) != prod(mesh_shape):
            raise RuntimeError("participant mesh device count does not match its shape")
        yield (
            participant_mesh,
            mesh_shape,
            participant_axis,
            discovered_shape,
            selected_fabric_name,
        )
    finally:
        if participant_mesh is not None and participant_mesh is not parent_mesh:
            ttnn.close_mesh_device(participant_mesh)
        if parent_mesh is not None:
            ttnn.close_mesh_device(parent_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@dataclass
class Workload:
    run: Callable
    gathered: object
    cleanup: Callable
    program_count: int = 1


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
    native_num_links=1,
    native_workers_per_link=None,
    native_subblock_shape=None,
    native_topology=ttnn.Topology.Linear,
    gather_output=False,
    variant="n_sharded",
    activation_all_gather="all_to_all",
    output_all_gather="all_to_all",
    output_gather_block_tiles=None,
    output_gather_m_block_tiles=1,
    output_gather_workers=2,
):
    torch.manual_seed(seed)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    m_elements = config.m_tiles * 32
    k_elements = config.device_count * config.k_tiles_per_device * 32
    n_elements_per_device = config.n_tiles_per_device * 32
    n_elements = (
        n_elements_per_device
        if variant == "replicated" or (gather_output and implementation == "ttmetal")
        else config.device_count * n_elements_per_device
    )
    activation = torch.randn((m_elements, k_elements), dtype=torch_dtype)
    weight = torch.randn((k_elements, n_elements), dtype=torch_dtype) / k_elements**0.5
    bias = torch.randn((1, n_elements), dtype=torch_dtype) * 0.1
    expected_ttlang = activation.float() @ weight.float() + bias.float()
    replicated_result = gather_output or variant == "replicated"
    native_weight = weight if replicated_result else weight[:, :n_elements_per_device]
    native_bias = bias if replicated_result else bias[:, :n_elements_per_device]
    expected_ttmetal = activation.float() @ native_weight.float() + native_bias.float()
    shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=1)
    replicate_mapper = ttnn.ReplicateTensorToMesh(mesh)

    workloads = {}
    for name in (
        ("ttlang", "ttmetal") if implementation == "both" else (implementation,)
    ):
        gathered = None
        activation_storage = activation
        if name == "ttlang" and config.padded_m_tiles != config.m_tiles:
            activation_storage = torch.nn.functional.pad(
                activation, (0, 0, 0, config.padded_m_tiles * 32 - m_elements)
            )
        activation_device = to_dram(activation_storage, mesh, mesh_mapper=shard_mapper)
        if name == "ttlang":
            output_mapper = (
                replicate_mapper if variant == "replicated" else shard_mapper
            )
            weight_device = to_dram(weight, mesh, mesh_mapper=output_mapper)
            bias_device = to_dram(bias, mesh, mesh_mapper=output_mapper)
            output = to_dram(
                torch.zeros((m_elements, n_elements), dtype=torch_dtype),
                mesh,
                mesh_mapper=output_mapper,
            )
            operation_factory = (
                make_replicated_all_gather_matmul_operation
                if variant == "replicated"
                else make_all_gather_minimal_matmul_operation
            )
            operation = operation_factory(
                config,
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc,
                all_gather_algorithm=activation_all_gather,
            )
            activation_gathered = None
            if variant == "replicated" and config.device_count > 1:
                activation_gathered = to_dram(
                    torch.zeros(
                        (config.padded_m_tiles * 32, k_elements), dtype=torch_dtype
                    ),
                    mesh,
                    mesh_mapper=replicate_mapper,
                )
            replicated_output = None
            output_gather = None
            if gather_output:
                replicated_output = to_dram(
                    torch.zeros((m_elements, n_elements), dtype=torch_dtype),
                    mesh,
                    mesh_mapper=replicate_mapper,
                )
                output_gather = make_column_all_gather(
                    config.mesh_shape,
                    m_tiles=config.m_tiles,
                    n_tiles_per_device=config.n_tiles_per_device,
                    worker_count=output_gather_workers,
                    block_tiles=output_gather_block_tiles or config.n_block_tiles,
                    m_block_tiles=output_gather_m_block_tiles,
                    algorithm=output_all_gather,
                )
                # Allocate the smaller collective first to preserve contiguous
                # L1 space below its persistent buffers for matmul.
                output_gather(output, replicated_output)
                ttnn.synchronize_device(mesh)

            def run_ttlang(
                operation=operation,
                output=output,
                weight_device=weight_device,
                bias_device=bias_device,
                replicated_output=replicated_output,
                output_gather=output_gather,
                activation_device=activation_device,
                activation_gathered=activation_gathered,
            ):
                operands = (activation_device, weight_device, bias_device, output)
                if activation_gathered is not None:
                    operation(*operands, activation_gathered)
                else:
                    operation(*operands)
                if output_gather is not None:
                    output_gather(output, replicated_output)
                    return replicated_output
                return output

            program_count = (
                operation.program_count if variant == "replicated" else 1
            ) + int(output_gather is not None)
            workloads[name] = Workload(
                run_ttlang, gathered, lambda result: None, program_count
            )
        else:
            # Match the upstream operation: every device holds the same weight
            # and bias and produces the same M x N result.
            weight_device = to_dram(native_weight, mesh, mesh_mapper=replicate_mapper)
            bias_device = to_dram(native_bias, mesh, mesh_mapper=replicate_mapper)
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
            selected_subblock = native_subblock_shape or native_subblock(config)
            native_config = ttnn.MinimalMatmulConfig(
                M_block_size=config.m_block_tiles,
                K_block_size=config.compute_k_tiles,
                N_block_size=config.n_block_tiles,
                subblock_h=selected_subblock[0],
                subblock_w=selected_subblock[1],
                compute_with_storage_grid_size=ttnn.CoreCoord(*config.grid),
            )
            compute_config = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=MATH_FIDELITIES[math_fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc,
                packer_l1_acc=True,
            )

            def run_ttmetal(
                activation_device=activation_device,
                gathered=gathered,
                semaphores=semaphores,
                weight_device=weight_device,
                bias_device=bias_device,
            ):
                result = ttnn.experimental.all_gather_minimal_matmul_async(
                    activation_device,
                    weight_device,
                    bias_tensor=bias_device,
                    config=native_config,
                    compute_kernel_config=compute_config,
                    persistent_output_buffer=gathered,
                    multi_device_global_semaphore=semaphores,
                    topology=native_topology,
                    cluster_axis=cluster_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    force_transpose=config.transpose,
                    num_links=native_num_links,
                    num_workers_per_link=(native_workers_per_link or config.m_workers),
                    num_buffers_per_channel=native_channel_buffers,
                )
                if len(result) != 1:
                    raise RuntimeError(f"expected one native output, got {len(result)}")
                return result[0]

            workloads[name] = Workload(run_ttmetal, gathered, ttnn.deallocate)

    def validate(name, output, gathered):
        output_concat_dimension = 1 if name == "ttlang" and not replicated_result else 0
        output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=output_concat_dimension),
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
            expected_output = expected_ttmetal.repeat(config.device_count, 1)
        else:
            expected_output = (
                expected_ttlang.repeat(config.device_count, 1)
                if replicated_result
                else expected_ttlang
            )
        assert_pcc(
            expected_output,
            output_torch,
            threshold=0.999 if dtype == "fp32" else 0.99,
        )
        # The FPU truncates FP32 sources to TF32 even with FP32 destinations.
        tolerance = 0.005 if dtype == "fp32" else 0.05
        assert_allclose(
            output_torch,
            expected_output,
            rtol=tolerance,
            atol=tolerance,
        )
        absolute_error = (output_torch - expected_output).abs()
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
    for name, workload in workloads.items():
        run, gathered, cleanup = workload.run, workload.gathered, workload.cleanup
        print(f"Compile and validate {name}", flush=True)
        for _iteration in range(arguments.warmup):
            output = run()
            ttnn.synchronize_device(mesh)
            correctness[name].append(validate(name, output, gathered))
            cleanup(output)

    samples = {name: [] for name in workloads}
    previous_run_ids = {}
    for sample_index in range(arguments.samples):
        for name, workload in workloads.items():
            run, gathered, cleanup = workload.run, workload.gathered, workload.cleanup
            preparation = (
                prepare_device_invocation(run)
                if arguments.trace and name == "ttlang"
                else nullcontext(run)
            )
            with (
                preparation as prepared_run,
                measured_invocation(
                    prepared_run, cleanup, mesh, trace=arguments.trace
                ) as output,
            ):
                profiler_log = read_device_profile(mesh)
                duration = latest_kernel_duration(
                    profiler_log,
                    mesh.get_device_ids(),
                    aggregation=arguments.device_aggregation,
                    program_count=workload.program_count,
                )
                correctness[name].append(validate(name, output, gathered))
            for device_id, device in duration["per_device"].items():
                run_id = device["run_host_id"]
                if run_id <= previous_run_ids.get(device_id, -1):
                    raise ValueError(f"stale profiler data for device {device_id}")
                previous_run_ids[device_id] = run_id
            samples[name].append(duration)
            print(
                f"{name} sample {sample_index + 1}: {duration['cycles']} cycles, "
                f"{duration['us']:.3f} us device interval",
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


def create_collective_workload(mesh, config, arguments):
    """Measure the same column-sharded payload under either collective algorithm."""
    activation = arguments.collective_only == "activation"
    shard_tiles = config.k_tiles_per_device if activation else config.n_tiles_per_device
    block_tiles = (
        config.k_block_tiles
        if activation
        else arguments.output_gather_block_tiles or config.n_block_tiles
    )
    algorithm = (
        arguments.activation_all_gather if activation else arguments.output_all_gather
    )
    dtype = torch.bfloat16 if arguments.dtype == "bf16" else torch.float32
    torch.manual_seed(arguments.seed)
    expected = torch.randn(
        (config.m_tiles * 32, shard_tiles * config.device_count * 32), dtype=dtype
    )
    input_shard = to_dram(
        expected, mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1)
    )
    output = to_dram(
        torch.zeros_like(expected), mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    collective = make_column_all_gather(
        config.mesh_shape,
        m_tiles=config.m_tiles,
        n_tiles_per_device=shard_tiles,
        worker_count=config.m_workers,
        block_tiles=block_tiles,
        m_block_tiles=(
            config.m_block_tiles
            if activation
            else arguments.output_gather_m_block_tiles
        ),
        algorithm=algorithm,
    )

    def run():
        collective(input_shard, output)
        return output

    def validate(name, result, gathered):
        actual = ttnn.to_torch(
            result, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float()
        for replica in actual.split(config.m_tiles * 32, dim=0):
            assert_allclose(replica, expected.float(), rtol=0, atol=0)
        return {"max_abs_error": 0.0, "rtol": 0, "atol": 0}

    return {"ttlang": (run, None, lambda result: None)}, validate


@contextmanager
def measured_invocation(run, cleanup, mesh, *, trace):
    trace_id = None
    output = None
    try:
        if trace:
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                output = run()
            finally:
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        else:
            output = run()
        ttnn.synchronize_device(mesh)
        yield output
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        if output is not None:
            cleanup(output)


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
            timeout=arguments.worker_timeout,
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
    if arguments.collective_only:
        if arguments.implementation != "ttlang" or arguments.gather_output:
            raise ValueError(
                "--collective-only requires --implementation ttlang without --gather-output"
            )
        arguments.trace = True
    if arguments.variant == "replicated" and arguments.gather_output:
        raise ValueError("replicated weights already produce replicated output")
    if arguments.gather_output:
        if arguments.n_tiles is None:
            raise ValueError(
                "--gather-output requires --n-tiles (complete output width)"
            )
        arguments.trace = True
    elif arguments.variant == "replicated":
        arguments.trace = True
    elif arguments.n_tiles is not None:
        raise ValueError("--n-tiles currently requires --gather-output")
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
                Path(__file__).resolve().parents[2]
                / "examples/all_gather_minimal_matmul/replicated/operation.py",
                Path(__file__).resolve().parents[2]
                / "examples/all_gather_minimal_matmul/collectives.py",
                Path(__file__).resolve().parents[2] / "python/ttl/kernel_runner.py",
            ]
        ),
        "arguments": {**vars(arguments), "json": str(arguments.json)},
        "methodology_reference": {
            "sweep": f"{REFERENCE_ROOT}/models/tt_dit/utils/sweep_mm_block_sizes.py",
            "test": f"{REFERENCE_ROOT}/models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py",
        },
    }
    with open_participant_mesh(
        arguments.mesh_shape,
        arguments.fabric_config,
        arguments.fabric_reliability,
        arguments.fabric_router_payload,
        trace_region_size=4194304 if arguments.trace else 0,
    ) as (
        mesh,
        mesh_shape,
        cluster_axis,
        discovered_shape,
        selected_fabric_name,
    ):
        n_tiles_per_device = arguments.n_tiles_per_device
        if arguments.n_tiles is not None:
            n_partitions = (
                prod(mesh_shape)
                if arguments.implementation == "ttlang"
                and arguments.variant == "n_sharded"
                else 1
            )
            if arguments.n_tiles % n_partitions:
                raise ValueError(
                    "complete N tile count must be divisible by device count"
                )
            n_tiles_per_device = arguments.n_tiles // n_partitions
        benchmark_case = BenchmarkCase(
            mesh_shape=mesh_shape,
            m_tiles=arguments.m_tiles,
            k_tiles_per_device=arguments.k_tiles_per_device,
            n_tiles_per_device=n_tiles_per_device,
            m_block_tiles=arguments.m_block_tiles,
            k_block_tiles=arguments.k_block_tiles,
            n_block_tiles=arguments.n_block_tiles,
            worker_grid=arguments.worker_grid,
            transpose=arguments.transpose,
        )
        if benchmark_case.k_tiles_per_device % benchmark_case.k_block_tiles:
            raise ValueError("k_tiles_per_device must be divisible by k_block_tiles")
        config = (
            benchmark_case.make_ttlang_config(arguments.reuse_activation)
            if arguments.implementation == "ttlang"
            else benchmark_case
        )
        selected_native_topology = topology_for_fabric_config(selected_fabric_name)
        native_workers_per_link = arguments.native_workers_per_link or config.m_workers
        selected_native_subblock = validate_native_subblock(
            config,
            arguments.native_subblock or native_subblock(config),
            arguments.fp32_dest_acc,
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
            fabric_config=str(FABRIC_CONFIGS[selected_fabric_name]),
            fp32_dest_acc_en=arguments.fp32_dest_acc,
            native_channel_buffers=arguments.native_channel_buffers,
            device_aggregation=arguments.device_aggregation,
            gather_output=arguments.gather_output,
            variant=arguments.variant,
            activation_all_gather=arguments.activation_all_gather,
            output_all_gather=arguments.output_all_gather,
            output_gather_block_tiles=(
                arguments.output_gather_block_tiles or config.n_block_tiles
            ),
            output_gather_m_block_tiles=arguments.output_gather_m_block_tiles,
            output_gather_workers=arguments.output_gather_workers,
            activation_storage_m=(
                config.padded_m_tiles * 32
                if arguments.implementation == "ttlang"
                else config.m_tiles * 32
            ),
            global_n_tiles=arguments.n_tiles,
            layout="TILE",
            memory="DRAM",
            per_device_matmul={
                "m": config.m_tiles * 32,
                "gathered_k": config.k_tiles_per_device * config.device_count * 32,
                "n": config.n_tiles_per_device * 32,
                "local_k": config.k_tiles_per_device * 32,
            },
            tensor_placement={
                "ttlang": {
                    "activation": "K-sharded",
                    "weight": (
                        "replicated"
                        if arguments.variant == "replicated"
                        else "N-sharded"
                    ),
                    "bias": (
                        "replicated"
                        if arguments.variant == "replicated"
                        else "N-sharded"
                    ),
                    "output": (
                        "replicated"
                        if arguments.gather_output or arguments.variant == "replicated"
                        else "N-sharded"
                    ),
                },
                "ttmetal": {
                    "activation": "K-sharded",
                    "weight": "replicated",
                    "bias": "replicated",
                    "output": "replicated",
                },
            },
            native_config={
                "topology": str(selected_native_topology),
                "links": arguments.native_num_links,
                "workers_per_link": native_workers_per_link,
                "compute_k_tiles": config.compute_k_tiles,
                "subblock": list(selected_native_subblock),
                "channel_buffers": arguments.native_channel_buffers,
                "packer_l1_acc": True,
            },
            execution="trace_replay" if arguments.trace else "ordinary_launch",
            fabric_reliability=str(
                FABRIC_RELIABILITY_MODES[arguments.fabric_reliability]
            ),
            fabric_router_payload=(
                arguments.fabric_router_payload or "installed_runtime_default"
            ),
        )
        if arguments.collective_only:
            workloads, validate = create_collective_workload(mesh, config, arguments)
        else:
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
                native_num_links=arguments.native_num_links,
                native_workers_per_link=native_workers_per_link,
                native_subblock_shape=selected_native_subblock,
                native_topology=selected_native_topology,
                gather_output=arguments.gather_output,
                variant=arguments.variant,
                activation_all_gather=arguments.activation_all_gather,
                output_all_gather=arguments.output_all_gather,
                output_gather_block_tiles=arguments.output_gather_block_tiles,
                output_gather_m_block_tiles=arguments.output_gather_m_block_tiles,
                output_gather_workers=arguments.output_gather_workers,
            )
        report["measurements"] = benchmark(workloads, validate, mesh, arguments)
    arguments.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["measurements"], indent=2), flush=True)
    print(f"Results: {arguments.json}", flush=True)


if __name__ == "__main__":
    main()
