# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for all-gather minimal matmul examples."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import ttnn

TILE_SIZE = 32
MAX_RECEIVE_DFB_SLOTS = 32
DEFAULT_M_TILES = 4
DEFAULT_K_TILES_PER_DEVICE = 8
DEFAULT_N_TILES_PER_DEVICE = 8
DEFAULT_K_TILES_PER_TRANSFER = 1
DEFAULT_PCC_THRESHOLD = 0.99
DEFAULT_FABRIC_RELIABILITY_MODE = "relaxed"
TT_METAL_EXAMPLE_DIR = (
    "third-party/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/"
    "all_gather_minimal_matmul_async"
)


@dataclass(frozen=True)
class ExampleConfig:
    mesh_shape: tuple[int, int]
    m_tiles: int
    k_tiles_per_device: int
    n_tiles_per_device: int
    k_tiles_per_transfer: int
    fabric_config: str
    fabric_reliability_mode: str
    seed: int
    pcc_threshold: float

    @property
    def num_devices(self) -> int:
        return num_devices(self.mesh_shape)

    @property
    def m_dim(self) -> int:
        return self.m_tiles * TILE_SIZE

    @property
    def k_dim(self) -> int:
        return self.num_devices * self.k_tiles_per_device * TILE_SIZE

    @property
    def n_dim(self) -> int:
        return self.num_devices * self.n_tiles_per_device * TILE_SIZE


@dataclass(frozen=True)
class HostTensors:
    activation: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class MeshTensors:
    activation_shard: ttnn.Tensor
    weight_shard: ttnn.Tensor
    gathered_activation: ttnn.Tensor
    output_shard: ttnn.Tensor


def num_devices(mesh_shape: tuple[int, int]) -> int:
    return mesh_shape[0] * mesh_shape[1]


def require_device_count(mesh_shape: tuple[int, int]) -> int:
    required_device_count = num_devices(mesh_shape)
    available_device_count = visible_tenstorrent_device_count()
    if available_device_count == 0:
        return required_device_count
    if available_device_count < required_device_count:
        raise RuntimeError(
            f"This example requires at least {required_device_count} devices."
        )
    return required_device_count


def parse_mesh_shape(value: str) -> tuple[int, int]:
    normalized = value.replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mesh shape must be ROWSxCOLS or auto")
    try:
        rows, cols = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mesh shape entries must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("mesh shape entries must be positive")
    return rows, cols


def parse_mesh_shape_or_auto(value: str) -> tuple[int, int] | None:
    if value.lower() == "auto":
        return None
    return parse_mesh_shape(value)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int_at_most_32(value: str) -> int:
    parsed = positive_int(value)
    if parsed > MAX_RECEIVE_DFB_SLOTS:
        raise argparse.ArgumentTypeError(
            f"value must be at most {MAX_RECEIVE_DFB_SLOTS}"
        )
    return parsed


def pcc_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PCC threshold must be a number") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("PCC threshold must be in [0, 1]")
    return parsed


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mesh-shape",
        type=parse_mesh_shape_or_auto,
        default=None,
        help="Logical mesh as ROWSxCOLS, or auto for fabric discovery.",
    )
    parser.add_argument(
        "--max-devices",
        type=positive_int_at_most_32,
        default=MAX_RECEIVE_DFB_SLOTS,
        help="Maximum auto-selected device count. The operation supports at most 32.",
    )
    parser.add_argument("--m-tiles", type=positive_int, default=DEFAULT_M_TILES)
    parser.add_argument(
        "--k-tiles-per-device",
        type=positive_int,
        default=DEFAULT_K_TILES_PER_DEVICE,
    )
    parser.add_argument(
        "--n-tiles-per-device",
        type=positive_int,
        default=DEFAULT_N_TILES_PER_DEVICE,
    )
    parser.add_argument(
        "--k-tiles-per-transfer",
        type=positive_int,
        default=DEFAULT_K_TILES_PER_TRANSFER,
        help="Activation K tiles per fabric transfer.",
    )
    parser.add_argument(
        "--fabric-config",
        choices=("auto", "1d", "2d"),
        default="auto",
        help="Fabric mode. auto selects 1d for linear meshes and 2d otherwise.",
    )
    parser.add_argument(
        "--fabric-reliability-mode",
        choices=("strict", "relaxed", "dynamic"),
        default=DEFAULT_FABRIC_RELIABILITY_MODE,
        help="TTNN fabric initialization reliability mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pcc-threshold",
        type=pcc_threshold,
        default=DEFAULT_PCC_THRESHOLD,
    )
    return parser


def normalize_discovered_mesh_shape(mesh_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(mesh_shape) == 1:
        return 1, int(mesh_shape[0])
    if len(mesh_shape) == 2:
        return int(mesh_shape[0]), int(mesh_shape[1])
    raise RuntimeError(
        "SystemMeshDescriptor returned an unsupported mesh rank: "
        f"{mesh_shape}. This example expects a 1D or 2D fabric mesh."
    )


def discover_fabric_mesh_shape() -> tuple[int, int]:
    snippet = (
        "import ttnn\n"
        "shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()\n"
        "print(tuple(int(extent) for extent in shape))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("SystemMeshDescriptor discovery timed out") from exc

    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(diagnostic.splitlines()[-1])

    for line in reversed(result.stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("("):
            discovered_mesh_shape = ast.literal_eval(stripped)
            return normalize_discovered_mesh_shape(discovered_mesh_shape)

    raise RuntimeError("SystemMeshDescriptor did not print a mesh shape")


def visible_tenstorrent_device_count() -> int:
    device_dir = Path("/dev/tenstorrent")
    if not device_dir.exists():
        return 0
    return sum(1 for path in device_dir.iterdir() if path.name.isdigit())


def factorized_mesh_shape(device_count: int) -> tuple[int, int]:
    if device_count == 32:
        return 4, 8

    best_mesh_shape = (1, device_count)
    best_aspect_error = device_count - 1
    for candidate_rows in range(1, device_count + 1):
        if device_count % candidate_rows != 0:
            continue
        candidate_cols = device_count // candidate_rows
        if candidate_rows > candidate_cols:
            continue
        aspect_error = candidate_cols - candidate_rows
        if aspect_error < best_aspect_error:
            best_mesh_shape = (candidate_rows, candidate_cols)
            best_aspect_error = aspect_error
    return best_mesh_shape


def fallback_mesh_shape(max_devices: int) -> tuple[int, int]:
    available_device_count = visible_tenstorrent_device_count()
    selected_device_count = min(available_device_count, max_devices)
    if selected_device_count <= 0:
        raise RuntimeError("no TT devices are available")
    return factorized_mesh_shape(selected_device_count)


def select_bounded_submesh(
    discovered_mesh_shape: tuple[int, int],
    max_devices: int,
) -> tuple[int, int]:
    discovered_device_count = num_devices(discovered_mesh_shape)
    if discovered_device_count <= max_devices:
        return discovered_mesh_shape

    discovered_rows, discovered_cols = discovered_mesh_shape
    best_mesh_shape = (1, 1)
    best_device_count = 1
    best_aspect_error = abs(discovered_cols - discovered_rows)

    for candidate_rows in range(1, discovered_rows + 1):
        for candidate_cols in range(1, discovered_cols + 1):
            candidate_device_count = candidate_rows * candidate_cols
            if candidate_device_count > max_devices:
                continue

            aspect_error = abs(
                candidate_rows * discovered_cols - candidate_cols * discovered_rows
            )
            better_device_count = candidate_device_count > best_device_count
            better_aspect = (
                candidate_device_count == best_device_count
                and aspect_error < best_aspect_error
            )
            better_row_count = (
                candidate_device_count == best_device_count
                and aspect_error == best_aspect_error
                and candidate_rows > best_mesh_shape[0]
            )
            if better_device_count or better_aspect or better_row_count:
                best_mesh_shape = (candidate_rows, candidate_cols)
                best_device_count = candidate_device_count
                best_aspect_error = aspect_error

    return best_mesh_shape


def resolve_mesh_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.mesh_shape is not None:
        if num_devices(args.mesh_shape) > MAX_RECEIVE_DFB_SLOTS:
            raise RuntimeError(
                "explicit mesh shape exceeds the direct example's 32-device "
                "receive DFB slot limit"
            )
        return args.mesh_shape

    try:
        discovered_mesh_shape = discover_fabric_mesh_shape()
    except Exception as exc:
        selected_mesh_shape = fallback_mesh_shape(args.max_devices)
        print(
            "auto mesh selection: "
            f"fabric discovery failed ({type(exc).__name__}: {exc}); "
            f"selected={selected_mesh_shape}"
        )
        return selected_mesh_shape

    selected_mesh_shape = select_bounded_submesh(
        discovered_mesh_shape, args.max_devices
    )
    if selected_mesh_shape != discovered_mesh_shape:
        print(
            "auto mesh selection: "
            f"discovered={discovered_mesh_shape} selected={selected_mesh_shape}"
        )
    return selected_mesh_shape


def validate_tile_config(
    k_tiles_per_device: int,
    k_tiles_per_transfer: int,
) -> None:
    if k_tiles_per_device % k_tiles_per_transfer != 0:
        raise RuntimeError(
            "k_tiles_per_device must be divisible by k_tiles_per_transfer"
        )


def config_from_args(args: argparse.Namespace) -> ExampleConfig:
    validate_tile_config(args.k_tiles_per_device, args.k_tiles_per_transfer)
    mesh_shape = resolve_mesh_shape(args)
    return ExampleConfig(
        mesh_shape=mesh_shape,
        m_tiles=args.m_tiles,
        k_tiles_per_device=args.k_tiles_per_device,
        n_tiles_per_device=args.n_tiles_per_device,
        k_tiles_per_transfer=args.k_tiles_per_transfer,
        fabric_config=args.fabric_config,
        fabric_reliability_mode=args.fabric_reliability_mode,
        seed=args.seed,
        pcc_threshold=args.pcc_threshold,
    )


def select_fabric_config(mesh_shape: tuple[int, int], requested_config: str):
    if requested_config == "1d":
        if mesh_shape[0] > 1 and mesh_shape[1] > 1:
            raise RuntimeError("FABRIC_1D requires a linear logical mesh")
        return ttnn.FabricConfig.FABRIC_1D
    if requested_config == "2d":
        return ttnn.FabricConfig.FABRIC_2D
    if mesh_shape[0] == 1 or mesh_shape[1] == 1:
        return ttnn.FabricConfig.FABRIC_1D
    return ttnn.FabricConfig.FABRIC_2D


def select_fabric_reliability_mode(requested_mode: str):
    if requested_mode == "strict":
        return ttnn.FabricReliabilityMode.STRICT_INIT
    if requested_mode == "dynamic":
        return ttnn.FabricReliabilityMode.DYNAMIC_RECONFIG
    return ttnn.FabricReliabilityMode.RELAXED_INIT


@contextmanager
def open_configured_mesh(config: ExampleConfig) -> Iterator[object]:
    ttnn.set_fabric_config(
        select_fabric_config(config.mesh_shape, config.fabric_config),
        select_fabric_reliability_mode(config.fabric_reliability_mode),
    )
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*config.mesh_shape))
    try:
        yield mesh_device
    finally:
        close_mesh_device = getattr(ttnn, "close_mesh_device", ttnn.close_device)
        close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def from_torch(
    tensor: torch.Tensor,
    mesh_device,
    mesh_mapper,
):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def make_host_tensors(config: ExampleConfig) -> HostTensors:
    activation = torch.randn((config.m_dim, config.k_dim), dtype=torch.bfloat16)
    weight = torch.randn((config.k_dim, config.n_dim), dtype=torch.bfloat16)
    weight = weight / float(config.k_dim)
    return HostTensors(activation=activation, weight=weight)


def make_mesh_tensors(
    config: ExampleConfig,
    mesh_device,
    host_tensors: HostTensors,
) -> MeshTensors:
    activation_shard = from_torch(
        host_tensors.activation,
        mesh_device,
        ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    weight_shard = from_torch(
        host_tensors.weight,
        mesh_device,
        ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    gathered_activation = from_torch(
        torch.zeros_like(host_tensors.activation),
        mesh_device,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    output_shard = from_torch(
        torch.zeros((config.m_dim, config.n_dim), dtype=torch.bfloat16),
        mesh_device,
        ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    return MeshTensors(
        activation_shard=activation_shard,
        weight_shard=weight_shard,
        gathered_activation=gathered_activation,
        output_shard=output_shard,
    )


def assert_pcc(expected: torch.Tensor, result: torch.Tensor, threshold: float) -> None:
    expected_flat = expected.flatten().float()
    result_flat = result.flatten().float()
    pcc = torch.corrcoef(torch.stack([result_flat, expected_flat]))[0, 1].item()
    print(f"PCC {pcc:.6f}")
    if pcc < threshold:
        raise AssertionError(f"PCC {pcc:.6f} is below {threshold:.6f}")


def assert_gathered_activation(
    expected: torch.Tensor,
    result: torch.Tensor,
    config: ExampleConfig,
) -> None:
    mismatches = []
    local_m_rows = config.m_tiles * TILE_SIZE
    local_k_cols = config.k_tiles_per_device * TILE_SIZE
    for device_index in range(config.num_devices):
        row_begin = device_index * local_m_rows
        row_end = row_begin + local_m_rows
        device_result = result[row_begin:row_end, :]
        for shard_index in range(config.num_devices):
            col_begin = shard_index * local_k_cols
            col_end = col_begin + local_k_cols
            result_shard = device_result[:, col_begin:col_end]
            expected_shard = expected[:, col_begin:col_end]
            if not torch.equal(result_shard, expected_shard):
                max_abs = torch.max(
                    torch.abs(result_shard.float() - expected_shard.float())
                ).item()
                mismatches.append((device_index, shard_index, max_abs))

    if mismatches:
        details = ", ".join(
            f"device {device_index} shard {shard_index} max_abs={max_abs:.6g}"
            for device_index, shard_index, max_abs in mismatches
        )
        raise AssertionError(f"gathered activation mismatch: {details}")


def validate_results(
    config: ExampleConfig,
    mesh_device,
    host_tensors: HostTensors,
    mesh_tensors: MeshTensors,
) -> None:
    gathered_result = ttnn.to_torch(
        mesh_tensors.gathered_activation,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    assert_gathered_activation(host_tensors.activation, gathered_result, config)

    output_result = ttnn.to_torch(
        mesh_tensors.output_shard,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )
    expected_output = host_tensors.activation.float() @ host_tensors.weight.float()
    assert_pcc(expected_output, output_result.float(), threshold=config.pcc_threshold)


def print_config(label: str, config: ExampleConfig) -> None:
    print(
        f"{label}: "
        f"mesh={config.mesh_shape} devices={config.num_devices} "
        f"M={config.m_dim} K={config.k_dim} N={config.n_dim} "
        f"K_transfer_tiles={config.k_tiles_per_transfer} "
        f"fabric_reliability={config.fabric_reliability_mode}"
    )
