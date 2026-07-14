# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared kernel execution logic for tt-lang.

Provides functions for building kernel descriptors, CB descriptors, and
executing kernels on device via ttnn.generic_op. Used by both the Python
DSL (CompiledTTNNKernel) and ME2E tests.

This module provides a single reusable implementation of kernel argument
building and execution.
"""

from dataclasses import dataclass, field
import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

ttnn = None  # Lazy-loaded via _ensure_ttnn()


def _ensure_ttnn():
    """Lazy import of ttnn."""
    global ttnn
    if ttnn is not None:
        return ttnn
    try:
        import ttnn as _ttnn

        ttnn = _ttnn
    except (ModuleNotFoundError, ImportError):
        pass
    return ttnn


from .dataflow_buffer import CompilerAllocatedDFBConfig
from .constants import DEFAULT_L1_CB_BUDGET_BYTES
from .dtype_utils import (
    format_name_to_ttnn_dtype,
    tile_bytes_from_dtype,
    torch_dtype_to_ttnn_datatype,
)


def _cb_data_format(cb):
    """ttnn data format for a DataflowBuffer, from its (torch or ttnn) dtype."""
    dtype = cb.dtype
    if hasattr(dtype, "name"):  # already a ttnn.DataType enum
        return dtype
    return torch_dtype_to_ttnn_datatype(dtype)


def get_min_remaining_l1_for_device(device):
    """Return the minimum remaining L1 CB budget (bytes) across all cores.

    Accounts for reduced ``worker_l1_size`` and L1 tensor allocations.
    Queries ``cb_limit`` (the hardware CB budget) and subtracts the maximum
    per-core L1 buffer usage reported by the device.

    ``get_buffer_pages`` is called on the original device rather than on
    per-coordinate submeshes because ``create_submesh`` produces a new
    device view that does not inherit buffer tracking from the parent.
    For mesh devices this reports allocations for the first physical
    device, which is representative because tt-lang distributes tensors
    uniformly across the mesh. If individual physical devices need tracking,
    ttnn.reports.get_buffer_pages would have to report allocations on the
    parent mesh instead of the first device within the mesh.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    info = ttnn._ttnn.reports.get_device_info(device)
    budget_bytes = info.cb_limit

    bytes_per_core: dict[tuple[int, int], int] = {}
    for page in ttnn._ttnn.reports.get_buffer_pages(device):
        if page.buffer_type == ttnn.BufferType.L1:
            key = (page.core_y, page.core_x)
            bytes_per_core[key] = bytes_per_core.get(key, 0) + page.page_size

    max_core_bytes = max(bytes_per_core.values()) if bytes_per_core else 0
    return max(0, budget_bytes - max_core_bytes)


@dataclass
class KernelSpec:
    """Specification for a single kernel to execute.

    Attributes:
        path: Path to the kernel C++ source file.
        thread_type: Type of kernel ("compute", "noc", or "ethernet").
        tensor_indices: List of global tensor indices this kernel accesses.
            For DM kernels, these determine which buffer addresses go in
            common_runtime_args, in order.
        config: Kernel config descriptor (ComputeConfigDescriptor,
            ReaderConfigDescriptor, WriterConfigDescriptor, or EthernetConfigDescriptor).
        pipe_computed_address_dfb_indices: Receiver DFB indices whose backing
            addresses are passed to this kernel.
        core_ranges: Optional per-kernel ttnn.CoreRangeSet. When set, this
            specialized kernel binary is dispatched only to these cores. When None,
            the whole-grid core_ranges passed to build_kernel_descriptors is used.
        extra_common_runtime_args: Per-kernel runtime args appended after
            shared compiler-managed arguments.
    """

    path: str
    thread_type: str
    tensor_indices: List[int]
    config: Any
    pipe_computed_address_dfb_indices: List[int] = field(default_factory=list)
    core_ranges: Optional[Any] = None
    extra_common_runtime_args: Optional[List[int]] = None


@dataclass
class PipeRuntimeResources:
    """Host allocations and runtime args for compiler-emitted pipe resources."""

    scratch_tensors: List[Any]
    global_semaphores: List[Any]
    computed_address_dfb_tensors: Dict[int, Any]
    computed_address_base_addresses: Dict[int, int]
    extra_common_runtime_args: List[int]
    expected_extra_common_runtime_args: int


@dataclass(frozen=True)
class FabricRouteSpec:
    """One logical local-to-remote route used by a generated kernel."""

    local_device: Tuple[int, ...]
    remote_device: Tuple[int, ...]
    source_nodes: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class _ResolvedFabricRoute:
    """Host-resolved values needed to configure one fabric connection."""

    connection_node_id: Any
    direction: int
    link_index: int
    hop_count: int


class _FabricRouteCache:
    """Cache control-plane route queries for one mesh and fabric configuration."""

    def __init__(self) -> None:
        self._mesh_device = None
        self._fabric_config = None
        self._routes: Dict[Tuple[int, int, int, int], _ResolvedFabricRoute] = {}

    @staticmethod
    def _node_key(node_id: Any) -> Tuple[int, int]:
        return (int(node_id.mesh_id), int(node_id.chip_id))

    def resolve(
        self,
        mesh_device: Any,
        source_node_id: Any,
        destination_node_id: Any,
    ) -> _ResolvedFabricRoute:
        fabric_config = ttnn.get_fabric_config()
        if self._mesh_device is not mesh_device or self._fabric_config != fabric_config:
            self._mesh_device = mesh_device
            self._fabric_config = fabric_config
            self._routes.clear()

        route_key = (
            *self._node_key(source_node_id),
            *self._node_key(destination_node_id),
        )
        if route_key not in self._routes:
            route_info = ttnn.get_fabric_route_info(source_node_id, destination_node_id)
            self._routes[route_key] = _ResolvedFabricRoute(
                connection_node_id=route_info.connection_node_id,
                direction=int(route_info.direction),
                link_index=int(route_info.link_index),
                hop_count=int(route_info.hop_count),
            )
        return self._routes[route_key]


@dataclass(frozen=True)
class MeshProgramPlacement:
    """Device range for one program inside a mesh descriptor."""

    start: Any
    end: Optional[Any] = None


def build_tensor_accessor_args(tensors: List[Any]) -> List[int]:
    """
    Build compile-time args for tensor accessors.

    Args:
        tensors: List of ttnn.Tensor objects on device.

    Returns:
        List of compile-time args (flattened TensorAccessorArgs for all tensors).
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    args = []
    for tensor in tensors:
        tensor_args = ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
        args.extend(tensor_args)
    return args


def build_kernel_descriptors(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    tensor_accessor_args: List[int],
    core_ranges: Any,
    grid_cols: int,
    grid_rows: int,
    num_cbs: int,
    pipe_computed_address_base_addresses: Optional[Dict[int, int]] = None,
    extra_common_runtime_args: Optional[List[int]] = None,
    expected_extra_common_runtime_args: Optional[int] = None,
    device_coordinates: Optional[List[int]] = None,
) -> List[Any]:
    """
    Build kernel descriptors for ttnn.generic_op.

    Args:
        kernel_specs: List of kernel specifications.
        tensors: List of ttnn.Tensor objects. Position in this list determines
            the global tensor index. Individual kernels access subsets via
            tensor_indices in each KernelSpec.
        tensor_accessor_args: Flattened compile-time args from all tensors.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        grid_cols: Number of grid columns (x dimension).
        grid_rows: Number of grid rows (y dimension).
        num_cbs: Total number of circular buffers (including intermediate CBs).
        pipe_computed_address_base_addresses: L1 base address by receiver DFB index for
            compiler-selected computed pipe addressing. These addresses are
            passed as common runtime arguments.
        extra_common_runtime_args: Compiler-managed common runtime args appended
            after tensor buffer addresses and computed receiver DFB bases.
        expected_extra_common_runtime_args: Expected number of compiler-managed
            pipe runtime args from the compiled resource plan.

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []

    # CB indices are 0, 1, 2, ... for each CB (including intermediate CBs).
    cb_indices = list(range(num_cbs))
    computed_address_base_addresses = pipe_computed_address_base_addresses or {}
    extra_args = list(extra_common_runtime_args or [])
    if (
        expected_extra_common_runtime_args is not None
        and len(extra_args) != expected_extra_common_runtime_args
    ):
        raise RuntimeError(
            "pipe resource plan expected "
            f"{expected_extra_common_runtime_args} extra common runtime args, "
            f"got {len(extra_args)}"
        )

    for spec in kernel_specs:
        # Build common_runtime_args using tensor_indices.
        # C++ indexes by function-local position, we provide addresses in that order.
        common_runtime_args = [
            tensors[idx].buffer_address() for idx in spec.tensor_indices
        ]
        computed_address_base_args = []
        for dfb_index in spec.pipe_computed_address_dfb_indices:
            if dfb_index not in computed_address_base_addresses:
                raise RuntimeError(
                    f"missing computed-address receiver DFB base for DFB {dfb_index}"
                )
            computed_address_base_args.append(
                computed_address_base_addresses[dfb_index]
            )
        common_runtime_args.extend(computed_address_base_args)
        common_runtime_args.extend(extra_args)
        common_runtime_args.extend(device_coordinates or [])
        common_runtime_args.extend(spec.extra_common_runtime_args or [])

        # Compile-time args are DFB indices followed by TensorAccessorArgs for
        # data-movement kernels. Allocation-dependent DFB bases remain runtime
        # args so cached programs do not retain stale addresses.
        if spec.thread_type == "compute":
            kernel_compile_time_args = cb_indices
        else:
            kernel_compile_time_args = cb_indices + list(tensor_accessor_args)

        # Prefer per-kernel core_ranges (specialize-cores clones); otherwise
        # fall back to the whole-grid core_ranges.
        kernel_ranges = (
            spec.core_ranges if spec.core_ranges is not None else core_ranges
        )

        kernel_desc = ttnn.KernelDescriptor(
            kernel_source=spec.path,
            core_ranges=kernel_ranges,
            compile_time_args=kernel_compile_time_args,
            common_runtime_args=common_runtime_args,
            config=spec.config,
        )
        kernel_descriptors.append(kernel_desc)

    return kernel_descriptors


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _first_device(tensors: List[Any]) -> Any:
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, "device"):
            device = tensor.device()
            if device is not None:
                return device
    raise ValueError("pipe runtime resource allocation requires a device tensor")


def _allocate_l1_sharded_storage_tensor(core_ranges: Any, num_bytes: int, device: Any):
    """Allocate row-major L1 storage with one 4-byte element per storage word."""
    aligned_bytes = _align_up(num_bytes, 32)
    elements_per_core = max(1, aligned_bytes // 4)
    grid_size = core_ranges.bounding_box().grid_size()
    num_cores = grid_size.x * grid_size.y
    shard_spec = ttnn.ShardSpec(
        core_ranges,
        (1, elements_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.empty(
        (num_cores, elements_per_core),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _get_cb_descriptor_rows(cb_configs: List[Any]) -> List[Any]:
    rows = []
    for cb_index, cb in enumerate(cb_configs):
        if cb is None:
            raise ValueError(
                f"Missing CB config for index {cb_index}. "
                f"All DFB indices must have associated DataflowBuffer configurations."
            )

        if isinstance(cb, tuple) and len(cb) == 5:
            shape, block_count, data_format, page_size, total_size = cb
            rows.append(
                (
                    data_format,
                    page_size,
                    total_size,
                    f"  CB[{cb_index}]: shape={shape} block_count={block_count} -> {total_size} bytes",
                )
            )
        elif isinstance(cb, CompilerAllocatedDFBConfig):
            data_format = format_name_to_ttnn_dtype(cb.data_format)
            page_size = tile_bytes_from_dtype(data_format)
            total_size = cb.num_tiles * cb.block_count * page_size
            rows.append(
                (
                    data_format,
                    page_size,
                    total_size,
                    f"  CB[{cb_index}]: compiler-allocated num_tiles={cb.num_tiles} "
                    f"block_count={cb.block_count} format={cb.data_format} -> {total_size} bytes",
                )
            )
        else:
            data_format = _cb_data_format(cb)
            page_size = tile_bytes_from_dtype(data_format)
            num_tiles = cb.shape[0] * cb.shape[1] * cb.block_count
            total_size = num_tiles * page_size
            rows.append(
                (
                    data_format,
                    page_size,
                    total_size,
                    f"  CB[{cb_index}]: shape={cb.shape} block_count={cb.block_count} -> {total_size} bytes",
                )
            )

    return rows


def build_pipe_sram_scratch_tensors(
    tensors: List[Any],
    core_ranges: Any,
    scratch_bytes: int,
    device: Optional[Any] = None,
) -> List[Any]:
    """Allocate per-core SRAM scratch tensors used by PipeNet metadata."""
    if scratch_bytes <= 0:
        return []

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    device = device if device is not None else _first_device(tensors)
    # [Device 2.0] This encodes compiler SRAM as a sharded TTNN tensor because
    # current generic_op has no typed device-side scratch allocation object.
    return [_allocate_l1_sharded_storage_tensor(core_ranges, scratch_bytes, device)]


def build_pipe_global_semaphores(
    tensors: List[Any],
    core_ranges: Any,
    count: int,
    device: Optional[Any] = None,
) -> Tuple[List[Any], List[int]]:
    """Allocate GlobalSemaphores used by PipeNet synchronization counters.

    A MeshDevice GlobalSemaphore has one common L1 address on the selected nodes
    of every device. Fabric atomics target the receiver device's instance at that
    address; node-local PipeNets can use the same storage when local semaphore
    ids are exhausted.
    """
    if count <= 0:
        return [], []

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    device = device if device is not None else _first_device(tensors)
    # [Device 2.0] Keep this allocation behind the pipe resource plan so future
    # typed semaphore objects replace only this host/runtime binding.
    semaphores = [
        ttnn.create_global_semaphore(device, core_ranges, 0) for _ in range(count)
    ]
    addresses = [int(ttnn.get_global_semaphore_address(sem)) for sem in semaphores]
    return semaphores, addresses


def build_pipe_computed_address_dfb_tensors(
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
    pipe_computed_address_dfb_indices: Optional[List[int]] = None,
    device: Optional[Any] = None,
) -> Dict[int, Any]:
    """Allocate hidden L1 backing tensors for computed pipe receiver DFBs."""
    dfb_indices = sorted(set(pipe_computed_address_dfb_indices or []))
    if not dfb_indices:
        return {}

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    rows = _get_cb_descriptor_rows(cb_configs)
    device = device if device is not None else _first_device(tensors)
    backing_tensors = {}
    for dfb_index in dfb_indices:
        if dfb_index < 0 or dfb_index >= len(rows):
            raise ValueError(
                f"computed-address receiver DFB index {dfb_index} is invalid"
            )
        total_size = rows[dfb_index][2]
        backing_tensors[dfb_index] = _allocate_l1_sharded_storage_tensor(
            core_ranges, total_size, device
        )
    return backing_tensors


def build_pipe_runtime_resources(
    tensors: List[Any],
    core_ranges: Any,
    cb_configs: Optional[List[Any]] = None,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    pipe_computed_address_dfb_indices: Optional[List[int]] = None,
    device: Optional[Any] = None,
) -> PipeRuntimeResources:
    """Allocate pipe resources and build their appended common runtime args."""
    computed_address_dfb_indices = list(pipe_computed_address_dfb_indices or [])
    resource_device = device
    if resource_device is None and (
        pipe_sram_scratch_bytes > 0
        or num_pipe_global_semaphores > 0
        or computed_address_dfb_indices
    ):
        resource_device = _first_device(tensors)

    computed_address_dfb_tensors = {}
    if computed_address_dfb_indices:
        if cb_configs is None:
            raise ValueError(
                "computed-address receiver DFB base allocation requires DFB configs"
            )
        computed_address_dfb_tensors = build_pipe_computed_address_dfb_tensors(
            tensors=tensors,
            cb_configs=cb_configs,
            core_ranges=core_ranges,
            pipe_computed_address_dfb_indices=computed_address_dfb_indices,
            device=resource_device,
        )

    scratch_tensors = build_pipe_sram_scratch_tensors(
        tensors=tensors,
        core_ranges=core_ranges,
        scratch_bytes=pipe_sram_scratch_bytes,
        device=resource_device,
    )
    global_semaphores, global_semaphore_addresses = build_pipe_global_semaphores(
        tensors=tensors,
        core_ranges=core_ranges,
        count=num_pipe_global_semaphores,
        device=resource_device,
    )
    # Keep this order in sync with PipeLowering.cpp: optional SRAM scratch base,
    # then GlobalSemaphore counter addresses.
    # [Device 2.0] This is the current ABI for pipe resource records; future
    # typed resource handles should preserve the same compiler-selected order.
    extra_common_runtime_args = [tensor.buffer_address() for tensor in scratch_tensors]
    extra_common_runtime_args.extend(global_semaphore_addresses)
    expected_extra_common_runtime_args = (
        len(scratch_tensors) + num_pipe_global_semaphores
    )
    computed_address_base_addresses = {
        dfb_index: int(tensor.buffer_address())
        for dfb_index, tensor in computed_address_dfb_tensors.items()
    }
    if os.environ.get("TTLANG_DEBUG_FABRIC_ARGS"):
        for dfb_index, tensor in computed_address_dfb_tensors.items():
            device_addresses = [
                int(device_tensor.buffer_address())
                for device_tensor in ttnn.get_device_tensors(tensor)
            ]
            print(
                "computed DFB addresses:",
                dfb_index,
                computed_address_base_addresses[dfb_index],
                device_addresses,
                flush=True,
            )
    return PipeRuntimeResources(
        scratch_tensors=scratch_tensors,
        global_semaphores=global_semaphores,
        computed_address_dfb_tensors=computed_address_dfb_tensors,
        computed_address_base_addresses=computed_address_base_addresses,
        extra_common_runtime_args=extra_common_runtime_args,
        expected_extra_common_runtime_args=expected_extra_common_runtime_args,
    )


def build_pipe_sync_semaphore_descriptors(
    core_ranges: Any,
    count: int,
) -> List[Any]:
    """Build local semaphore descriptors referenced by pipe lowering."""
    if count <= 0:
        return []

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    return [
        ttnn.SemaphoreDescriptor(sem_id, core_ranges=core_ranges, initial_value=0)
        for sem_id in range(count)
    ]


def normalize_program_hash(program_hash: Optional[int]) -> Optional[int]:
    """Return a uint64 program-cache hash suitable for tt-metal."""
    if program_hash is None:
        return None
    return int(program_hash) & ((1 << 64) - 1)


def build_cb_descriptors(
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
    pipe_computed_address_backing_tensors: Optional[Dict[int, Any]] = None,
) -> List[Any]:
    """
    Build circular buffer descriptors for ttnn.generic_op.

    Args:
        tensors: List of ttnn.Tensor objects. Each tensor's position (0, 1, 2, ...)
            corresponds to its CB index. For intermediate CBs (not backed by
            input/output tensors), pass None in the corresponding position.
        cb_configs: List of DataflowBuffer objects for each DFB, indexed by DFB index.
            Each DFB has shape, block_count, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for DFB allocation.
        pipe_computed_address_backing_tensors: Hidden L1 backing tensors for DFBs whose
            receiver base is passed as a common runtime argument.

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    # Compute sizes first so we fail before allocating ttnn descriptors on overflow.
    rows = _get_cb_descriptor_rows(cb_configs)
    backing_tensors = pipe_computed_address_backing_tensors or {}

    remaining_bytes = DEFAULT_L1_CB_BUDGET_BYTES
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, "device"):
            device = tensor.device()
            if device is None:
                continue
            remaining_bytes = get_min_remaining_l1_for_device(device)
            break

    # Must stay aligned with MLIR ttl-validate-cb-budget (TileType::getSizeBytes)
    # and tile_bytes_from_dtype; see issue #511. Computed-address backing tensors
    # are allocated separately before this check, so their L1 is already
    # reflected in remaining_bytes; counting them here would double-charge them.
    counted_rows = [
        row for cb_index, row in enumerate(rows) if cb_index not in backing_tensors
    ]
    static_cb_bytes = sum(total_size for _, _, total_size, _ in counted_rows)
    if static_cb_bytes > remaining_bytes:
        breakdown = "\n".join(r[3] for r in counted_rows)
        raise ValueError(
            "Total circular buffer allocation ("
            f"{static_cb_bytes} bytes) exceeds L1 budget ({remaining_bytes} bytes). "
            "This checks static CB backing store only (not all L1 on core).\n"
            + breakdown
            + "\n  hint: reduce DFB shapes or block_count."
        )

    cb_descriptors = []
    for cb_index, (data_format, page_size, total_size, _) in enumerate(rows):
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=data_format,
            page_size=page_size,
        )
        cb_desc = ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )
        if cb_index in backing_tensors:
            backing_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb_index,
                backing_tensors[cb_index],
                total_size=total_size,
                core_ranges=core_ranges,
            )
            cb_desc.set_buffer_from_cb(backing_desc)
        cb_descriptors.append(cb_desc)

    return cb_descriptors


def build_generic_op_io_tensors(
    tensors: List[Any],
    pipe_sram_scratch_tensors: List[Any],
    pipe_computed_address_dfb_tensors: Optional[Dict[int, Any]] = None,
) -> List[Any]:
    """Return io_tensors with the user-visible output in the final position."""
    if not tensors:
        raise ValueError("kernel must have at least one output tensor")

    computed_address_dfb_tensors = [
        pipe_computed_address_dfb_tensors[dfb_index]
        for dfb_index in sorted(pipe_computed_address_dfb_tensors or {})
    ]
    io_tensors = (
        list(pipe_sram_scratch_tensors) + computed_address_dfb_tensors + list(tensors)
    )
    if len(io_tensors) < 2:
        io_tensors = [io_tensors[-1]] + io_tensors
    return io_tensors


def build_program_descriptor(
    kernel_descriptors: List[Any],
    cb_descriptors: List[Any],
    semaphore_descriptors: List[Any],
) -> Any:
    """Build the single-device descriptor used by current intra-chip execution."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    return ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=semaphore_descriptors,
    )


def _build_mesh_coordinate(coord: Any) -> Any:
    if isinstance(coord, (tuple, list)):
        try:
            return ttnn.MeshCoordinate(*coord)
        except TypeError:
            return ttnn.MeshCoordinate(coord)
    return coord


def _build_mesh_coordinate_range(placement: Any) -> Any:
    if isinstance(placement, MeshProgramPlacement):
        start = _build_mesh_coordinate(placement.start)
        end = _build_mesh_coordinate(
            placement.start if placement.end is None else placement.end
        )
        return ttnn.MeshCoordinateRange(start, end)
    if isinstance(placement, (tuple, list)):
        coord = _build_mesh_coordinate(placement)
        return ttnn.MeshCoordinateRange(coord, coord)
    return placement


def build_mesh_program_descriptor(
    program_descriptor: Any,
    mesh_program_placements: List[Any],
) -> Any:
    """Build a mesh descriptor that runs a program over selected device ranges."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    if not mesh_program_placements:
        raise ValueError("mesh_program_placements must not be empty")

    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    for placement in mesh_program_placements:
        mesh_range = _build_mesh_coordinate_range(placement)
        mesh_program_descriptor[mesh_range] = program_descriptor
    return mesh_program_descriptor


def _iter_device_domain_coordinates(device_domain):
    component_coordinates = []
    for component in device_domain.components:
        component_coordinates.append(
            tuple(itertools.product(*(range(extent) for extent in component.extent)))
        )
    for coordinates in itertools.product(*component_coordinates):
        runtime_coordinates = [
            value for coordinate in coordinates for value in coordinate
        ]
        yield tuple(runtime_coordinates), runtime_coordinates


def build_device_mesh_program_descriptor(
    program_descriptors: Dict[tuple, Any],
) -> Any:
    """Build a mesh descriptor containing one program per logical device."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    if not program_descriptors:
        raise ValueError("program_descriptors must not be empty")

    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    for mesh_coordinate, program_descriptor in program_descriptors.items():
        coordinate = _build_mesh_coordinate(mesh_coordinate)
        mesh_range = ttnn.MeshCoordinateRange(coordinate, coordinate)
        mesh_program_descriptor[mesh_range] = program_descriptor
    return mesh_program_descriptor


def configure_routing_plane_runtime_args(
    program_descriptor: Any,
    kernel_fabric_routes: List[List[FabricRouteSpec]],
    mesh_device: Any,
    device_coordinates: tuple,
    grid_cols: int,
    grid_rows: int,
    fabric_route_cache: Optional[_FabricRouteCache] = None,
) -> None:
    """Attach per-node routing-plane setup arguments to one device program."""
    if len(kernel_fabric_routes) != len(program_descriptor.kernels):
        raise ValueError(
            "kernel_fabric_routes must have one entry per kernel descriptor"
        )
    if not any(kernel_fabric_routes):
        return
    source_node_id = mesh_device.get_fabric_node_id(
        _build_mesh_coordinate(device_coordinates)
    )
    route_cache = (
        fabric_route_cache if fabric_route_cache is not None else _FabricRouteCache()
    )
    for kernel_index, routes in enumerate(kernel_fabric_routes):
        if not routes:
            continue

        kernel_descriptor = program_descriptor.kernels[kernel_index]
        for node_y in range(grid_rows):
            for node_x in range(grid_cols):
                node_coordinates = (node_x, node_y)
                active_remote_devices = []
                remote_index = {}
                route_remote_slots = [0] * len(routes)
                for route_index, route in enumerate(routes):
                    if route.local_device != device_coordinates:
                        continue
                    if node_coordinates not in route.source_nodes:
                        continue
                    if route.remote_device not in remote_index:
                        remote_index[route.remote_device] = len(active_remote_devices)
                        active_remote_devices.append(route.remote_device)
                    route_remote_slots[route_index] = remote_index[route.remote_device]

                destination_node_ids = [
                    mesh_device.get_fabric_node_id(_build_mesh_coordinate(coordinates))
                    for coordinates in active_remote_devices
                ]
                route_infos = [
                    route_cache.resolve(
                        mesh_device, source_node_id, destination_node_id
                    )
                    for destination_node_id in destination_node_ids
                ]
                connection_index_by_direction = {}
                connection_destination_node_ids = []
                connection_link_indices = []
                remote_connection_slots = []
                for route_info in route_infos:
                    connection_index = connection_index_by_direction.get(
                        route_info.direction
                    )
                    if connection_index is None:
                        connection_index = len(connection_destination_node_ids)
                        connection_index_by_direction[route_info.direction] = (
                            connection_index
                        )
                        connection_destination_node_ids.append(
                            route_info.connection_node_id
                        )
                        connection_link_indices.append(route_info.link_index)
                    elif (
                        connection_link_indices[connection_index]
                        != route_info.link_index
                    ):
                        raise ValueError(
                            "routes sharing one fabric direction must use one link"
                        )
                    elif (
                        connection_destination_node_ids[connection_index]
                        != route_info.connection_node_id
                    ):
                        raise ValueError(
                            "routes sharing one fabric direction must use one "
                            "connection node"
                        )
                    remote_connection_slots.append(connection_index)

                route_slots = [0] * len(routes)
                chip_routes = [0] * len(routes)
                for route_index, remote_slot in enumerate(route_remote_slots):
                    if remote_slot >= len(route_infos):
                        continue
                    if routes[route_index].local_device != device_coordinates:
                        continue
                    if node_coordinates not in routes[route_index].source_nodes:
                        continue
                    route_slots[route_index] = remote_connection_slots[remote_slot]
                    chip_routes[route_index] = int(route_infos[remote_slot].hop_count)
                runtime_prefix = [
                    len(connection_destination_node_ids),
                    *route_slots,
                    *chip_routes,
                ]
                worker_node = ttnn.CoreCoord(node_x, node_y)
                kernel_descriptor.runtime_args[node_x][node_y] = list(runtime_prefix)
                if not connection_destination_node_ids:
                    continue
                fabric_args = ttnn.setup_routing_plane_connection(
                    source_node_id,
                    connection_destination_node_ids,
                    connection_link_indices,
                    program_descriptor,
                    kernel_index,
                    worker_node,
                )
                kernel_descriptor.runtime_args[node_x][node_y].extend(fabric_args)
                if os.environ.get("TTLANG_DEBUG_FABRIC_ARGS"):
                    print(
                        "fabric runtime args:",
                        device_coordinates,
                        kernel_index,
                        (node_x, node_y),
                        connection_destination_node_ids,
                        kernel_descriptor.runtime_args[node_x][node_y],
                        flush=True,
                    )


def run_kernel_on_device(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
    program_hash: Optional[int] = None,
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    pipe_global_semaphore_lifetime: Optional[List[Any]] = None,
    mesh_program_placements: Optional[List[Any]] = None,
    device_domain: Optional[Any] = None,
    kernel_fabric_routes: Optional[List[List[FabricRouteSpec]]] = None,
    fabric_route_cache: Optional[_FabricRouteCache] = None,
) -> Any:
    """
    Execute kernels on device using ttnn.generic_op.

    This is the main entry point for kernel execution. It builds all
    descriptors and runs the program.

    Args:
        kernel_specs: List of kernel specifications (path, thread_type, tensor_indices, config).
        tensors: List of ttnn.Tensor objects. Position in this list determines the
            global tensor index. Individual kernels access subsets via tensor_indices
            in each KernelSpec.
        cb_configs: List of DataflowBuffer objects for each DFB, indexed by DFB index.
            Includes both tensor-backed DFBs and intermediate DFBs. Each DFB has shape,
            block_count, tensor (for dtype), and _cb_index attributes.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        program_hash: Hash for tt-metal program cache.
        num_pipe_sync_semaphores: Number of pipe synchronization semaphores
            allocated by the compiler.
        pipe_sram_scratch_bytes: Per-core SRAM scratch bytes required by
            PipeNet metadata.
        num_pipe_global_semaphores: Number of GlobalSemaphore-backed PipeNet
            synchronization counters allocated by the compiler.
        pipe_global_semaphore_lifetime: Optional list replaced with the current
            call's GlobalSemaphore objects. Cached kernels keep this bounded
            owner list so repeated calls do not retain old semaphore objects.
        mesh_program_placements: Optional mesh device ranges. When present,
            execution uses ttnn.MeshProgramDescriptor instead of
            ttnn.ProgramDescriptor.
        fabric_route_cache: Optional cache owned by a compiled kernel. Route
            results are reused while the mesh and fabric configuration remain
            unchanged.

    Returns:
        Result from ttnn.generic_op (typically None or output tensor).
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    # Build tensor accessor args.
    tensor_accessor_args = build_tensor_accessor_args(tensors)

    # Get grid dimensions from core_ranges.
    grid_size = core_ranges.bounding_box().grid_size()
    grid_cols = grid_size.x
    grid_rows = grid_size.y

    pipe_runtime_resources = build_pipe_runtime_resources(
        tensors=tensors,
        core_ranges=core_ranges,
        cb_configs=cb_configs,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
        pipe_computed_address_dfb_indices=sorted(
            {
                dfb_index
                for spec in kernel_specs
                for dfb_index in spec.pipe_computed_address_dfb_indices
            }
        ),
    )
    if pipe_global_semaphore_lifetime is not None:
        pipe_global_semaphore_lifetime[:] = pipe_runtime_resources.global_semaphores

    # Build kernel descriptors.
    kernel_descriptors = build_kernel_descriptors(
        kernel_specs=kernel_specs,
        tensors=tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_cbs=len(cb_configs),
        pipe_computed_address_base_addresses=(
            pipe_runtime_resources.computed_address_base_addresses
        ),
        extra_common_runtime_args=pipe_runtime_resources.extra_common_runtime_args,
        expected_extra_common_runtime_args=(
            pipe_runtime_resources.expected_extra_common_runtime_args
        ),
    )

    # Build CB descriptors.
    cb_descriptors = build_cb_descriptors(
        tensors=tensors,
        cb_configs=cb_configs,
        core_ranges=core_ranges,
        pipe_computed_address_backing_tensors=(
            pipe_runtime_resources.computed_address_dfb_tensors
        ),
    )

    semaphore_descriptors = build_pipe_sync_semaphore_descriptors(
        core_ranges=core_ranges,
        count=num_pipe_sync_semaphores,
    )

    # Build and execute program.
    program_descriptor = build_program_descriptor(
        kernel_descriptors=kernel_descriptors,
        cb_descriptors=cb_descriptors,
        semaphore_descriptors=semaphore_descriptors,
    )
    normalized_program_hash = normalize_program_hash(program_hash)
    if normalized_program_hash is not None:
        program_descriptor.custom_program_hash = normalized_program_hash
    program = program_descriptor
    if device_domain is not None:
        mesh_device = _first_device(tensors)
        fabric_routes = kernel_fabric_routes or [[] for _ in kernel_specs]
        program_descriptors = {}
        for mesh_coordinate, runtime_coordinates in _iter_device_domain_coordinates(
            device_domain
        ):
            device_kernel_descriptors = build_kernel_descriptors(
                kernel_specs=kernel_specs,
                tensors=tensors,
                tensor_accessor_args=tensor_accessor_args,
                core_ranges=core_ranges,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                num_cbs=len(cb_configs),
                pipe_computed_address_base_addresses=(
                    pipe_runtime_resources.computed_address_base_addresses
                ),
                extra_common_runtime_args=(
                    pipe_runtime_resources.extra_common_runtime_args
                ),
                expected_extra_common_runtime_args=(
                    pipe_runtime_resources.expected_extra_common_runtime_args
                ),
                device_coordinates=runtime_coordinates,
            )
            device_program = build_program_descriptor(
                kernel_descriptors=device_kernel_descriptors,
                cb_descriptors=cb_descriptors,
                semaphore_descriptors=semaphore_descriptors,
            )
            if normalized_program_hash is not None:
                device_program.custom_program_hash = normalized_program_hash
            configure_routing_plane_runtime_args(
                program_descriptor=device_program,
                kernel_fabric_routes=fabric_routes,
                mesh_device=mesh_device,
                device_coordinates=mesh_coordinate,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                fabric_route_cache=fabric_route_cache,
            )
            program_descriptors[mesh_coordinate] = device_program
        program = build_device_mesh_program_descriptor(program_descriptors)
    elif mesh_program_placements is not None:
        program = build_mesh_program_descriptor(
            program_descriptor=program_descriptor,
            mesh_program_placements=mesh_program_placements,
        )

    # ttnn.generic_op requires io_tensors to contain at least one input
    # and one output (size >= 2).  Output-only kernels (e.g. fill with no
    # input tensor) have only the output tensor; duplicate it so the runtime
    # sees [out, out].  The first copy acts as a dummy input that no kernel
    # thread actually reads.
    # TODO: Remove this workaround if ttnn.generic_op relaxes the >= 2
    # tensor requirement
    io_tensors = build_generic_op_io_tensors(
        tensors=tensors,
        pipe_sram_scratch_tensors=pipe_runtime_resources.scratch_tensors,
        pipe_computed_address_dfb_tensors=(
            pipe_runtime_resources.computed_address_dfb_tensors
        ),
    )

    return ttnn.generic_op(io_tensors, program)


def _dtype_to_ttnn_str(data_format) -> str:
    """Convert a data format to ttnn.dtype string for code emission."""
    dtype_str = str(data_format)
    if "bfloat16" in dtype_str.lower():
        return "ttnn.bfloat16"
    elif "float32" in dtype_str.lower():
        return "ttnn.float32"
    elif "float16" in dtype_str.lower():
        return "ttnn.float16"
    elif "uint32" in dtype_str.lower():
        return "ttnn.uint32"
    elif "uint16" in dtype_str.lower():
        return "ttnn.uint16"
    elif "int32" in dtype_str.lower():
        return "ttnn.int32"
    return "ttnn.bfloat16"


def _serialize_core_ranges(
    core_ranges: Optional[Any],
) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Serialize a ttnn.CoreRangeSet to nested ((sx, sy), (ex, ey)) tuples.

    Returns None when core_ranges is None (whole-grid fallback in the runner).
    """
    if core_ranges is None:
        return None
    serialized = []
    for core_range in core_ranges.ranges():
        serialized.append(
            (
                (int(core_range.start.x), int(core_range.start.y)),
                (int(core_range.end.x), int(core_range.end.y)),
            )
        )
    return serialized


def _serialize_noc_role(spec: KernelSpec) -> Optional[int]:
    """Map KernelSpec.config to the ttl.noc_index role for the emitted runner.

    Returns None for compute, 0 for reader, 1 for writer. The emitted file has
    no MLIR module, so the role already resolved from ttl.noc_index in
    _compile_ttnn_kernel is baked in here (same idea as KERNEL_CORE_RANGES).
    """
    if spec.thread_type == "compute":
        return None
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    if isinstance(spec.config, ttnn.ReaderConfigDescriptor):
        return 0
    if isinstance(spec.config, ttnn.WriterConfigDescriptor):
        return 1
    raise TypeError(
        f"Unsupported NOC config on kernel '{spec.path}': {type(spec.config)!r}"
    )


def _mesh_program_placement_to_source(placement: Any) -> str:
    if isinstance(placement, MeshProgramPlacement):
        return f"MeshProgramPlacement({placement.start!r}, {placement.end!r})"
    if isinstance(placement, (tuple, list)):
        return repr(tuple(placement))
    raise TypeError(
        "standalone runner mesh placements must be coordinate tuples "
        "or MeshProgramPlacement values"
    )


def emit_runner_source(
    kernel_specs: List[KernelSpec],
    cb_configs: List[Any],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    kernel_name: str = "kernel",
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    program_hash: Optional[int] = None,
    mesh_program_placements: Optional[List[Any]] = None,
) -> str:
    """
    Emit Python source code for a standalone runner that invokes ttnn.generic_op.

    Generates a ready-to-use Python file with all the CB and kernel
    descriptor setup. Tensor-specific values (buffer addresses, accessor args)
    are marked with TODO comments for the user to fill in.

    program_hash, if provided, is normalized to uint64 and embedded as the
    emitted runner's tt-metal program-cache key.
    mesh_program_placements, if provided, selects the device ranges that run
    the emitted program.
    """
    lines = []

    lines.append("# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC")
    lines.append("# SPDX-License-Identifier: Apache-2.0")
    lines.append("")
    lines.append(f'"""Auto-generated runner for {kernel_name}."""')
    lines.append("")
    lines.append("import ttnn")
    lines.append("")
    lines.append("from ttl.kernel_runner import (")
    lines.append("    KernelSpec,")
    lines.append("    MeshProgramPlacement,")
    lines.append("    build_cb_descriptors,")
    lines.append("    build_generic_op_io_tensors,")
    lines.append("    build_kernel_descriptors,")
    lines.append("    build_mesh_program_descriptor,")
    lines.append("    build_pipe_runtime_resources,")
    lines.append("    build_pipe_sync_semaphore_descriptors,")
    lines.append("    build_program_descriptor,")
    lines.append("    build_tensor_accessor_args,")
    lines.append(")")
    lines.append("")

    lines.append(f"GRID_COLS = {grid_cols}")
    lines.append(f"GRID_ROWS = {grid_rows}")
    lines.append(f"NUM_TENSORS = {num_tensors}")
    lines.append(f"PROGRAM_HASH = {normalize_program_hash(program_hash)!r}")
    lines.append(f"NUM_PIPE_SYNC_SEMAPHORES = {num_pipe_sync_semaphores}")
    lines.append(f"PIPE_SRAM_SCRATCH_BYTES = {pipe_sram_scratch_bytes}")
    lines.append(f"NUM_PIPE_GLOBAL_SEMAPHORES = {num_pipe_global_semaphores}")
    computed_address_dfb_indices = sorted(
        {
            dfb_index
            for spec in kernel_specs
            for dfb_index in spec.pipe_computed_address_dfb_indices
        }
    )
    lines.append(
        f"PIPE_COMPUTED_ADDRESS_DFB_INDICES = {computed_address_dfb_indices!r}"
    )
    if mesh_program_placements is None:
        lines.append("MESH_PROGRAM_PLACEMENTS = None")
    else:
        lines.append("MESH_PROGRAM_PLACEMENTS = [")
        for placement in mesh_program_placements:
            lines.append(f"    {_mesh_program_placement_to_source(placement)},")
        lines.append("]")
    lines.append("")

    lines.append("KERNEL_PATHS = [")
    for spec in kernel_specs:
        lines.append(f'    ("{spec.path}", "{spec.thread_type}"),')
    lines.append("]")
    lines.append("")

    lines.append("KERNEL_TENSOR_INDICES = [")
    for spec in kernel_specs:
        lines.append(f"    {spec.tensor_indices!r},  # {spec.thread_type}")
    lines.append("]")
    lines.append("")

    lines.append("KERNEL_PIPE_COMPUTED_ADDRESS_DFB_INDICES = [")
    for spec in kernel_specs:
        lines.append(
            f"    {spec.pipe_computed_address_dfb_indices!r},  # {spec.thread_type}"
        )
    lines.append("]")
    lines.append("")

    # Per-kernel dispatch ranges from KernelSpec.core_ranges. None means use
    # the whole-grid core_ranges below; a list of ((sx, sy), (ex, ey)) pairs
    # rebuilds a CoreRangeSet for specialize-cores clones.
    lines.append("KERNEL_CORE_RANGES = [")
    for spec in kernel_specs:
        lines.append(
            f"    {_serialize_core_ranges(spec.core_ranges)!r},  # {spec.thread_type}"
        )
    lines.append("]")
    lines.append("")

    # Per-kernel NOC roles from KernelSpec.config (set from ttl.noc_index in
    # _compile_ttnn_kernel). None = compute, 0 = reader, 1 = writer.
    lines.append("KERNEL_NOC_INDICES = [")
    for spec in kernel_specs:
        lines.append(f"    {_serialize_noc_role(spec)!r},  # {spec.thread_type}")
    lines.append("]")
    lines.append("")

    lines.append("KERNEL_EXTRA_COMMON_RUNTIME_ARGS = [")
    for spec in kernel_specs:
        extra_args = list(spec.extra_common_runtime_args or [])
        lines.append(f"    {extra_args!r},  # {spec.thread_type}")
    lines.append("]")
    lines.append("")
    lines.append("CB_CONFIGS = [")
    for i, cb in enumerate(cb_configs):
        if cb is None:
            lines.append(f"    None,  # CB {i}")
            continue
        data_format = _cb_data_format(cb)
        page_size = tile_bytes_from_dtype(data_format)
        dtype_str = _dtype_to_ttnn_str(data_format)
        num_tiles = cb.shape[0] * cb.shape[1] * cb.block_count
        total_size = num_tiles * page_size
        lines.append(
            f"    ({cb.shape!r}, {cb.block_count}, {dtype_str}, {page_size}, {total_size}),  # CB {i}"
        )
    lines.append("]")
    lines.append("")

    lines.append("")
    lines.append("def run(tensors, device=None):")
    lines.append(f'    """Run the {kernel_name} on device."""')
    lines.append(
        f"    assert len(tensors) == {num_tensors}, f'Expected {num_tensors} tensors, got {{len(tensors)}}'"
    )
    lines.append("")
    lines.append("    if device is None:")
    lines.append("        device = tensors[0].device()")
    lines.append("")

    lines.append("    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(")
    lines.append("        ttnn.CoreCoord(0, 0),")
    lines.append("        ttnn.CoreCoord(GRID_COLS - 1, GRID_ROWS - 1)")
    lines.append("    )])")
    lines.append("")

    lines.append("    tensor_accessor_args = build_tensor_accessor_args(tensors)")
    lines.append("    pipe_resources = build_pipe_runtime_resources(")
    lines.append("        tensors=tensors,")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        cb_configs=CB_CONFIGS,")
    lines.append("        pipe_sram_scratch_bytes=PIPE_SRAM_SCRATCH_BYTES,")
    lines.append("        num_pipe_global_semaphores=NUM_PIPE_GLOBAL_SEMAPHORES,")
    lines.append(
        "        pipe_computed_address_dfb_indices=PIPE_COMPUTED_ADDRESS_DFB_INDICES,"
    )
    lines.append("        device=device,")
    lines.append("    )")
    lines.append("")

    lines.append("    def _core_ranges_from_spec(ranges_spec):")
    lines.append("        if ranges_spec is None:")
    lines.append("            return None")
    lines.append("        return ttnn.CoreRangeSet([")
    lines.append("            ttnn.CoreRange(")
    lines.append("                ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey)")
    lines.append("            )")
    lines.append("            for (sx, sy), (ex, ey) in ranges_spec")
    lines.append("        ])")
    lines.append("")
    lines.append("    kernel_specs = []")
    lines.append("")
    lines.append(
        "    for kernel_idx, (kernel_path, thread_type) in enumerate(KERNEL_PATHS):"
    )
    lines.append("        noc_index = KERNEL_NOC_INDICES[kernel_idx]")
    lines.append("        if thread_type == 'compute' or noc_index is None:")
    lines.append("            config = ttnn.ComputeConfigDescriptor()")
    lines.append("        elif noc_index == 0:")
    lines.append("            config = ttnn.ReaderConfigDescriptor()")
    lines.append("        else:")
    lines.append("            config = ttnn.WriterConfigDescriptor()")
    lines.append("")
    lines.append("        kernel_specs.append(")
    lines.append("            KernelSpec(")
    lines.append("                path=kernel_path,")
    lines.append("                thread_type=thread_type,")
    lines.append("                tensor_indices=KERNEL_TENSOR_INDICES[kernel_idx],")
    lines.append("                config=config,")
    lines.append(
        "                pipe_computed_address_dfb_indices=KERNEL_PIPE_COMPUTED_ADDRESS_DFB_INDICES[kernel_idx],"
    )
    lines.append(
        "                core_ranges=_core_ranges_from_spec("
        "KERNEL_CORE_RANGES[kernel_idx]),"
    )
    lines.append(
        "                extra_common_runtime_args="
        "KERNEL_EXTRA_COMMON_RUNTIME_ARGS[kernel_idx],"
    )
    lines.append("            )")
    lines.append("        )")
    lines.append("    kernel_descriptors = build_kernel_descriptors(")
    lines.append("        kernel_specs=kernel_specs,")
    lines.append("        tensors=tensors,")
    lines.append("        tensor_accessor_args=tensor_accessor_args,")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        grid_cols=GRID_COLS,")
    lines.append("        grid_rows=GRID_ROWS,")
    lines.append("        num_cbs=len(CB_CONFIGS),")
    lines.append(
        "        pipe_computed_address_base_addresses=pipe_resources.computed_address_base_addresses,"
    )
    lines.append(
        "        extra_common_runtime_args=pipe_resources.extra_common_runtime_args,"
    )
    lines.append("        expected_extra_common_runtime_args=(")
    lines.append("            pipe_resources.expected_extra_common_runtime_args")
    lines.append("        ),")
    lines.append("    )")
    lines.append("")

    lines.append("    cb_descriptors = build_cb_descriptors(")
    lines.append("        tensors=tensors,")
    lines.append("        cb_configs=CB_CONFIGS,")
    lines.append("        core_ranges=core_ranges,")
    lines.append(
        "        pipe_computed_address_backing_tensors=pipe_resources.computed_address_dfb_tensors,"
    )
    lines.append("    )")
    lines.append("")

    lines.append("    semaphore_descriptors = build_pipe_sync_semaphore_descriptors(")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        count=NUM_PIPE_SYNC_SEMAPHORES,")
    lines.append("    )")
    lines.append("")

    lines.append("    program_descriptor = build_program_descriptor(")
    lines.append("        kernel_descriptors=kernel_descriptors,")
    lines.append("        cb_descriptors=cb_descriptors,")
    lines.append("        semaphore_descriptors=semaphore_descriptors,")
    lines.append("    )")
    lines.append("    if PROGRAM_HASH is not None:")
    lines.append("        program_descriptor.custom_program_hash = PROGRAM_HASH")
    lines.append("    program = program_descriptor")
    lines.append("    if MESH_PROGRAM_PLACEMENTS is not None:")
    lines.append("        program = build_mesh_program_descriptor(")
    lines.append("            program_descriptor=program_descriptor,")
    lines.append("            mesh_program_placements=MESH_PROGRAM_PLACEMENTS,")
    lines.append("        )")
    lines.append("")
    lines.append("    io_tensors = build_generic_op_io_tensors(")
    lines.append("        tensors=tensors,")
    lines.append("        pipe_sram_scratch_tensors=pipe_resources.scratch_tensors,")
    lines.append(
        "        pipe_computed_address_dfb_tensors=pipe_resources.computed_address_dfb_tensors,"
    )
    lines.append("    )")
    lines.append("    result = ttnn.generic_op(io_tensors, program)")
    lines.append("    return result")
    lines.append("")

    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append('    print("Runner generated. See run() function for usage.")')
    lines.append("")

    return "\n".join(lines)


def emit_runner_file(
    kernel_specs: List[KernelSpec],
    cb_configs: List[Any],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    output_path: str,
    kernel_name: str = "kernel",
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    program_hash: Optional[int] = None,
    mesh_program_placements: Optional[List[Any]] = None,
) -> str:
    """
    Emit a Python runner file for the compiled kernel.

    program_hash, if provided, is forwarded to the emitted runner as its
    normalized tt-metal program-cache key.
    mesh_program_placements, if provided, is forwarded as the emitted
    program's device ranges.

    Returns the output path.
    """
    import os

    source = emit_runner_source(
        kernel_specs=kernel_specs,
        cb_configs=cb_configs,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_tensors=num_tensors,
        program_hash=program_hash,
        kernel_name=kernel_name,
        num_pipe_sync_semaphores=num_pipe_sync_semaphores,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
        mesh_program_placements=mesh_program_placements,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(source)

    print(f"Runner written to {output_path}")
    return output_path


__all__ = [
    "KernelSpec",
    "FabricRouteSpec",
    "MeshProgramPlacement",
    "PipeRuntimeResources",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "build_pipe_sram_scratch_tensors",
    "build_pipe_global_semaphores",
    "build_pipe_computed_address_dfb_tensors",
    "build_pipe_runtime_resources",
    "build_pipe_sync_semaphore_descriptors",
    "normalize_program_hash",
    "build_generic_op_io_tensors",
    "build_device_mesh_program_descriptor",
    "configure_routing_plane_runtime_args",
    "build_mesh_program_descriptor",
    "build_program_descriptor",
    "run_kernel_on_device",
    "emit_runner_source",
    "emit_runner_file",
]
