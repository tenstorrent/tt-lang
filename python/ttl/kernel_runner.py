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


from .dataflow_buffer import (
    DFBStorageSegment,
    PhysicalDFBConfig,
    _validate_tensor_backed_dfb_range,
    _validate_tensor_backed_dfb_tensor,
)
from .constants import DEFAULT_L1_CB_BUDGET_BYTES
from .dtype_utils import format_name_to_ttnn_dtype
from ._fabric_target import (
    FabricRouteCache as _FabricRouteCache,
    FabricRouteSpec,
    configure_routing_plane_runtime_args as _configure_routing_plane_runtime_args,
)
from .kernel import KernelSelector


@dataclass(frozen=True)
class _DFBAllocation:
    """Derived tt-metal descriptor fields for one physical DFB."""

    data_format: Any
    num_tiles: int
    block_count: int
    tile: Optional[Tuple[int, int]]
    page_size: int
    total_size: int


def _validate_physical_dfb_config(
    config: PhysicalDFBConfig, physical_index: int
) -> None:
    """Enforce dense table order required by compile-time DFB indices."""
    if config.dfb_index != physical_index:
        raise ValueError(
            f"DFB config at physical index {physical_index} has dfb_index "
            f"{config.dfb_index}"
        )
    seen_nodes = set()
    for segment_position, segment in enumerate(config.storage_segments):
        if not segment.nodes:
            raise ValueError(
                f"DFB[{config.dfb_index}] storage segment {segment_position} "
                "has no launch nodes"
            )
        for node in segment.nodes:
            if node in seen_nodes:
                raise ValueError(
                    f"DFB[{config.dfb_index}] assigns launch node {node} to "
                    "multiple storage segments"
                )
            seen_nodes.add(node)


def _get_dfb_allocation(config: PhysicalDFBConfig) -> _DFBAllocation:
    """Derive the runtime layout and L1 size of one physical DFB."""
    if not isinstance(config, PhysicalDFBConfig):
        raise TypeError(
            "DFB runtime configuration must be a finalized PhysicalDFBConfig, "
            f"got {type(config).__name__}"
        )
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    data_format = format_name_to_ttnn_dtype(config.data_format)
    num_tiles = config.num_tiles
    block_count = config.block_count
    tile_shape = config.tile
    page_size = config.page_size
    return _DFBAllocation(
        data_format=data_format,
        num_tiles=num_tiles,
        block_count=block_count,
        tile=tile_shape,
        page_size=page_size,
        total_size=num_tiles * block_count * page_size,
    )


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
        compiler_include_paths: Additional -I paths for the JIT compiler.
        pipe_computed_address_dfb_indices: Receiver DFB indices whose backing
            addresses are passed to this kernel.
        core_ranges: Optional per-kernel ttnn.CoreRangeSet. When set, this
            specialized kernel binary is dispatched only to these cores. When None,
            the whole-grid core_ranges passed to build_kernel_descriptors is used.
        extra_common_runtime_args: Per-kernel runtime args appended after
            shared compiler-managed arguments.
        logical_kernel: Target-independent selector retained across kernel cloning.
    """

    path: str
    thread_type: str
    tensor_indices: List[int]
    config: Any
    compiler_include_paths: List[str] = field(default_factory=list)
    pipe_computed_address_dfb_indices: List[int] = field(default_factory=list)
    core_ranges: Optional[Any] = None
    extra_common_runtime_args: Optional[List[int]] = None
    logical_kernel: Optional[KernelSelector] = None


@dataclass
class PipeRuntimeResources:
    """Host allocations and runtime args for compiler-emitted pipe resources."""

    scratch_tensors: List[Any]
    global_semaphores: List[Any]
    computed_address_dfb_tensors: Dict[int, Any]
    computed_address_base_addresses: Dict[int, int]
    extra_common_runtime_args: List[int]
    expected_extra_common_runtime_args: int


@dataclass
class PipeGlobalSemaphoreCache:
    """Own stable compiler-managed GlobalSemaphores across cached dispatches."""

    _device: Optional[Any] = field(default=None, init=False)
    _core_coordinates: Optional[Tuple[Tuple[int, int], ...]] = field(
        default=None, init=False
    )
    _semaphores: List[Any] = field(default_factory=list, init=False)

    def acquire(
        self,
        tensors: List[Any],
        core_ranges: Any,
        count: int,
        device: Optional[Any] = None,
    ) -> List[Any]:
        """Return zeroed semaphores with stable addresses for one context."""
        if count <= 0:
            return []

        ttnn_api = _ensure_ttnn()
        if ttnn_api is None:
            raise RuntimeError("ttnn is not available")
        resource_device = device if device is not None else _first_device(tensors)
        core_coordinates = _core_range_coordinates(core_ranges)
        if (
            self._device is resource_device
            and self._core_coordinates == core_coordinates
            and len(self._semaphores) == count
        ):
            for semaphore in self._semaphores:
                ttnn_api.reset_global_semaphore_value(semaphore, 0)
            return self._semaphores

        self._semaphores, _addresses = build_pipe_global_semaphores(
            tensors=tensors,
            core_ranges=core_ranges,
            count=count,
            device=resource_device,
        )
        self._device = resource_device
        self._core_coordinates = core_coordinates
        return self._semaphores


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
            compiler_include_paths=spec.compiler_include_paths,
        )
        kernel_descriptors.append(kernel_desc)

    return kernel_descriptors


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _core_range_coordinates(core_ranges: Any) -> Tuple[Tuple[int, int], ...]:
    """Return the canonical worker coordinates for a semaphore allocation."""
    return tuple(
        sorted(
            (core_x, core_y)
            for core_range in core_ranges.ranges()
            for core_y in range(int(core_range.start.y), int(core_range.end.y) + 1)
            for core_x in range(int(core_range.start.x), int(core_range.end.x) + 1)
        )
    )


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
    """Allocate GlobalSemaphores used by compiler-managed PipeNet counters.

    A MeshDevice GlobalSemaphore has one common L1 address on the selected nodes
    of every device. Fabric atomics target the receiver device's instance at
    that address; node-local PipeNets use the same storage after local semaphore
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
    cb_configs: List[PhysicalDFBConfig],
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

    device = device if device is not None else _first_device(tensors)
    backing_tensors = {}
    for dfb_index in dfb_indices:
        if dfb_index < 0 or dfb_index >= len(cb_configs):
            raise ValueError(
                f"computed-address receiver DFB index {dfb_index} is invalid"
            )
        config = cb_configs[dfb_index]
        allocation = _get_dfb_allocation(config)
        _validate_physical_dfb_config(config, dfb_index)
        backing_core_ranges = _computed_address_storage_core_ranges(config, core_ranges)
        backing_tensors[dfb_index] = _allocate_l1_sharded_storage_tensor(
            backing_core_ranges, allocation.total_size, device
        )
    return backing_tensors


def build_pipe_runtime_resources(
    tensors: List[Any],
    core_ranges: Any,
    cb_configs: Optional[List[PhysicalDFBConfig]] = None,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    pipe_computed_address_dfb_indices: Optional[List[int]] = None,
    pipe_global_semaphore_cache: Optional[PipeGlobalSemaphoreCache] = None,
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
    if pipe_global_semaphore_cache is None:
        global_semaphores, global_semaphore_addresses = build_pipe_global_semaphores(
            tensors=tensors,
            core_ranges=core_ranges,
            count=num_pipe_global_semaphores,
            device=resource_device,
        )
    else:
        global_semaphores = pipe_global_semaphore_cache.acquire(
            tensors=tensors,
            core_ranges=core_ranges,
            count=num_pipe_global_semaphores,
            device=resource_device,
        )
        global_semaphore_addresses = [
            int(ttnn.get_global_semaphore_address(semaphore))
            for semaphore in global_semaphores
        ]
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


def _make_node_core_ranges(nodes: Tuple[Tuple[int, int], ...]) -> Any:
    """Build an exact CoreRangeSet without including unselected nodes."""
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(node_x, node_y),
                ttnn.CoreCoord(node_x, node_y),
            )
            for node_x, node_y in nodes
        ]
    )


def _computed_address_storage_core_ranges(
    config: PhysicalDFBConfig, core_ranges: Any
) -> Any:
    """Select nodes that use compiler-managed storage for a physical DFB."""
    if not config.storage_segments:
        return core_ranges

    storage_nodes = sorted(
        {
            node
            for segment in config.storage_segments
            if not segment.is_tensor_backed
            for node in segment.nodes
        }
    )
    if not storage_nodes:
        raise ValueError(
            f"computed-address DFB[{config.dfb_index}] has no "
            "compiler-managed storage segment"
        )
    return _make_node_core_ranges(tuple(storage_nodes))


def _validate_tensor_backed_dfb_binding(
    tensors: List[Any],
    config: PhysicalDFBConfig,
    segment: DFBStorageSegment,
) -> Any:
    """Validate one tensor binding and return its operation tensor."""
    tensor_index = segment.tensor_index
    tensor_count = len(tensors)
    if tensor_index is None or tensor_index < 0 or tensor_index >= tensor_count:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing index {tensor_index} "
            f"is outside [0, {tensor_count})"
        )
    tensor = tensors[tensor_index]
    if tensor is None:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing {tensor_index} is absent"
        )
    if config.data_format not in {"bfloat16", "bf16", "float32", "f32"}:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing format "
            f"{config.data_format} is not supported; expected BF16 or FP32"
        )
    context = f"DFB[{config.dfb_index}] tensor backing"
    properties = _validate_tensor_backed_dfb_tensor(tensor, context=context)
    expected_dtype = format_name_to_ttnn_dtype(config.data_format)
    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing dtype {tensor.dtype} "
            f"does not match {expected_dtype}"
        )
    if properties.tile_shape != config.tile:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing tile shape "
            f"{properties.tile_shape} does not match {config.tile}"
        )
    if properties.page_size != config.page_size:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing page size "
            f"{properties.page_size} does not match {config.page_size}"
        )
    try:
        tensor_nodes = {
            (int(core.x), int(core.y))
            for core in ttnn.get_optimal_worker_cores_for_sharded_tensor(tensor)
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"DFB[{config.dfb_index}] cannot determine the tensor's sharded "
            "L1 node set"
        ) from error
    missing_nodes = sorted(set(segment.nodes) - tensor_nodes)
    if missing_nodes:
        raise ValueError(
            f"DFB[{config.dfb_index}] tensor backing has no shard data on "
            f"launch nodes {missing_nodes}"
        )
    total_size = config.num_tiles * config.block_count * config.page_size
    _validate_tensor_backed_dfb_range(
        properties,
        byte_offset=segment.byte_offset,
        byte_size=total_size,
        context=context,
    )
    return tensor


def _validate_tensor_backing_aliases(
    tensors: List[Any], cb_configs: List[PhysicalDFBConfig]
) -> None:
    """Reject overlapping tensor storage not represented by one physical DFB."""
    bindings = []
    for config in cb_configs:
        for segment in config.storage_segments:
            if not segment.is_tensor_backed:
                continue
            tensor = _validate_tensor_backed_dfb_binding(tensors, config, segment)
            try:
                absolute_start = int(tensor.buffer_address()) + segment.byte_offset
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    f"DFB[{config.dfb_index}] tensor backing does not expose a "
                    "valid buffer_address()"
                ) from None
            absolute_end = (
                absolute_start
                + config.num_tiles * config.block_count * config.page_size
            )
            nodes = frozenset(segment.nodes)
            for (
                previous_index,
                previous_nodes,
                previous_start,
                previous_end,
            ) in bindings:
                if (
                    nodes.isdisjoint(previous_nodes)
                    or absolute_start >= previous_end
                    or previous_start >= absolute_end
                ):
                    continue
                if absolute_start != previous_start or absolute_end != previous_end:
                    raise ValueError(
                        "tensor-backed DFB byte ranges partially overlap on a "
                        "shared launch node"
                    )
                if config.dfb_index != previous_index:
                    raise ValueError(
                        "identical tensor-backed DFB ranges require one physical "
                        "DFB index on a shared launch node"
                    )
            bindings.append((config.dfb_index, nodes, absolute_start, absolute_end))


def _build_tensor_backed_dfb_descriptor(
    tensors: List[Any],
    config: PhysicalDFBConfig,
    allocation: _DFBAllocation,
    cb_format: Any,
    segment: DFBStorageSegment,
) -> Any:
    tensor_index = segment.tensor_index
    assert tensor_index is not None
    tensor = tensors[tensor_index]
    segment_core_ranges = _make_node_core_ranges(segment.nodes)
    descriptor = ttnn.cb_descriptor_from_sharded_tensor(
        config.dfb_index,
        tensor,
        address_offset=segment.byte_offset,
        total_size=allocation.total_size,
        core_ranges=segment_core_ranges,
    )
    native_tile = tuple(tensor.get_tile().tile_shape)
    if native_tile != config.tile:
        # A physical view retains the tensor-owned address but needs the DFB
        # page interpretation selected by the source program.
        descriptor.format_descriptors = [cb_format]
    return descriptor


def build_cb_descriptors(
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    pipe_computed_address_backing_tensors: Optional[Dict[int, Any]] = None,
) -> List[Any]:
    """
    Build circular buffer descriptors for ttnn.generic_op.

    Args:
        tensors: Positional operation tensors. Tensor-backed storage segments
            refer to entries in this list by tensor index.
        cb_configs: Finalized runtime configurations indexed by physical DFB
            index.
        core_ranges: ttnn.CoreRangeSet for DFB allocation.
        pipe_computed_address_backing_tensors: Hidden L1 backing tensors for DFBs whose
            receiver base is passed as a common runtime argument.

    Returns:
        List of ttnn.CBDescriptor objects. A configuration with storage
        segments produces one descriptor per segment.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    backing_tensors = pipe_computed_address_backing_tensors or {}
    invalid_backing_indices = sorted(
        set(backing_tensors).difference(range(len(cb_configs)))
    )
    if invalid_backing_indices:
        raise ValueError(
            "computed-address backing tensors reference invalid DFB indices "
            f"{invalid_backing_indices}"
        )

    _validate_tensor_backing_aliases(tensors, cb_configs)

    # Compute sizes first so overflow is diagnosed before creating descriptors.
    allocations = []
    static_cb_bytes = 0
    static_allocation_summaries = []
    for physical_index, config in enumerate(cb_configs):
        allocation = _get_dfb_allocation(config)
        _validate_physical_dfb_config(config, physical_index)
        allocation_summary = (
            f"  DFB[{physical_index}]: num_tiles={allocation.num_tiles} "
            f"block_count={allocation.block_count} "
            f"format={config.data_format} tile={allocation.tile} -> "
            f"{allocation.total_size} bytes"
        )
        allocations.append(allocation)
        has_static_storage = not config.storage_segments or any(
            not segment.is_tensor_backed for segment in config.storage_segments
        )
        if physical_index not in backing_tensors and has_static_storage:
            static_cb_bytes += allocation.total_size
            static_allocation_summaries.append(allocation_summary)

    remaining_bytes = DEFAULT_L1_CB_BUDGET_BYTES
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, "device"):
            device = tensor.device()
            if device is None:
                continue
            remaining_bytes = get_min_remaining_l1_for_device(device)
            break

    # Must stay aligned with MLIR ttl-validate-cb-budget and the finalized DFB
    # page-size metadata. Computed-address backing tensors are allocated
    # separately before this check, so their L1 is already reflected in
    # remaining_bytes; counting them here would double-charge them.
    if static_cb_bytes > remaining_bytes:
        breakdown = "\n".join(static_allocation_summaries)
        raise ValueError(
            "Total circular buffer allocation ("
            f"{static_cb_bytes} bytes) exceeds L1 budget ({remaining_bytes} bytes). "
            "This checks static CB backing store only (not all L1 on core).\n"
            + breakdown
            + "\n  hint: reduce DFB shapes or block_count."
        )

    cb_descriptors = []
    for cb_index, allocation in enumerate(allocations):
        config = cb_configs[cb_index]
        tile_descriptor = (
            ttnn.TileDescriptor(ttnn.Tile(allocation.tile))
            if allocation.tile is not None
            else None
        )
        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=allocation.data_format,
            page_size=allocation.page_size,
            **({"tile": tile_descriptor} if tile_descriptor is not None else {}),
        )
        if cb_index in backing_tensors:
            backing_core_ranges = _computed_address_storage_core_ranges(
                config, core_ranges
            )
            cb_desc = ttnn.CBDescriptor(
                total_size=allocation.total_size,
                core_ranges=backing_core_ranges,
                format_descriptors=[cb_format],
            )
            backing_desc = ttnn.cb_descriptor_from_sharded_tensor(
                cb_index,
                backing_tensors[cb_index],
                total_size=allocation.total_size,
                core_ranges=backing_core_ranges,
            )
            cb_desc.set_buffer_from_cb(backing_desc)
            cb_descriptors.append(cb_desc)

            for segment in config.storage_segments:
                if not segment.is_tensor_backed:
                    continue
                cb_descriptors.append(
                    _build_tensor_backed_dfb_descriptor(
                        tensors, config, allocation, cb_format, segment
                    )
                )
            continue

        if not config.storage_segments:
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=allocation.total_size,
                    core_ranges=core_ranges,
                    format_descriptors=[cb_format],
                )
            )
            continue

        for segment in config.storage_segments:
            segment_core_ranges = _make_node_core_ranges(segment.nodes)
            if not segment.is_tensor_backed:
                cb_descriptors.append(
                    ttnn.CBDescriptor(
                        total_size=allocation.total_size,
                        core_ranges=segment_core_ranges,
                        format_descriptors=[cb_format],
                    )
                )
                continue

            cb_descriptors.append(
                _build_tensor_backed_dfb_descriptor(
                    tensors, config, allocation, cb_format, segment
                )
            )

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
    """Attach validated routing-plane target bindings to one device program."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    _configure_routing_plane_runtime_args(
        ttnn_api=ttnn,
        program_descriptor=program_descriptor,
        kernel_fabric_routes=kernel_fabric_routes,
        mesh_device=mesh_device,
        device_coordinates=device_coordinates,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        route_cache=fabric_route_cache,
    )


def run_kernel_on_device(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    program_hash: Optional[int] = None,
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    pipe_global_semaphore_cache: Optional[PipeGlobalSemaphoreCache] = None,
    mesh_program_placements: Optional[List[Any]] = None,
    device_domain: Optional[Any] = None,
    kernel_fabric_routes: Optional[List[List[FabricRouteSpec]]] = None,
    fabric_route_cache: Optional[_FabricRouteCache] = None,
    device: Optional[Any] = None,
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
        cb_configs: Finalized physical DFB configurations, in physical-index
            order.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        program_hash: Hash for tt-metal program cache.
        num_pipe_sync_semaphores: Number of pipe synchronization semaphores
            allocated by the compiler.
        pipe_sram_scratch_bytes: Per-core SRAM scratch bytes required by
            PipeNet metadata.
        num_pipe_global_semaphores: Number of GlobalSemaphore-backed PipeNet
            counters allocated by the compiler.
        pipe_global_semaphore_cache: Optional cache that owns stable
            GlobalSemaphore allocations across repeated program-cache hits and
            resets their values before reuse.
        mesh_program_placements: Optional mesh device ranges. When present,
            execution uses ttnn.MeshProgramDescriptor instead of
            ttnn.ProgramDescriptor.
        fabric_route_cache: Optional cache owned by a compiled kernel.
            Direction results are reused while the mesh and fabric
            configuration remain unchanged.
        device: Optional device used for hidden runtime allocations and fabric
            binding. Tensor-backed calls infer it from the first device tensor.

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
        pipe_global_semaphore_cache=pipe_global_semaphore_cache,
        device=device,
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

    normalized_program_hash = normalize_program_hash(program_hash)

    def build_device_program(device_coordinates=None):
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
            extra_common_runtime_args=(
                pipe_runtime_resources.extra_common_runtime_args
            ),
            expected_extra_common_runtime_args=(
                pipe_runtime_resources.expected_extra_common_runtime_args
            ),
            device_coordinates=device_coordinates,
        )
        program_descriptor = build_program_descriptor(
            kernel_descriptors=kernel_descriptors,
            cb_descriptors=cb_descriptors,
            semaphore_descriptors=semaphore_descriptors,
        )
        if normalized_program_hash is not None:
            program_descriptor.custom_program_hash = normalized_program_hash
        return program_descriptor

    if device_domain is not None:
        mesh_device = device if device is not None else _first_device(tensors)
        fabric_routes = kernel_fabric_routes or [[] for _ in kernel_specs]
        program_descriptors = {}
        for mesh_coordinate, runtime_coordinates in _iter_device_domain_coordinates(
            device_domain
        ):
            device_program = build_device_program(runtime_coordinates)
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
    else:
        program_descriptor = build_device_program()
        program = program_descriptor
        if mesh_program_placements is not None:
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


def _device_domain_to_source(device_domain: Optional[Any]) -> str:
    if device_domain is None:
        return "None"
    components = {
        component.name: tuple(component.extent)
        for component in device_domain.components
    }
    if len(components) == 1:
        name, extent = next(iter(components.items()))
        return f"DeviceDomain({extent!r}, name={name!r})"
    return f"DeviceDomain.product(**{components!r})"


def _fabric_routes_to_source(
    kernel_fabric_routes: Optional[List[List[FabricRouteSpec]]],
) -> str:
    if kernel_fabric_routes is None:
        return "None"
    kernel_routes = []
    for routes in kernel_fabric_routes:
        route_sources = [
            "FabricRouteSpec("
            f"{route.local_device!r}, {route.remote_device!r}, "
            f"{route.source_nodes!r}, {route.route_index!r})"
            for route in routes
        ]
        kernel_routes.append("[" + ", ".join(route_sources) + "]")
    return "[" + ", ".join(kernel_routes) + "]"


def emit_runner_source(
    kernel_specs: List[KernelSpec],
    cb_configs: List[PhysicalDFBConfig],
    grid_cols: int,
    grid_rows: int,
    num_tensors: int,
    kernel_name: str = "kernel",
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    program_hash: Optional[int] = None,
    mesh_program_placements: Optional[List[Any]] = None,
    device_domain: Optional[Any] = None,
    kernel_fabric_routes: Optional[List[List[FabricRouteSpec]]] = None,
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
    lines.append("from ttl.dataflow_buffer import DFBStorageSegment")
    lines.append("from ttl.dataflow_buffer import PhysicalDFBConfig")
    lines.append("from ttl.domains import DeviceDomain")
    lines.append("from ttl.kernel_runner import (")
    lines.append("    FabricRouteSpec,")
    lines.append("    KernelSpec,")
    lines.append("    MeshProgramPlacement,")
    lines.append("    PipeGlobalSemaphoreCache,")
    lines.append("    run_kernel_on_device,")
    lines.append(")")
    lines.append("")

    lines.append(f"GRID_COLS = {grid_cols}")
    lines.append(f"GRID_ROWS = {grid_rows}")
    lines.append(f"NUM_TENSORS = {num_tensors}")
    lines.append(f"PROGRAM_HASH = {normalize_program_hash(program_hash)!r}")
    lines.append(f"NUM_PIPE_SYNC_SEMAPHORES = {num_pipe_sync_semaphores}")
    lines.append(f"PIPE_SRAM_SCRATCH_BYTES = {pipe_sram_scratch_bytes}")
    lines.append(f"NUM_PIPE_GLOBAL_SEMAPHORES = {num_pipe_global_semaphores}")
    lines.append("PIPE_GLOBAL_SEMAPHORE_CACHE = PipeGlobalSemaphoreCache()")
    if mesh_program_placements is None:
        lines.append("MESH_PROGRAM_PLACEMENTS = None")
    else:
        lines.append("MESH_PROGRAM_PLACEMENTS = [")
        for placement in mesh_program_placements:
            lines.append(f"    {_mesh_program_placement_to_source(placement)},")
        lines.append("]")
    lines.append(f"DEVICE_DOMAIN = {_device_domain_to_source(device_domain)}")
    lines.append(
        "KERNEL_FABRIC_ROUTES = " f"{_fabric_routes_to_source(kernel_fabric_routes)}"
    )
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
    for physical_index, config in enumerate(cb_configs):
        _get_dfb_allocation(config)
        _validate_physical_dfb_config(config, physical_index)
        lines.append("    PhysicalDFBConfig(")
        lines.append(f"        dfb_index={config.dfb_index},")
        lines.append(f"        num_tiles={config.num_tiles},")
        lines.append(f"        data_format={config.data_format!r},")
        lines.append(f"        block_count={config.block_count},")
        lines.append(f"        page_size={config.page_size},")
        lines.append(f"        tile={config.tile!r},")
        if config.storage_segments:
            lines.append("        storage_segments=(")
            for segment in config.storage_segments:
                lines.append("            DFBStorageSegment(")
                lines.append(f"                nodes={segment.nodes!r},")
                lines.append(f"                tensor_index={segment.tensor_index!r},")
                lines.append(f"                byte_offset={segment.byte_offset},")
                lines.append(f"                byte_size={segment.byte_size!r},")
                lines.append("            ),")
            lines.append("        ),")
        lines.append(f"    ),  # DFB {physical_index}")
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
    lines.append("    return run_kernel_on_device(")
    lines.append("        kernel_specs=kernel_specs,")
    lines.append("        tensors=tensors,")
    lines.append("        cb_configs=CB_CONFIGS,")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        program_hash=PROGRAM_HASH,")
    lines.append("        num_pipe_sync_semaphores=NUM_PIPE_SYNC_SEMAPHORES,")
    lines.append("        pipe_sram_scratch_bytes=PIPE_SRAM_SCRATCH_BYTES,")
    lines.append("        num_pipe_global_semaphores=NUM_PIPE_GLOBAL_SEMAPHORES,")
    lines.append("        pipe_global_semaphore_cache=PIPE_GLOBAL_SEMAPHORE_CACHE,")
    lines.append("        mesh_program_placements=MESH_PROGRAM_PLACEMENTS,")
    lines.append("        device_domain=DEVICE_DOMAIN,")
    lines.append("        kernel_fabric_routes=KERNEL_FABRIC_ROUTES,")
    lines.append("        device=device,")
    lines.append("    )")
    lines.append("")

    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append('    print("Runner generated. See run() function for usage.")')
    lines.append("")

    return "\n".join(lines)


def emit_runner_file(
    kernel_specs: List[KernelSpec],
    cb_configs: List[PhysicalDFBConfig],
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
    device_domain: Optional[Any] = None,
    kernel_fabric_routes: Optional[List[List[FabricRouteSpec]]] = None,
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
        device_domain=device_domain,
        kernel_fabric_routes=kernel_fabric_routes,
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
    "PipeGlobalSemaphoreCache",
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
