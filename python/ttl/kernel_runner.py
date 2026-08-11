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
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    TTNN reports ``cb_limit`` and L1 ``page_address`` as absolute addresses,
    not byte counts. Static CBs grow upward from
    ``address_at_first_l1_cb_buffer``, while tensor pages occupy the region
    above them. The safe budget is therefore the address interval from the CB
    base to the lowest allocated L1 page, or to ``cb_limit`` when no L1 tensor
    exists. Using page placement also accounts for allocator alignment and
    fragmentation instead of assuming packed tensor pages.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    info = ttnn._ttnn.reports.get_device_info(device)
    cb_base = int(info.address_at_first_l1_cb_buffer)
    first_occupied = int(info.cb_limit)
    for page in ttnn._ttnn.reports.get_buffer_pages(device):
        if page.buffer_type == ttnn.BufferType.L1:
            first_occupied = min(first_occupied, int(page.page_address))

    return max(0, first_occupied - cb_base)


def get_remaining_l1_by_core_for_device(device, core_coordinates):
    """Return the minimum remaining CB budget for each logical worker core.

    Buffer-page reports include logical core coordinates and device ids.  A
    mesh may place a different lowest L1 tensor page on the same logical core
    of different devices, so retain the minimum address across devices for
    each coordinate without collapsing unrelated cores together.
    """

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    info = ttnn._ttnn.reports.get_device_info(device)
    cb_base = int(info.address_at_first_l1_cb_buffer)
    cb_limit = int(info.cb_limit)
    first_occupied = {
        tuple(core): cb_limit for core in core_coordinates
    }
    for page in ttnn._ttnn.reports.get_buffer_pages(device):
        if page.buffer_type != ttnn.BufferType.L1:
            continue
        core = (int(page.core_x), int(page.core_y))
        if core in first_occupied:
            first_occupied[core] = min(
                first_occupied[core], int(page.page_address)
            )

    return {
        core: max(0, address - cb_base)
        for core, address in first_occupied.items()
    }


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
        core_ranges: Optional per-kernel ttnn.CoreRangeSet. When set, this
            specialized kernel binary is dispatched only to these cores. When None,
            the whole-grid core_ranges passed to build_kernel_descriptors is used.
        used_cb_indices: Physical CB slots referenced by the final kernel body.
            None means metadata is unavailable and conservatively uses every CB;
            an empty list means this kernel uses no CBs.
        compiler_include_paths: Additional -I paths for the JIT compiler.
    """

    path: str
    thread_type: str
    tensor_indices: List[int]
    config: Any
    core_ranges: Optional[Any] = None
    used_cb_indices: Optional[List[int]] = None
    compiler_include_paths: List[str] = field(default_factory=list)


@dataclass
class PipeRuntimeResources:
    """Host allocations and runtime args for compiler-emitted pipe resources."""

    scratch_tensors: List[Any]
    global_semaphores: List[Any]
    extra_common_runtime_args: List[int]
    expected_extra_common_runtime_args: int


@dataclass
class ProgramRuntimeResources:
    """Optional host resources supplied by an ``@ttl.operation`` factory.

    This is the narrow escape hatch used by opaque kernels whose TT-Metal ABI
    needs host-created resources that the TT-Lang dialect does not model yet
    (currently fabric connection arguments and program semaphores). Runtime
    arguments are keyed by unified thread name and use TTNN's normal
    ``[(CoreCoord, [args...]), ...]`` representation.
    """

    semaphore_descriptors: List[Any] = field(default_factory=list)
    runtime_args_by_thread: Dict[str, List[Tuple[Any, List[int]]]] = field(
        default_factory=dict
    )
    defines_by_thread: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=dict
    )
    lifetimes: List[Any] = field(default_factory=list)
    # Optional exact replacement for the whole-grid descriptors normally
    # synthesized from ``cb_configs``.  This is deliberately a runtime-resource
    # escape hatch: opaque fused programs can describe one CB id with multiple
    # disjoint per-core descriptors while the compiler's static CB allocation
    # model remains conservative.  ``run_kernel_on_device`` validates the
    # replacement before it can bypass the default descriptor builder.
    cb_descriptors_override: Optional[List[Any]] = None
    # Optional capacity-only specialization for selected CB ids.  Each value
    # is ``[(CoreRangeSet, num_pages), ...]`` and must partition the program
    # grid.  Formats still come from the compiler-derived ``cb_configs``;
    # unlisted ids retain their normal whole-grid descriptors.
    cb_pages_by_core: Dict[int, List[Tuple[Any, int]]] = field(
        default_factory=dict
    )


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
    extra_common_runtime_args: Optional[List[int]] = None,
    expected_extra_common_runtime_args: Optional[int] = None,
    runtime_args_by_thread: Optional[Dict[str, List[Tuple[Any, List[int]]]]] = None,
    defines_by_thread: Optional[Dict[str, List[Tuple[str, str]]]] = None,
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
        extra_common_runtime_args: Compiler-managed common runtime args appended
            after tensor buffer addresses.
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

    runtime_args_by_thread = runtime_args_by_thread or {}
    defines_by_thread = defines_by_thread or {}

    for spec in kernel_specs:
        # Build common_runtime_args using tensor_indices.
        # C++ indexes by function-local position, we provide addresses in that order.
        common_runtime_args = [
            tensors[idx].buffer_address() for idx in spec.tensor_indices
        ]
        common_runtime_args.extend(extra_args)

        # Compute kernels only need CB indices.
        # DM kernels need CB indices + TensorAccessorArgs config.
        if spec.thread_type == "compute":
            kernel_compile_time_args = cb_indices
        else:
            kernel_compile_time_args = cb_indices + list(tensor_accessor_args)

        # Prefer per-kernel core_ranges (specialize-cores clones); otherwise
        # fall back to the whole-grid core_ranges.
        kernel_ranges = (
            spec.core_ranges if spec.core_ranges is not None else core_ranges
        )

        if spec.thread_type == "compute":
            thread_name = "trisc"
        elif isinstance(spec.config, ttnn.ReaderConfigDescriptor):
            thread_name = "ncrisc"
        else:
            thread_name = "brisc"

        thread_runtime_args = runtime_args_by_thread.get(thread_name, [])
        if spec.core_ranges is not None:
            # A specialized descriptor may cover only one core or one logical
            # row. TTNN requires every per-core runtime-argument entry to fall
            # inside that descriptor's CoreRangeSet.
            thread_runtime_args = [
                entry
                for entry in thread_runtime_args
                if kernel_ranges.contains(entry[0])
            ]

        kernel_desc = ttnn.KernelDescriptor(
            kernel_source=spec.path,
            core_ranges=kernel_ranges,
            compile_time_args=kernel_compile_time_args,
            defines=defines_by_thread.get(thread_name, []),
            runtime_args=thread_runtime_args,
            common_runtime_args=common_runtime_args,
            config=spec.config,
            compiler_include_paths=spec.compiler_include_paths,
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

    aligned_bytes = _align_up(scratch_bytes, 32)
    elements_per_core = max(1, aligned_bytes // 4)
    grid_size = core_ranges.bounding_box().grid_size()
    num_cores = grid_size.x * grid_size.y
    device = device if device is not None else _first_device(tensors)
    # [Device 2.0] This encodes compiler SRAM as a sharded TTNN tensor because
    # current generic_op has no typed device-side scratch allocation object.
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
    scratch_tensor = ttnn.empty(
        (num_cores, elements_per_core),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return [scratch_tensor]


def build_pipe_global_semaphores(
    tensors: List[Any],
    core_ranges: Any,
    count: int,
    device: Optional[Any] = None,
) -> Tuple[List[Any], List[int]]:
    """Allocate GlobalSemaphores used by PipeNet ready counters.

    PipeNet coordinates are per-device core coordinates. When tensors live on
    a TTNN MeshDevice, the same intra-chip PipeNet program is replicated across
    device shards; this allocates one MeshDevice GlobalSemaphore object whose
    address is passed to that replicated program. It does not create an
    inter-chip PipeNet or assign per-mesh-coordinate pipe synchronization state.
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


def build_pipe_runtime_resources(
    tensors: List[Any],
    core_ranges: Any,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    device: Optional[Any] = None,
) -> PipeRuntimeResources:
    """Allocate pipe resources and build their appended common runtime args."""
    resource_device = device
    if resource_device is None and (
        pipe_sram_scratch_bytes > 0 or num_pipe_global_semaphores > 0
    ):
        resource_device = _first_device(tensors)

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
    # then GlobalSemaphore ready-counter addresses.
    # [Device 2.0] This is the current ABI for pipe resource records; future
    # typed resource handles should preserve the same compiler-selected order.
    extra_common_runtime_args = [tensor.buffer_address() for tensor in scratch_tensors]
    extra_common_runtime_args.extend(global_semaphore_addresses)
    expected_extra_common_runtime_args = (
        len(scratch_tensors) + num_pipe_global_semaphores
    )
    return PipeRuntimeResources(
        scratch_tensors=scratch_tensors,
        global_semaphores=global_semaphores,
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


@dataclass(frozen=True)
class CBGeometry:
    """Physical layout of one CB slot.

    Single source of truth for CB sizing: both the ttnn descriptors handed to
    generic_op and the debug CB table are derived from these fields, so the
    table cannot disagree with what the device is configured with.
    """

    data_format: Any  # ttnn.DataType
    page_size: int  # bytes per page (one tile)
    num_pages: int
    total_size: int  # bytes
    tile_descriptor: Any  # ttnn.TileDescriptor, None when compiler-allocated
    tile: Optional[Tuple[int, int]]  # None when compiler-allocated
    shape: Optional[Tuple[int, ...]]  # None when compiler-allocated
    block_count: int
    breakdown: str  # one-line human summary, used in the L1 budget error


def cb_geometry(index: int, cb: Any) -> CBGeometry:
    """Resolve one DFB config to the physical CB layout it will be given."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    if cb is None:
        raise ValueError(
            f"Missing CB config for index {index}. "
            f"All DFB indices must have associated DataflowBuffer configurations."
        )

    if isinstance(cb, CompilerAllocatedDFBConfig):
        data_format = format_name_to_ttnn_dtype(cb.data_format)
        page_size = tile_bytes_from_dtype(data_format)
        num_pages = cb.num_tiles * cb.block_count
        total_size = num_pages * page_size
        return CBGeometry(
            data_format=data_format,
            page_size=page_size,
            num_pages=num_pages,
            total_size=total_size,
            tile_descriptor=None,
            tile=None,
            shape=None,
            block_count=cb.block_count,
            breakdown=(
                f"  CB[{index}]: compiler-allocated num_tiles={cb.num_tiles} "
                f"block_count={cb.block_count} format={cb.data_format} -> {total_size} bytes"
            ),
        )

    data_format = _cb_data_format(cb)
    tile = ttnn.Tile(cb.tile)
    page_size = tile.get_tile_size(data_format)
    num_pages = cb.shape[0] * cb.shape[1] * cb.block_count
    total_size = num_pages * page_size
    return CBGeometry(
        data_format=data_format,
        page_size=page_size,
        num_pages=num_pages,
        total_size=total_size,
        tile_descriptor=ttnn.TileDescriptor(tile),
        tile=tuple(cb.tile),
        shape=tuple(cb.shape),
        block_count=cb.block_count,
        breakdown=(
            f"  CB[{index}]: shape={cb.shape} block_count={cb.block_count} "
            f"-> {total_size} bytes"
        ),
    )


def _core_range_coordinates(core_ranges: Any, *, label: str) -> set[Tuple[int, int]]:
    """Expand a CoreRangeSet-like object into logical ``(x, y)`` pairs."""
    if core_ranges is None or not hasattr(core_ranges, "ranges"):
        raise ValueError(f"{label} must be a CoreRangeSet with ranges()")

    coordinates = set()
    for core_range in core_ranges.ranges():
        try:
            start_x = int(core_range.start.x)
            start_y = int(core_range.start.y)
            end_x = int(core_range.end.x)
            end_y = int(core_range.end.y)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid core range") from exc
        if end_x < start_x or end_y < start_y:
            raise ValueError(f"{label} contains an inverted core range")
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                coordinates.add((x, y))
    if not coordinates:
        raise ValueError(f"{label} must cover at least one core")
    return coordinates


def _tile_descriptor_key(tile: Any) -> Optional[Tuple[int, int]]:
    """Return a stable tile-shape key for a TTNN TileDescriptor-like object."""
    if tile is None:
        return None
    height = getattr(tile, "height", None)
    width = getattr(tile, "width", None)
    if callable(height):
        height = height()
    if callable(width):
        width = width()
    if height is None or width is None:
        shape = getattr(tile, "tile_shape", None)
        if callable(shape):
            shape = shape()
        if shape is None or len(shape) != 2:
            raise ValueError("CB format tile must expose height/width or tile_shape")
        height, width = shape
    return int(height), int(width)


def _remaining_cb_budget(tensors: List[Any]) -> int:
    """Return the same device-aware CB budget used by the default builder."""
    remaining_bytes = DEFAULT_L1_CB_BUDGET_BYTES
    for tensor in tensors:
        if tensor is None or not hasattr(tensor, "device"):
            continue
        device = tensor.device()
        if device is not None:
            return get_min_remaining_l1_for_device(device)
    return remaining_bytes


def _remaining_cb_budgets_by_core(
    tensors: List[Any], core_coordinates: set[Tuple[int, int]]
) -> Dict[Tuple[int, int], int]:
    """Return device-aware CB budgets without conflating unrelated cores."""

    for tensor in tensors:
        if tensor is None or not hasattr(tensor, "device"):
            continue
        device = tensor.device()
        if device is not None:
            return get_remaining_l1_by_core_for_device(
                device, core_coordinates
            )
    return {
        core: DEFAULT_L1_CB_BUDGET_BYTES for core in core_coordinates
    }


def validate_cb_descriptors_override(
    descriptors: List[Any],
    program_core_ranges: Any,
    tensors: List[Any],
    num_cbs: int,
    required_cb_ids: Optional[set[int]] = None,
) -> List[Any]:
    """Validate an exact, possibly per-core CB descriptor replacement.

    The same numeric CB id may occur in more than one descriptor only when
    those descriptors cover disjoint cores and use one identical page format.
    Their capacities may differ.  L1 usage is checked as the maximum sum on
    any one core, rather than summing mutually exclusive descriptors globally.

    This intentionally supports the static subset of Blaze's descriptor
    model.  It does not support runtime phase reconfiguration, overlapping
    aliases, or multiple format ids attached to one backing descriptor.
    """
    if descriptors is None:
        raise ValueError("CB descriptor override must not be None")
    try:
        descriptors = list(descriptors)
    except TypeError as exc:
        raise ValueError("CB descriptor override must be iterable") from exc

    program_cores = _core_range_coordinates(
        program_core_ranges, label="program core ranges"
    )
    format_by_id: Dict[int, Tuple[str, Optional[Tuple[int, int]], int]] = {}
    claims: Dict[Tuple[int, Tuple[int, int]], int] = {}
    bytes_by_core = {core: 0 for core in program_cores}
    claim_sizes_by_core: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
        core: [] for core in program_cores
    }

    for descriptor_index, descriptor in enumerate(descriptors):
        formats = list(getattr(descriptor, "format_descriptors", ()))
        if len(formats) != 1:
            raise ValueError(
                "CB descriptor override entry "
                f"{descriptor_index} must contain exactly one format descriptor"
            )
        fmt = formats[0]
        try:
            cb_id = int(fmt.buffer_index)
            page_size = int(fmt.page_size)
            total_size = int(descriptor.total_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"CB descriptor override entry {descriptor_index} is malformed"
            ) from exc
        if cb_id < 0 or cb_id >= num_cbs:
            raise ValueError(
                f"CB descriptor override entry {descriptor_index} has CB id "
                f"{cb_id}, outside [0, {num_cbs})"
            )
        if page_size <= 0 or total_size <= 0 or total_size % page_size:
            raise ValueError(
                f"CB[{cb_id}] override requires positive page-aligned total_size; "
                f"got total_size={total_size}, page_size={page_size}"
            )

        # Real TTNN descriptors expose the enum safely through the integer
        # binding.  Accessing ``data_format`` directly is broken on some
        # pybind builds (it returns an unregistered ``tt::DataFormat``).
        data_format = getattr(fmt, "data_format_as_uint8", None)
        if data_format is None:
            data_format = getattr(fmt, "data_format", None)
        if data_format is None:
            raise ValueError(f"CB[{cb_id}] override is missing a data format")
        tile_key = _tile_descriptor_key(getattr(fmt, "tile", None))
        format_key = (str(data_format), tile_key, page_size)
        previous_format = format_by_id.setdefault(cb_id, format_key)
        if previous_format != format_key:
            raise ValueError(
                f"CB[{cb_id}] override uses inconsistent page formats across cores: "
                f"{previous_format} vs {format_key}"
            )

        descriptor_cores = _core_range_coordinates(
            getattr(descriptor, "core_ranges", None),
            label=f"CB[{cb_id}] descriptor core ranges",
        )
        outside = descriptor_cores - program_cores
        if outside:
            raise ValueError(
                f"CB[{cb_id}] override claims cores outside the program grid: "
                f"{sorted(outside)}"
            )
        for core in descriptor_cores:
            claim_key = (cb_id, core)
            if claim_key in claims:
                raise ValueError(
                    f"CB[{cb_id}] override has overlapping descriptors on core {core}"
                )
            claims[claim_key] = descriptor_index
            bytes_by_core[core] += total_size
            claim_sizes_by_core[core].append((cb_id, total_size))

    required_ids = (
        set(range(num_cbs))
        if required_cb_ids is None
        else set(required_cb_ids)
    )
    invalid_required = sorted(
        cb_id for cb_id in required_ids if cb_id < 0 or cb_id >= num_cbs
    )
    if invalid_required:
        raise ValueError(
            f"required CB ids are outside [0, {num_cbs}): {invalid_required}"
        )
    missing_ids = sorted(required_ids - set(format_by_id))
    if missing_ids:
        raise ValueError(
            "CB descriptor override does not describe every configured CB id; "
            f"missing {missing_ids}"
        )

    budget_bytes = _remaining_cb_budget(tensors)
    budgets_by_core = _remaining_cb_budgets_by_core(
        tensors, program_cores
    )
    if bytes_by_core:
        peak_core, peak_bytes = max(
            bytes_by_core.items(), key=lambda item: (item[1], item[0])
        )
        if os.environ.get("TTLANG_DUMP_CB_LAYOUT"):
            print(
                "TTLANG_CB_LAYOUT "
                f"budget={budget_bytes} peak={peak_bytes} core={peak_core}"
            )
            for core, total in sorted(
                bytes_by_core.items(), key=lambda item: item[1], reverse=True
            )[:12]:
                breakdown = ",".join(
                    f"{cb_id}:{size}"
                    for cb_id, size in sorted(claim_sizes_by_core[core])
                )
                print(
                    f"TTLANG_CB_CORE core={core} bytes={total} "
                    f"budget={budgets_by_core[core]} {breakdown}"
                )
        overflow_core, overflow_bytes = max(
            bytes_by_core.items(),
            key=lambda item: (
                item[1] - budgets_by_core[item[0]], item[0]
            ),
        )
        overflow_budget = budgets_by_core[overflow_core]
        if overflow_bytes > overflow_budget:
            breakdown = ", ".join(
                f"CB[{cb_id}]={size}"
                for cb_id, size in sorted(
                    claim_sizes_by_core[overflow_core]
                )
            )
            raise ValueError(
                "Per-core circular buffer descriptor override allocation "
                f"({overflow_bytes} bytes on core {overflow_core}) exceeds "
                f"L1 budget ({overflow_budget} bytes). Claims: {breakdown}"
            )

    return descriptors


def _cb_descriptor(index: int, geometry: CBGeometry, total_size: int, core_ranges):
    cb_format = ttnn.CBFormatDescriptor(
        buffer_index=index,
        data_format=geometry.data_format,
        page_size=geometry.page_size,
        **(
            {"tile": geometry.tile_descriptor}
            if geometry.tile_descriptor is not None
            else {}
        ),
    )
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[cb_format],
    )


def _core_ranges_from_coordinates(coordinates: set[Tuple[int, int]]):
    """Build a compact, deterministic CoreRangeSet for exact coordinates."""

    remaining = set(coordinates)
    rectangles = []
    while remaining:
        start_x, start_y = min(remaining, key=lambda core: (core[1], core[0]))
        end_x = start_x
        while (end_x + 1, start_y) in remaining:
            end_x += 1
        end_y = start_y
        while all(
            (x, end_y + 1) in remaining
            for x in range(start_x, end_x + 1)
        ):
            end_y += 1
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                remaining.remove((x, y))
        rectangles.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(start_x, start_y),
                ttnn.CoreCoord(end_x, end_y),
            )
        )
    return ttnn.CoreRangeSet(rectangles)


def _used_cb_indices_by_core(
    kernel_specs: Optional[List[KernelSpec]],
    program_core_ranges: Any,
    num_cbs: int,
) -> Optional[Dict[Tuple[int, int], set[int]]]:
    """Union specialized kernel CB use on each logical core.

    Returns None when no kernel carries the annotation, preserving the
    historical whole-grid descriptor behavior. In a mixed set, an unannotated
    kernel conservatively uses every configured slot on its own launch cores.
    """

    if not kernel_specs or not any(
        spec.used_cb_indices is not None for spec in kernel_specs
    ):
        return None

    program_cores = _core_range_coordinates(
        program_core_ranges, label="program core ranges"
    )
    used_by_core = {core: set() for core in program_cores}
    all_indices = set(range(num_cbs))
    for spec_index, spec in enumerate(kernel_specs):
        spec_ranges = (
            spec.core_ranges
            if spec.core_ranges is not None
            else program_core_ranges
        )
        spec_cores = _core_range_coordinates(
            spec_ranges, label=f"kernel spec {spec_index} core ranges"
        )
        outside = spec_cores - program_cores
        if outside:
            raise ValueError(
                f"kernel spec {spec_index} claims cores outside the program "
                f"grid: {sorted(outside)}"
            )
        indices = (
            all_indices
            if spec.used_cb_indices is None
            else {int(index) for index in spec.used_cb_indices}
        )
        invalid = sorted(
            index for index in indices if index < 0 or index >= num_cbs
        )
        if invalid:
            raise ValueError(
                f"kernel spec {spec_index} uses CB ids outside "
                f"[0, {num_cbs}): {invalid}"
            )
        for core in spec_cores:
            used_by_core[core].update(indices)
    return used_by_core


def build_cb_descriptors_by_core(
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
    pages_by_core: Dict[int, List[Tuple[Any, int]]],
    kernel_specs: Optional[List[KernelSpec]] = None,
) -> List[Any]:
    """Build a full descriptor table with selected per-core capacities."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    geometries = [cb_geometry(i, cb) for i, cb in enumerate(cb_configs)]
    program_cores = _core_range_coordinates(
        core_ranges, label="program core ranges"
    )
    specialized = {}
    for raw_index, entries in dict(pages_by_core).items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid per-core CB id {raw_index!r}") from exc
        if index < 0 or index >= len(geometries):
            raise ValueError(
                f"per-core CB id {index} is outside [0, {len(geometries)})"
            )
        if index in specialized:
            raise ValueError(f"duplicate per-core CB id {index}")
        try:
            entries = list(entries)
        except TypeError as exc:
            raise ValueError(
                f"per-core CB[{index}] configuration must be iterable"
            ) from exc
        if not entries:
            raise ValueError(f"per-core CB[{index}] configuration is empty")

        covered = set()
        configured_pages = []
        pages_for_core = {}
        for entry_index, entry in enumerate(entries):
            try:
                entry_ranges, raw_pages = entry
                pages = int(raw_pages)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"per-core CB[{index}] entry {entry_index} must be "
                    "(CoreRangeSet, positive pages)"
                ) from exc
            if pages <= 0:
                raise ValueError(
                    f"per-core CB[{index}] entry {entry_index} has "
                    f"non-positive page count {pages}"
                )
            entry_cores = _core_range_coordinates(
                entry_ranges,
                label=f"per-core CB[{index}] entry {entry_index}",
            )
            outside = entry_cores - program_cores
            overlap = entry_cores & covered
            if outside:
                raise ValueError(
                    f"per-core CB[{index}] claims cores outside the program "
                    f"grid: {sorted(outside)}"
                )
            if overlap:
                raise ValueError(
                    f"per-core CB[{index}] entries overlap on "
                    f"{sorted(overlap)}"
                )
            covered.update(entry_cores)
            configured_pages.append(pages)
            pages_for_core.update({core: pages for core in entry_cores})
        if covered != program_cores:
            raise ValueError(
                f"per-core CB[{index}] must cover the whole program grid; "
                f"missing {sorted(program_cores - covered)}"
            )
        if max(configured_pages) != geometries[index].num_pages:
            raise ValueError(
                f"per-core CB[{index}] must preserve the compiler-derived "
                f"maximum of {geometries[index].num_pages} pages; got "
                f"{max(configured_pages)}"
            )
        specialized[index] = pages_for_core

    used_by_core = _used_cb_indices_by_core(
        kernel_specs, core_ranges, len(geometries)
    )
    # TT-Metal keeps one allocation cursor per descriptor CoreRange. Refine
    # every descriptor onto one common disjoint partition, keyed by the complete
    # per-core page signature. A zero page count means the specialized kernels
    # on that partition never access this hardware slot, so no descriptor is
    # emitted there.
    cores_by_signature = {}
    for core in sorted(program_cores):
        signature = tuple(
            (
                0
                if used_by_core is not None
                and index not in used_by_core[core]
                else (
                    specialized[index][core]
                    if index in specialized
                    else geometry.num_pages
                )
            )
            for index, geometry in enumerate(geometries)
        )
        cores_by_signature.setdefault(signature, set()).add(core)
    partitions = [
        (signature, _core_ranges_from_coordinates(coordinates))
        for signature, coordinates in cores_by_signature.items()
    ]

    descriptors = []
    required_cb_ids = set()
    descriptor_indices = (
        [
            index
            for index in range(len(geometries))
            if index not in specialized
        ]
        + list(specialized)
        if used_by_core is None
        else list(range(len(geometries)))
    )
    for index in descriptor_indices:
        geometry = geometries[index]
        for signature, partition_ranges in partitions:
            pages = signature[index]
            if pages == 0:
                continue
            required_cb_ids.add(index)
            descriptors.append(
                _cb_descriptor(
                    index,
                    geometry,
                    pages * geometry.page_size,
                    partition_ranges,
                )
            )
    return validate_cb_descriptors_override(
        descriptors=descriptors,
        program_core_ranges=core_ranges,
        tensors=tensors,
        num_cbs=len(cb_configs),
        required_cb_ids=required_cb_ids,
    )


def build_cb_descriptors(
    tensors: List[Any],
    cb_configs: List[Any],
    core_ranges: Any,
    kernel_specs: Optional[List[KernelSpec]] = None,
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

    Returns:
        List of ttnn.CBDescriptor objects.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    if _used_cb_indices_by_core(
        kernel_specs, core_ranges, len(cb_configs)
    ) is not None:
        return build_cb_descriptors_by_core(
            tensors=tensors,
            cb_configs=cb_configs,
            core_ranges=core_ranges,
            pages_by_core={},
            kernel_specs=kernel_specs,
        )

    # Compute sizes first so we fail before allocating ttnn descriptors on overflow.
    geometries = [cb_geometry(i, cb) for i, cb in enumerate(cb_configs)]
    total_cb_bytes = sum(g.total_size for g in geometries)

    remaining_bytes = _remaining_cb_budget(tensors)

    # Must stay aligned with MLIR ttl-validate-cb-budget (TileType::getSizeBytes) and
    # tile_bytes_from_dtype; see issue #511.
    if total_cb_bytes > remaining_bytes:
        breakdown = "\n".join(g.breakdown for g in geometries)
        raise ValueError(
            "Total circular buffer allocation ("
            f"{total_cb_bytes} bytes) exceeds L1 budget ({remaining_bytes} bytes). "
            "This checks static CB backing store only (not all L1 on core).\n"
            + breakdown
            + "\n  hint: reduce DFB shapes or block_count."
        )

    return [
        _cb_descriptor(i, geometry, geometry.total_size, core_ranges)
        for i, geometry in enumerate(geometries)
    ]


def build_cb_descriptors_from_layouts(
    tensors: List[Any],
    cb_layouts: List[Tuple[Any, int, Any, int, int]],
    core_ranges: Any,
    kernel_specs: Optional[List[KernelSpec]] = None,
) -> List[Any]:
    """Rebuild live descriptor scoping from serialized emitted-runner layouts."""

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    geometries = []
    for shape, block_count, data_format, page_size, total_size in cb_layouts:
        geometries.append(
            CBGeometry(
                data_format=data_format,
                page_size=int(page_size),
                num_pages=int(total_size) // int(page_size),
                total_size=int(total_size),
                tile_descriptor=None,
                tile=None,
                shape=tuple(shape),
                block_count=int(block_count),
                breakdown="",
            )
        )

    used_by_core = _used_cb_indices_by_core(
        kernel_specs, core_ranges, len(geometries)
    )
    if used_by_core is None:
        return [
            _cb_descriptor(index, geometry, geometry.total_size, core_ranges)
            for index, geometry in enumerate(geometries)
        ]

    cores_by_signature = {}
    for core, used_indices in sorted(used_by_core.items()):
        signature = tuple(index in used_indices for index in range(len(geometries)))
        cores_by_signature.setdefault(signature, set()).add(core)
    partitions = [
        (signature, _core_ranges_from_coordinates(coordinates))
        for signature, coordinates in cores_by_signature.items()
    ]
    descriptors = []
    required_cb_ids = set()
    for index, geometry in enumerate(geometries):
        for signature, partition_ranges in partitions:
            if not signature[index]:
                continue
            required_cb_ids.add(index)
            descriptors.append(
                _cb_descriptor(
                    index, geometry, geometry.total_size, partition_ranges
                )
            )
    return validate_cb_descriptors_override(
        descriptors=descriptors,
        program_core_ranges=core_ranges,
        tensors=tensors,
        num_cbs=len(geometries),
        required_cb_ids=required_cb_ids,
    )


def build_generic_op_io_tensors(
    tensors: List[Any],
    pipe_sram_scratch_tensors: List[Any],
) -> List[Any]:
    """Return io_tensors for ttnn.generic_op, including pipe SRAM scratch."""
    io_tensors = list(tensors) + list(pipe_sram_scratch_tensors)
    if not io_tensors:
        raise ValueError("kernel must have at least one output tensor")
    if len(io_tensors) < 2:
        io_tensors = [io_tensors[-1]] + io_tensors
    return io_tensors


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
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    runtime_resource_lifetime: Optional[List[Any]] = None,
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
            ready counters allocated by the compiler.
        pipe_global_semaphore_lifetime: Optional list replaced with the current
            call's GlobalSemaphore objects. Cached kernels keep this bounded
            owner list so repeated calls do not retain old semaphore objects.

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
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
    )
    if pipe_global_semaphore_lifetime is not None:
        pipe_global_semaphore_lifetime[:] = pipe_runtime_resources.global_semaphores

    program_resources = ProgramRuntimeResources()
    if runtime_resource_factory is not None:
        program_resources = runtime_resource_factory(
            tensors=list(tensors),
            core_ranges=core_ranges,
            first_free_semaphore_id=num_pipe_sync_semaphores,
        )
        if not isinstance(program_resources, ProgramRuntimeResources):
            raise TypeError(
                "runtime_resource_factory must return ProgramRuntimeResources, "
                f"got {type(program_resources).__name__}"
            )
    if runtime_resource_lifetime is not None:
        runtime_resource_lifetime[:] = program_resources.lifetimes

    # Build kernel descriptors.
    kernel_descriptors = build_kernel_descriptors(
        kernel_specs=kernel_specs,
        tensors=tensors,
        tensor_accessor_args=tensor_accessor_args,
        core_ranges=core_ranges,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        num_cbs=len(cb_configs),
        extra_common_runtime_args=pipe_runtime_resources.extra_common_runtime_args,
        expected_extra_common_runtime_args=(
            pipe_runtime_resources.expected_extra_common_runtime_args
        ),
        runtime_args_by_thread=program_resources.runtime_args_by_thread,
        defines_by_thread=program_resources.defines_by_thread,
    )

    # Build CB descriptors, unless this operation supplied an exact per-core
    # replacement.  Keep the bypass here (after resource construction) so all
    # other operations retain the compiler-derived whole-grid behavior.
    if (
        program_resources.cb_descriptors_override is not None
        and program_resources.cb_pages_by_core
    ):
        raise ValueError(
            "ProgramRuntimeResources cannot set both cb_descriptors_override "
            "and cb_pages_by_core"
        )
    if program_resources.cb_descriptors_override is not None:
        cb_descriptors = validate_cb_descriptors_override(
            descriptors=program_resources.cb_descriptors_override,
            program_core_ranges=core_ranges,
            tensors=tensors,
            num_cbs=len(cb_configs),
        )
    elif program_resources.cb_pages_by_core:
        cb_descriptors = build_cb_descriptors_by_core(
            tensors=tensors,
            cb_configs=cb_configs,
            core_ranges=core_ranges,
            pages_by_core=program_resources.cb_pages_by_core,
            kernel_specs=kernel_specs,
        )
    else:
        cb_descriptors = build_cb_descriptors(
            tensors=tensors,
            cb_configs=cb_configs,
            core_ranges=core_ranges,
            kernel_specs=kernel_specs,
        )

    semaphore_descriptors = build_pipe_sync_semaphore_descriptors(
        core_ranges=core_ranges,
        count=num_pipe_sync_semaphores,
    )
    semaphore_descriptors.extend(program_resources.semaphore_descriptors)

    # Build and execute program.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=semaphore_descriptors,
    )
    normalized_program_hash = normalize_program_hash(program_hash)
    if normalized_program_hash is not None:
        program.custom_program_hash = normalized_program_hash

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
) -> str:
    """
    Emit Python source code for a standalone runner that invokes ttnn.generic_op.

    Generates a ready-to-use Python file with all the CB and kernel
    descriptor setup. Tensor-specific values (buffer addresses, accessor args)
    are marked with TODO comments for the user to fill in.

    program_hash, if provided, is normalized to uint64 and embedded as the
    emitted runner's tt-metal program-cache key.
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
    lines.append("    build_cb_descriptors_from_layouts,")
    lines.append("    build_generic_op_io_tensors,")
    lines.append("    build_kernel_descriptors,")
    lines.append("    build_pipe_runtime_resources,")
    lines.append("    build_pipe_sync_semaphore_descriptors,")
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

    lines.append("KERNEL_USED_CB_INDICES = [")
    for spec in kernel_specs:
        lines.append(
            f"    {spec.used_cb_indices!r},  # {spec.thread_type}"
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

    lines.append("CB_CONFIGS = [")
    _ensure_ttnn()
    if ttnn is None and cb_configs:
        raise RuntimeError("ttnn is not available")
    for i, cb in enumerate(cb_configs):
        if cb is None:
            lines.append(f"    None,  # CB {i}")
            continue
        if isinstance(cb, CompilerAllocatedDFBConfig):
            data_format = format_name_to_ttnn_dtype(cb.data_format)
            page_size = tile_bytes_from_dtype(data_format)
            num_tiles = cb.num_tiles * cb.block_count
            shape = (1, cb.num_tiles)
        else:
            data_format = _cb_data_format(cb)
            page_size = ttnn.Tile(cb.tile).get_tile_size(data_format)
            num_tiles = cb.shape[0] * cb.shape[1] * cb.block_count
            shape = cb.shape
        dtype_str = _dtype_to_ttnn_str(data_format)
        total_size = num_tiles * page_size
        lines.append(
            f"    ({shape!r}, {cb.block_count}, {dtype_str}, {page_size}, {total_size}),  # CB {i}"
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
    lines.append("        pipe_sram_scratch_bytes=PIPE_SRAM_SCRATCH_BYTES,")
    lines.append("        num_pipe_global_semaphores=NUM_PIPE_GLOBAL_SEMAPHORES,")
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
        "                core_ranges=_core_ranges_from_spec("
        "KERNEL_CORE_RANGES[kernel_idx]),"
    )
    lines.append(
        "                used_cb_indices=KERNEL_USED_CB_INDICES[kernel_idx],"
    )
    lines.append("            )")
    lines.append("        )")
    lines.append("    cb_descriptors = build_cb_descriptors_from_layouts(")
    lines.append("        tensors=tensors,")
    lines.append("        cb_layouts=CB_CONFIGS,")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        kernel_specs=kernel_specs,")
    lines.append("    )")
    lines.append("    kernel_descriptors = build_kernel_descriptors(")
    lines.append("        kernel_specs=kernel_specs,")
    lines.append("        tensors=tensors,")
    lines.append("        tensor_accessor_args=tensor_accessor_args,")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        grid_cols=GRID_COLS,")
    lines.append("        grid_rows=GRID_ROWS,")
    lines.append("        num_cbs=len(CB_CONFIGS),")
    lines.append(
        "        extra_common_runtime_args=pipe_resources.extra_common_runtime_args,"
    )
    lines.append("        expected_extra_common_runtime_args=(")
    lines.append("            pipe_resources.expected_extra_common_runtime_args")
    lines.append("        ),")
    lines.append("    )")
    lines.append("")

    lines.append("    semaphore_descriptors = build_pipe_sync_semaphore_descriptors(")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        count=NUM_PIPE_SYNC_SEMAPHORES,")
    lines.append("    )")
    lines.append("")

    lines.append("    program = ttnn.ProgramDescriptor(")
    lines.append("        kernels=kernel_descriptors,")
    lines.append("        cbs=cb_descriptors,")
    lines.append("        semaphores=semaphore_descriptors,")
    lines.append("    )")
    lines.append("    if PROGRAM_HASH is not None:")
    lines.append("        program.custom_program_hash = PROGRAM_HASH")
    lines.append("")
    lines.append("    io_tensors = build_generic_op_io_tensors(")
    lines.append("        tensors=tensors,")
    lines.append("        pipe_sram_scratch_tensors=pipe_resources.scratch_tensors,")
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
) -> str:
    """
    Emit a Python runner file for the compiled kernel.

    program_hash, if provided, is forwarded to the emitted runner as its
    normalized tt-metal program-cache key.

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
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(source)

    print(f"Runner written to {output_path}")
    return output_path


__all__ = [
    "CBGeometry",
    "KernelSpec",
    "PipeRuntimeResources",
    "ProgramRuntimeResources",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "build_cb_descriptors_by_core",
    "validate_cb_descriptors_override",
    "cb_geometry",
    "build_pipe_sram_scratch_tensors",
    "build_pipe_global_semaphores",
    "build_pipe_runtime_resources",
    "build_pipe_sync_semaphore_descriptors",
    "normalize_program_hash",
    "build_generic_op_io_tensors",
    "run_kernel_on_device",
    "emit_runner_source",
    "emit_runner_file",
]
