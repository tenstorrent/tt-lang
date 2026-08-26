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

from dataclasses import dataclass, field, replace
import hashlib
import json
import operator
import os
import threading
import warnings
import weakref
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ttnn = None  # Lazy-loaded via _ensure_ttnn()


def _ensure_ttnn():
    """Lazy import of ttnn."""
    global ttnn
    if ttnn is not None:
        return ttnn
    try:
        import ttnn as _ttnn

        if not hasattr(_ttnn, "DataType"):
            return None
        ttnn = _ttnn
    except (ModuleNotFoundError, ImportError):
        pass
    return ttnn


from .dataflow_buffer import (
    DFBReconfigurationPlan,
    DFBStorageSegment,
    PhysicalDFBConfig,
    _validate_tensor_backed_dfb_range,
    _validate_tensor_backed_dfb_tensor,
)
from .constants import DEFAULT_L1_CB_BUDGET_BYTES
from . import dtype_utils
from .kernel import Kernel, KernelKind, KernelSelector
from .runtime_resources import (
    CoreRuntimeArgs,
    KernelDefine,
    KernelRuntimeResources,
    ProgramRuntimeResources,
)


def format_name_to_ttnn_dtype(name: str):
    """Resolve a DFB format name using this module's ttnn binding."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    return dtype_utils.format_name_to_ttnn_dtype(name, ttnn)


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


def _detect_device_arch(device) -> Optional[str]:
    """Return a normalized architecture string from a TTNN device if present."""
    for attribute_name in (
        "arch",
        "architecture",
        "chip_type",
        "device_type",
        "_arch",
        "_architecture",
    ):
        # Closed device handles may raise exceptions other than AttributeError.
        try:
            architecture = getattr(device, attribute_name)
        except Exception:
            continue
        if callable(architecture):
            try:
                architecture = architecture()
            except Exception:
                continue
        return str(architecture).lower().rsplit(".", maxsplit=1)[-1]
    return None


def _get_l1_allocation_quantum_bytes(device) -> int:
    """Return the largest L1 allocation quantum supported for the target."""
    architecture = _detect_device_arch(device)
    if architecture == "wormhole_b0":
        return 32
    return 64


def _device_l1_cb_usage(
    device,
    excluded_l1_buffer_addresses: Sequence[int] = (),
) -> Tuple[int, Dict[Tuple[int, int], int]]:
    """Return ``(cb_limit, L1 bytes used per logical (x, y) core)``."""
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    budget_bytes = ttnn._ttnn.reports.get_device_info(device).cb_limit
    excluded_addresses = frozenset(
        int(address) for address in excluded_l1_buffer_addresses
    )
    used_bytes: Dict[Tuple[int, int], int] = {}
    for page in ttnn._ttnn.reports.get_buffer_pages(device):
        if page.buffer_type != ttnn.BufferType.L1:
            continue
        if int(page.address) in excluded_addresses:
            continue
        core = (page.core_x, page.core_y)
        used_bytes[core] = used_bytes.get(core, 0) + page.page_size
    return budget_bytes, used_bytes


def get_min_remaining_l1_for_device(
    device, excluded_l1_buffer_addresses: Sequence[int] = ()
):
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

    ``excluded_l1_buffer_addresses`` removes retained compiler-owned buffers
    before computing the per-core maximum. This reconstructs the compilation
    budget without changing the contribution of unrelated allocations.
    """
    budget_bytes, bytes_per_core = _device_l1_cb_usage(
        device, excluded_l1_buffer_addresses
    )
    max_core_bytes = max(bytes_per_core.values()) if bytes_per_core else 0
    return max(0, budget_bytes - max_core_bytes)


def _get_remaining_l1_by_core_for_device(
    device, cores: set[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    """Return each requested logical core's remaining static DFB budget."""
    budget_bytes, used_bytes = _device_l1_cb_usage(device)
    return {core: max(0, budget_bytes - used_bytes.get(core, 0)) for core in cores}


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
        logical_kernel: Target-independent selector retained across kernel cloning.
        used_dfb_indices: Physical DFB slots referenced by the final kernel body.
            None means metadata is unavailable and conservatively uses every DFB;
            an empty list means this kernel uses no DFBs.
    """

    path: str
    thread_type: str
    tensor_indices: List[int]
    config: Any
    compiler_include_paths: List[str] = field(default_factory=list)
    pipe_computed_address_dfb_indices: List[int] = field(default_factory=list)
    core_ranges: Optional[Any] = None
    logical_kernel: Optional[KernelSelector] = None
    used_dfb_indices: Optional[List[int]] = None


@dataclass(frozen=True)
class LogicalKernelId:
    kind: KernelKind
    name: Optional[str]
    operation: Optional[str]
    implicit_role: Optional[str]


@dataclass(frozen=True)
class _CoreRuntimeArgsPlan:
    coordinate: Tuple[int, int]
    values: Tuple[int, ...]


@dataclass(frozen=True)
class _KernelDescriptorResourcePlan:
    kernel_spec_index: int
    logical_kernel: LogicalKernelId
    coordinates: Tuple[Tuple[int, int], ...]
    runtime_args: Tuple[_CoreRuntimeArgsPlan, ...]
    defines: Tuple[Tuple[str, str], ...]


@dataclass
class _KernelDescriptorVariant:
    core_ranges: Any
    compile_time_args: List[int]
    runtime_args: Any


@dataclass(frozen=True)
class ProgramResourcePlan:
    semaphore_descriptors: Tuple[object, ...]
    kernel_descriptors: Tuple[_KernelDescriptorResourcePlan, ...]
    lifetimes: Tuple[object, ...]
    structural_fingerprint: int


@dataclass(frozen=True)
class _SemaphoreResourceFingerprint:
    semaphore_id: int
    coordinates: Tuple[Tuple[int, int], ...]
    initial_value: int
    core_type: str


@dataclass
class PipeRuntimeResources:
    """Host allocations and runtime args for compiler-emitted pipe resources."""

    scratch_tensors: List[Any]
    global_semaphores: List[Any]
    computed_address_dfb_tensors: Dict[int, Any]
    computed_address_dfb_allocation_bytes: Dict[int, int]
    computed_address_base_addresses: Dict[int, int]
    extra_common_runtime_args: List[int]
    expected_extra_common_runtime_args: int
    l1_buffer_addresses: frozenset[int] = frozenset()


@dataclass
class KernelRuntimeResourceCache:
    """Persistent L1 resources shared by serialized cached invocations."""

    lock: Any = field(default_factory=threading.RLock, repr=False)
    compatibility_key: Optional[Tuple[Any, ...]] = None
    device: Optional[Any] = None
    pipe_resources: Optional[PipeRuntimeResources] = None
    reconfiguration_resources: Optional["DFBReconfigurationRuntimeResources"] = None
    owned_l1_buffer_addresses: frozenset[int] = frozenset()
    portable_resource_lifetimes: Tuple[object, ...] = ()
    portable_resource_device: Optional[Any] = None


def _release_portable_runtime_resources_impl(
    cache: KernelRuntimeResourceCache,
) -> None:
    if not cache.portable_resource_lifetimes:
        return
    if cache.portable_resource_device is not None:
        _ensure_ttnn()
        if ttnn is None:
            raise RuntimeError("ttnn is not available")
        ttnn.synchronize_device(cache.portable_resource_device)
    cache.portable_resource_lifetimes = ()
    cache.portable_resource_device = None


def _release_cached_runtime_resources_impl(
    cache: KernelRuntimeResourceCache,
) -> None:
    if cache.compatibility_key is None and not cache.portable_resource_lifetimes:
        return
    resource_device = (
        cache.device if cache.device is not None else cache.portable_resource_device
    )
    if resource_device is not None:
        _ensure_ttnn()
        if ttnn is None:
            raise RuntimeError("ttnn is not available")
        ttnn.synchronize_device(resource_device)
    cache.compatibility_key = None
    cache.device = None
    cache.pipe_resources = None
    cache.reconfiguration_resources = None
    cache.owned_l1_buffer_addresses = frozenset()
    cache.portable_resource_lifetimes = ()
    cache.portable_resource_device = None


def release_cached_runtime_resources(cache: KernelRuntimeResourceCache) -> None:
    """Synchronize and release one operation's persistent L1 resources."""
    with cache.lock:
        _release_cached_runtime_resources_impl(cache)


# A failed device synchronization must retain owners referenced by in-flight work.
_RETAINED_RUNTIME_RESOURCE_CACHES = []


def finalize_runtime_resource_cache(runtime_resource_cache):
    """Synchronize before releasing resources owned by a collected object."""
    try:
        release_cached_runtime_resources(runtime_resource_cache)
    except BaseException as error:
        _RETAINED_RUNTIME_RESOURCE_CACHES.append(runtime_resource_cache)
        error_message = str(error) or type(error).__name__
        try:
            warnings.warn(
                f"failed to synchronize operation runtime resources: {error_message}",
                RuntimeWarning,
                stacklevel=2,
            )
        except BaseException:
            pass


def attach_runtime_resource_finalizer(owner, runtime_resource_cache):
    """Attach exception-safe resource cleanup to a weak-referenceable owner."""
    resource_finalizer = weakref.finalize(
        owner, finalize_runtime_resource_cache, runtime_resource_cache
    )
    resource_finalizer.atexit = False
    return resource_finalizer


def _retain_unsynchronized_runtime_resources(
    device,
    pipe_resources: PipeRuntimeResources,
    reconfiguration_resources: "DFBReconfigurationRuntimeResources",
    portable_resource_lifetimes: Tuple[object, ...] = (),
) -> None:
    """Retain one uncached generation when device completion is unknown."""
    retained_cache = KernelRuntimeResourceCache(
        compatibility_key=("uncached-unsynchronized",),
        device=device,
        pipe_resources=pipe_resources,
        reconfiguration_resources=reconfiguration_resources,
        portable_resource_lifetimes=portable_resource_lifetimes,
        portable_resource_device=device,
    )
    _RETAINED_RUNTIME_RESOURCE_CACHES.append(retained_cache)


def _detach_cached_runtime_resources(
    cache: KernelRuntimeResourceCache,
) -> None:
    """Move an unsynchronized generation out of the active cache."""
    retained_cache = KernelRuntimeResourceCache(
        compatibility_key=cache.compatibility_key,
        device=cache.device,
        pipe_resources=cache.pipe_resources,
        reconfiguration_resources=cache.reconfiguration_resources,
        owned_l1_buffer_addresses=cache.owned_l1_buffer_addresses,
        portable_resource_lifetimes=cache.portable_resource_lifetimes,
        portable_resource_device=cache.portable_resource_device,
    )
    if (
        retained_cache.pipe_resources is not None
        or retained_cache.reconfiguration_resources is not None
        or retained_cache.portable_resource_lifetimes
    ):
        _RETAINED_RUNTIME_RESOURCE_CACHES.append(retained_cache)
    cache.compatibility_key = None
    cache.device = None
    cache.pipe_resources = None
    cache.reconfiguration_resources = None
    cache.owned_l1_buffer_addresses = frozenset()
    cache.portable_resource_lifetimes = ()
    cache.portable_resource_device = None


def _invalidate_cached_runtime_resources_after_dispatch_error(
    cache: KernelRuntimeResourceCache,
) -> None:
    """Synchronize and discard state that a failed dispatch may have changed."""
    try:
        _release_cached_runtime_resources_impl(cache)
    except BaseException:
        _detach_cached_runtime_resources(cache)
        raise


def _synchronize_or_retain_runtime_resources(
    device,
    pipe_resources: PipeRuntimeResources,
    reconfiguration_resources: "DFBReconfigurationRuntimeResources",
    portable_resource_lifetimes: Tuple[object, ...] = (),
) -> None:
    """Synchronize one uncached generation or retain all of its owners."""
    try:
        ttnn.synchronize_device(device)
    except BaseException:
        _retain_unsynchronized_runtime_resources(
            device,
            pipe_resources,
            reconfiguration_resources,
            portable_resource_lifetimes,
        )
        raise


def _format_logical_kernel(kernel: LogicalKernelId) -> str:
    if kernel.name is None:
        return f"canonical {kernel.kind.value} kernel"
    return f"{kernel.kind.value} kernel {kernel.name!r}"


def _normalize_logical_kernel_selector(
    selector: object,
    *,
    operation_name: str,
    source: str,
) -> LogicalKernelId:
    if isinstance(selector, KernelKind):
        return LogicalKernelId(selector, None, None, None)
    if not isinstance(selector, Kernel):
        raise TypeError(
            f"@ttl.operation {operation_name!r}: {source} must select a "
            f"KernelKind or Kernel, got {type(selector).__name__}"
        )

    try:
        name = selector.identity
    except ValueError as error:
        raise ValueError(
            f"@ttl.operation {operation_name!r}: {source} uses an unbound Kernel"
        ) from error
    operation_identity = selector._operation_identity
    implicit_role = selector._implicit_role
    assert (operation_identity is None) != (
        implicit_role is None
    ), "a bound Kernel must identify exactly one operation or compiler-owned role"
    return LogicalKernelId(
        selector.kind,
        name,
        operation_identity,
        implicit_role,
    )


def _normalize_index(
    value: object,
    *,
    operation_name: str,
    field: str,
) -> int:
    if isinstance(value, bool):
        raise TypeError(
            f"@ttl.operation {operation_name!r}: {field} must be an integer, "
            "got bool"
        )
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(
            f"@ttl.operation {operation_name!r}: {field} must be an integer, "
            f"got {type(value).__name__}"
        ) from error


def _normalize_coordinate(
    core: object,
    *,
    operation_name: str,
    field: str,
) -> Tuple[int, int]:
    try:
        core_x = core.x
        core_y = core.y
    except AttributeError as error:
        raise TypeError(
            f"@ttl.operation {operation_name!r}: {field} must provide integer "
            "x and y coordinates"
        ) from error
    coordinate = (
        _normalize_index(
            core_x,
            operation_name=operation_name,
            field=f"{field}.x",
        ),
        _normalize_index(
            core_y,
            operation_name=operation_name,
            field=f"{field}.y",
        ),
    )
    if coordinate[0] < 0 or coordinate[1] < 0:
        raise ValueError(
            f"@ttl.operation {operation_name!r}: {field} coordinate "
            f"{coordinate} must be nonnegative"
        )
    return coordinate


def _canonicalize_core_ranges(
    core_ranges: object,
    *,
    operation_name: str,
    field: str,
) -> Tuple[Tuple[int, int], ...]:
    try:
        ranges = tuple(core_ranges.ranges())
    except AttributeError as error:
        raise TypeError(
            f"@ttl.operation {operation_name!r}: {field} must provide ranges()"
        ) from error

    coordinates = set()
    for range_index, core_range in enumerate(ranges):
        try:
            start = _normalize_coordinate(
                core_range.start,
                operation_name=operation_name,
                field=f"{field} range {range_index} start",
            )
            end = _normalize_coordinate(
                core_range.end,
                operation_name=operation_name,
                field=f"{field} range {range_index} end",
            )
        except AttributeError as error:
            raise TypeError(
                f"@ttl.operation {operation_name!r}: {field} range "
                f"{range_index} must provide start and end coordinates"
            ) from error
        if start[0] > end[0] or start[1] > end[1]:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: {field} range "
                f"{range_index} has start {start} after end {end}"
            )
        coordinates.update(
            (core_x, core_y)
            for core_y in range(start[1], end[1] + 1)
            for core_x in range(start[0], end[0] + 1)
        )
    if not coordinates:
        raise ValueError(
            f"@ttl.operation {operation_name!r}: {field} must not be empty"
        )
    return tuple(
        sorted(coordinates, key=lambda coordinate: (coordinate[1], coordinate[0]))
    )


def _normalize_defines(
    defines: Tuple[KernelDefine, ...],
    *,
    operation_name: str,
    resource_index: int,
) -> Tuple[Tuple[str, str], ...]:
    normalized = []
    names = set()
    for define_index, define in enumerate(defines):
        if not isinstance(define.name, str):
            raise TypeError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} define {define_index} name must be a str, "
                f"got {type(define.name).__name__}"
            )
        if not define.name or "\0" in define.name:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} define {define_index} name must be nonempty "
                "and contain no NUL"
            )
        if not isinstance(define.value, str):
            raise TypeError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} define {define_index} value must be a str, "
                f"got {type(define.value).__name__}"
            )
        if define.name in names:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} defines name {define.name!r} more than once"
            )
        names.add(define.name)
        normalized.append((define.name, define.value))
    return tuple(normalized)


def _normalize_runtime_args(
    runtime_args: Tuple[CoreRuntimeArgs, ...],
    *,
    operation_name: str,
    resource_index: int,
    operation_coordinates: frozenset[Tuple[int, int]],
) -> Tuple[_CoreRuntimeArgsPlan, ...]:
    normalized = []
    seen_coordinates = set()
    for runtime_arg_index, runtime_arg in enumerate(runtime_args):
        coordinate = _normalize_coordinate(
            runtime_arg.core,
            operation_name=operation_name,
            field=(
                f"kernel resource {resource_index} runtime argument "
                f"{runtime_arg_index} core"
            ),
        )
        if coordinate not in operation_coordinates:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} runtime argument {runtime_arg_index} core "
                f"{coordinate} is outside the operation core range"
            )
        if coordinate in seen_coordinates:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} specifies runtime arguments for core "
                f"{coordinate} more than once"
            )
        seen_coordinates.add(coordinate)
        values = tuple(
            _normalize_index(
                value,
                operation_name=operation_name,
                field=(
                    f"kernel resource {resource_index} runtime argument "
                    f"{runtime_arg_index} value {value_index}"
                ),
            )
            for value_index, value in enumerate(runtime_arg.values)
        )
        normalized.append(_CoreRuntimeArgsPlan(coordinate, values))
    return tuple(
        sorted(
            normalized,
            key=lambda runtime_arg: (
                runtime_arg.coordinate[1],
                runtime_arg.coordinate[0],
            ),
        )
    )


def _normalize_semaphore_core_type(
    core_type: object,
    *,
    operation_name: str,
    descriptor_index: int,
) -> str:
    if isinstance(core_type, str):
        name = core_type
    else:
        name = getattr(core_type, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"@ttl.operation {operation_name!r}: semaphore descriptor "
            f"{descriptor_index} core_type must be a named value, got "
            f"{type(core_type).__name__}"
        )
    return name


def _validate_semaphore_descriptors(
    semaphore_descriptors: Tuple[object, ...],
    *,
    operation_name: str,
    operation_coordinates: frozenset[Tuple[int, int]],
    first_free_semaphore_id: int,
) -> Tuple[_SemaphoreResourceFingerprint, ...]:
    seen_ids = set()
    fingerprints = []
    for descriptor_index, descriptor in enumerate(semaphore_descriptors):
        semaphore_id = _normalize_index(
            descriptor.id,
            operation_name=operation_name,
            field=f"semaphore descriptor {descriptor_index} id",
        )
        if semaphore_id < first_free_semaphore_id:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: semaphore descriptor "
                f"{descriptor_index} id {semaphore_id} is below first free "
                f"semaphore id {first_free_semaphore_id}"
            )
        if semaphore_id in seen_ids:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: semaphore id "
                f"{semaphore_id} was specified more than once"
            )
        seen_ids.add(semaphore_id)
        descriptor_coordinates = _canonicalize_core_ranges(
            descriptor.core_ranges,
            operation_name=operation_name,
            field=f"semaphore descriptor {descriptor_index} core_ranges",
        )
        outside_coordinates = frozenset(descriptor_coordinates) - operation_coordinates
        if outside_coordinates:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: semaphore descriptor "
                f"{descriptor_index} has cores outside the operation range: "
                f"{tuple(sorted(outside_coordinates, key=lambda core: (core[1], core[0])))}"
            )
        initial_value = _normalize_index(
            descriptor.initial_value,
            operation_name=operation_name,
            field=f"semaphore descriptor {descriptor_index} initial_value",
        )
        fingerprints.append(
            _SemaphoreResourceFingerprint(
                semaphore_id=semaphore_id,
                coordinates=descriptor_coordinates,
                initial_value=initial_value,
                core_type=_normalize_semaphore_core_type(
                    descriptor.core_type,
                    operation_name=operation_name,
                    descriptor_index=descriptor_index,
                ),
            )
        )
    return tuple(sorted(fingerprints, key=lambda descriptor: descriptor.semaphore_id))


def _validate_runtime_resource_record_types(
    resources: object,
    *,
    operation_name: str,
) -> ProgramRuntimeResources:
    if not isinstance(resources, ProgramRuntimeResources):
        raise TypeError(
            f"@ttl.operation {operation_name!r}: runtime_resource_factory must "
            "return ProgramRuntimeResources, got "
            f"{type(resources).__name__}"
        )
    for field_name in (
        "semaphore_descriptors",
        "kernel_resources",
        "lifetimes",
    ):
        field_value = getattr(resources, field_name)
        if not isinstance(field_value, tuple):
            raise TypeError(
                f"@ttl.operation {operation_name!r}: {field_name} must be a "
                f"tuple, got {type(field_value).__name__}"
            )

    for resource_index, kernel_resource in enumerate(resources.kernel_resources):
        if not isinstance(kernel_resource, KernelRuntimeResources):
            raise TypeError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} must be KernelRuntimeResources, got "
                f"{type(kernel_resource).__name__}"
            )
        if not isinstance(kernel_resource.runtime_args, tuple):
            raise TypeError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} runtime_args must be a tuple, got "
                f"{type(kernel_resource.runtime_args).__name__}"
            )
        if not isinstance(kernel_resource.defines, tuple):
            raise TypeError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} defines must be a tuple, got "
                f"{type(kernel_resource.defines).__name__}"
            )
        for runtime_arg_index, runtime_arg in enumerate(kernel_resource.runtime_args):
            if not isinstance(runtime_arg, CoreRuntimeArgs):
                raise TypeError(
                    f"@ttl.operation {operation_name!r}: kernel resource "
                    f"{resource_index} runtime argument {runtime_arg_index} must "
                    f"be CoreRuntimeArgs, got {type(runtime_arg).__name__}"
                )
            if not isinstance(runtime_arg.values, tuple):
                raise TypeError(
                    f"@ttl.operation {operation_name!r}: kernel resource "
                    f"{resource_index} runtime argument {runtime_arg_index} values "
                    f"must be a tuple, got {type(runtime_arg.values).__name__}"
                )
        for define_index, define in enumerate(kernel_resource.defines):
            if not isinstance(define, KernelDefine):
                raise TypeError(
                    f"@ttl.operation {operation_name!r}: kernel resource "
                    f"{resource_index} define {define_index} must be a "
                    f"KernelDefine, got {type(define).__name__}"
                )

    for descriptor_index, descriptor in enumerate(resources.semaphore_descriptors):
        for field_name in ("id", "core_ranges", "initial_value", "core_type"):
            if not hasattr(descriptor, field_name):
                raise TypeError(
                    f"@ttl.operation {operation_name!r}: semaphore descriptor "
                    f"{descriptor_index} must provide {field_name}"
                )
    return resources


_RESOURCE_PLAN_VERSION = 1
_RESOURCE_PLAN_PERSONALIZATION = b"ttlang-rr-plan"
_RESOURCE_HASH_PERSONALIZATION = b"ttlang-rr-hash"


def _digest_primitive_payload(payload: object, personalization: bytes) -> int:
    encoded_payload = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.blake2b(
        encoded_payload,
        digest_size=8,
        person=personalization,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _compute_resource_plan_fingerprint(
    kernel_descriptors: Tuple[_KernelDescriptorResourcePlan, ...],
    semaphore_descriptors: Tuple[_SemaphoreResourceFingerprint, ...],
) -> int:
    kernel_payload = tuple(
        (
            descriptor.logical_kernel.kind.value,
            descriptor.logical_kernel.name,
            descriptor.coordinates,
            descriptor.defines,
            tuple(
                (runtime_arg.coordinate, len(runtime_arg.values))
                for runtime_arg in descriptor.runtime_args
            ),
        )
        for descriptor in kernel_descriptors
    )
    semaphore_payload = tuple(
        (
            descriptor.semaphore_id,
            descriptor.coordinates,
            descriptor.initial_value,
            descriptor.core_type,
        )
        for descriptor in semaphore_descriptors
    )
    return _digest_primitive_payload(
        (
            "operation-runtime-resource-plan",
            _RESOURCE_PLAN_VERSION,
            kernel_payload,
            semaphore_payload,
        ),
        _RESOURCE_PLAN_PERSONALIZATION,
    )


def plan_program_runtime_resources(
    *,
    operation_name: str,
    resources: ProgramRuntimeResources,
    kernel_specs: Sequence[KernelSpec],
    operation_core_ranges: object,
    first_free_semaphore_id: int,
) -> ProgramResourcePlan:
    resources = _validate_runtime_resource_record_types(
        resources,
        operation_name=operation_name,
    )

    normalized_first_free_id = _normalize_index(
        first_free_semaphore_id,
        operation_name=operation_name,
        field="first_free_semaphore_id",
    )
    if normalized_first_free_id < 0:
        raise ValueError(
            f"@ttl.operation {operation_name!r}: first_free_semaphore_id must "
            f"be nonnegative, got {normalized_first_free_id}"
        )

    operation_coordinates_tuple = _canonicalize_core_ranges(
        operation_core_ranges,
        operation_name=operation_name,
        field="operation core_ranges",
    )
    operation_coordinates = frozenset(operation_coordinates_tuple)

    descriptor_identities = []
    descriptor_coordinates = []
    descriptors_by_identity: Dict[LogicalKernelId, List[int]] = {}
    for kernel_spec_index, kernel_spec in enumerate(kernel_specs):
        logical_kernel = _normalize_logical_kernel_selector(
            kernel_spec.logical_kernel,
            operation_name=operation_name,
            source=f"kernel descriptor {kernel_spec_index}",
        )
        coordinates = (
            operation_coordinates_tuple
            if kernel_spec.core_ranges is None
            else _canonicalize_core_ranges(
                kernel_spec.core_ranges,
                operation_name=operation_name,
                field=f"kernel descriptor {kernel_spec_index} core_ranges",
            )
        )
        outside_coordinates = frozenset(coordinates) - operation_coordinates
        if outside_coordinates:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: kernel descriptor "
                f"{kernel_spec_index} has cores outside the operation range: "
                f"{tuple(sorted(outside_coordinates, key=lambda core: (core[1], core[0])))}"
            )
        descriptor_identities.append(logical_kernel)
        descriptor_coordinates.append(coordinates)
        descriptors_by_identity.setdefault(logical_kernel, []).append(kernel_spec_index)

    for logical_kernel, matching_descriptors in descriptors_by_identity.items():
        for position, previous_descriptor_index in enumerate(matching_descriptors):
            previous_coordinates = frozenset(
                descriptor_coordinates[previous_descriptor_index]
            )
            for descriptor_index in matching_descriptors[position + 1 :]:
                overlap = tuple(
                    candidate_coordinate
                    for candidate_coordinate in descriptor_coordinates[descriptor_index]
                    if candidate_coordinate in previous_coordinates
                )
                assert not overlap, (
                    f"compiler emitted kernel descriptors "
                    f"{previous_descriptor_index} and {descriptor_index} for "
                    f"{_format_logical_kernel(logical_kernel)} with overlapping "
                    f"cores {overlap}"
                )

    descriptor_runtime_args: Dict[int, List[_CoreRuntimeArgsPlan]] = {}
    descriptor_defines: Dict[int, Tuple[Tuple[str, str], ...]] = {}
    seen_resource_identities = set()
    for resource_index, kernel_resource in enumerate(resources.kernel_resources):
        logical_kernel = _normalize_logical_kernel_selector(
            kernel_resource.kernel,
            operation_name=operation_name,
            source=f"kernel resource {resource_index}",
        )
        if logical_kernel in seen_resource_identities:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: runtime resources for "
                f"{_format_logical_kernel(logical_kernel)} were specified more "
                "than once"
            )
        seen_resource_identities.add(logical_kernel)
        matching_descriptors = descriptors_by_identity.get(logical_kernel, [])
        if not matching_descriptors:
            raise ValueError(
                f"@ttl.operation {operation_name!r}: kernel resource "
                f"{resource_index} selects {_format_logical_kernel(logical_kernel)}, "
                "but the operation emitted no matching kernel descriptor"
            )
        normalized_defines = _normalize_defines(
            kernel_resource.defines,
            operation_name=operation_name,
            resource_index=resource_index,
        )
        for descriptor_index in matching_descriptors:
            descriptor_defines[descriptor_index] = normalized_defines
        normalized_runtime_args = _normalize_runtime_args(
            kernel_resource.runtime_args,
            operation_name=operation_name,
            resource_index=resource_index,
            operation_coordinates=operation_coordinates,
        )
        for runtime_arg_index, runtime_arg in enumerate(normalized_runtime_args):
            candidate_descriptors = [
                descriptor_index
                for descriptor_index in matching_descriptors
                if runtime_arg.coordinate
                in frozenset(descriptor_coordinates[descriptor_index])
            ]
            if not candidate_descriptors:
                ranges = tuple(
                    (descriptor_index, descriptor_coordinates[descriptor_index])
                    for descriptor_index in matching_descriptors
                )
                raise ValueError(
                    f"@ttl.operation {operation_name!r}: kernel resource "
                    f"{resource_index} runtime argument {runtime_arg_index} core "
                    f"{runtime_arg.coordinate} is not covered by any descriptor "
                    f"for {_format_logical_kernel(logical_kernel)}; descriptor "
                    f"ranges are {ranges}"
                )
            if len(candidate_descriptors) != 1:
                raise AssertionError(
                    "validated descriptor partitions must select exactly one "
                    "descriptor"
                )
            descriptor_index = candidate_descriptors[0]
            descriptor_runtime_args.setdefault(descriptor_index, []).append(runtime_arg)

    semaphore_fingerprints = _validate_semaphore_descriptors(
        resources.semaphore_descriptors,
        operation_name=operation_name,
        operation_coordinates=operation_coordinates,
        first_free_semaphore_id=normalized_first_free_id,
    )
    kernel_descriptor_plans = tuple(
        _KernelDescriptorResourcePlan(
            kernel_spec_index=kernel_spec_index,
            logical_kernel=logical_kernel,
            coordinates=descriptor_coordinates[kernel_spec_index],
            runtime_args=tuple(descriptor_runtime_args.get(kernel_spec_index, ())),
            defines=descriptor_defines.get(kernel_spec_index, ()),
        )
        for kernel_spec_index, logical_kernel in enumerate(descriptor_identities)
    )
    return ProgramResourcePlan(
        semaphore_descriptors=resources.semaphore_descriptors,
        kernel_descriptors=kernel_descriptor_plans,
        lifetimes=resources.lifetimes,
        structural_fingerprint=_compute_resource_plan_fingerprint(
            kernel_descriptor_plans,
            semaphore_fingerprints,
        ),
    )


@dataclass
class DFBReconfigurationRuntimeResources:
    """Host allocations referenced by synchronized DFB reconfiguration."""

    scratch_tensors: Dict[int, Any]
    configuration_tensors: List[Any]
    configuration_runtime_args: Dict[Tuple[int, int], List[int]]
    device: Optional[Any] = None
    l1_buffer_addresses: frozenset[int] = frozenset()


_DFB_RECONFIGURATION_MAX_INDICES = 64
_DFB_RECONFIGURATION_WORDS_PER_DFB = 4
_DFB_RECONFIGURATION_LOW_MASK_WORD = (
    _DFB_RECONFIGURATION_MAX_INDICES * _DFB_RECONFIGURATION_WORDS_PER_DFB
)
_DFB_RECONFIGURATION_HIGH_MASK_WORD = _DFB_RECONFIGURATION_LOW_MASK_WORD + 1
_DFB_RECONFIGURATION_SYNCHRONIZATION_WORD = _DFB_RECONFIGURATION_HIGH_MASK_WORD + 1
_DFB_RECONFIGURATION_WORDS_PER_CORE = _DFB_RECONFIGURATION_SYNCHRONIZATION_WORD + 6


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


def _make_singleton_core_ranges(coordinates: Iterable[Tuple[int, int]]) -> Any:
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(core_x, core_y),
                ttnn.CoreCoord(core_x, core_y),
            )
            for core_x, core_y in coordinates
        ]
    )


def _build_reconfiguration_descriptor_variants(
    kernel_ranges: Any,
    cb_indices: List[int],
    tensor_accessor_args: List[int],
    thread_type: str,
    caller_runtime_args: Sequence[Tuple[Any, Sequence[int]]],
    reconfiguration_args: Dict[Tuple[int, int], List[int]],
) -> List[_KernelDescriptorVariant]:
    caller_args_by_core = {
        (int(core.x), int(core.y)): values for core, values in caller_runtime_args
    }
    variants_by_caller_arg_count: Dict[int, List[Tuple[Any, List[int]]]] = {}
    for core in ttnn.corerange_to_cores(kernel_ranges, row_wise=True):
        core_key = (int(core.x), int(core.y))
        if core_key not in reconfiguration_args:
            raise RuntimeError(
                f"missing DFB reconfiguration runtime arguments for core {core_key}"
            )
        caller_args = list(caller_args_by_core.get(core_key, ()))
        combined_args = caller_args + list(reconfiguration_args[core_key])
        variants_by_caller_arg_count.setdefault(len(caller_args), []).append(
            (core, combined_args)
        )

    descriptor_variants: List[_KernelDescriptorVariant] = []
    for caller_arg_count, core_args in sorted(variants_by_caller_arg_count.items()):
        variant_runtime_args = ttnn.RuntimeArgs()
        variant_cores = []
        for core, values in core_args:
            variant_cores.append(core)
            variant_runtime_args[core.x][core.y] = values
        compile_time_args = cb_indices + [caller_arg_count]
        if thread_type != "compute":
            compile_time_args.extend(tensor_accessor_args)
        descriptor_variants.append(
            _KernelDescriptorVariant(
                core_ranges=_make_singleton_core_ranges(
                    (int(core.x), int(core.y)) for core in variant_cores
                ),
                compile_time_args=compile_time_args,
                runtime_args=variant_runtime_args,
            )
        )
    return descriptor_variants


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
    descriptor_resource_plans: Optional[Sequence[_KernelDescriptorResourcePlan]] = None,
    dfb_reconfiguration_runtime_args: Optional[Dict[Tuple[int, int], List[int]]] = None,
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
        descriptor_resource_plans: Immutable caller resource plans aligned with
            kernel_specs.
        dfb_reconfiguration_runtime_args: Per-core L1 configuration addresses
            in finalized boundary order.

    Returns:
        List of ttnn.KernelDescriptor objects.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    kernel_descriptors = []
    if descriptor_resource_plans is not None and len(descriptor_resource_plans) != len(
        kernel_specs
    ):
        raise ValueError(
            "kernel descriptor resource plan count must match kernel spec count: "
            f"got {len(descriptor_resource_plans)} plans for {len(kernel_specs)} specs"
        )

    # CB indices are 0, 1, 2, ... for each CB (including intermediate CBs).
    cb_indices = list(range(num_cbs))
    computed_address_base_addresses = pipe_computed_address_base_addresses or {}
    extra_args = list(extra_common_runtime_args or [])
    reconfiguration_args = dict(dfb_reconfiguration_runtime_args or {})
    if (
        expected_extra_common_runtime_args is not None
        and len(extra_args) != expected_extra_common_runtime_args
    ):
        raise RuntimeError(
            "pipe resource plan expected "
            f"{expected_extra_common_runtime_args} extra common runtime args, "
            f"got {len(extra_args)}"
        )

    for kernel_spec_index, spec in enumerate(kernel_specs):
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

        # Prefer per-kernel core_ranges (specialize-cores clones); otherwise
        # fall back to the whole-grid core_ranges.
        kernel_ranges = (
            spec.core_ranges if spec.core_ranges is not None else core_ranges
        )
        runtime_args = []
        defines = []
        if descriptor_resource_plans is not None:
            descriptor_resource_plan = descriptor_resource_plans[kernel_spec_index]
            if descriptor_resource_plan.kernel_spec_index != kernel_spec_index:
                raise ValueError(
                    "kernel descriptor resource plan index mismatch: expected "
                    f"{kernel_spec_index}, got "
                    f"{descriptor_resource_plan.kernel_spec_index}"
                )
            runtime_args = [
                (
                    ttnn.CoreCoord(
                        runtime_arg.coordinate[0], runtime_arg.coordinate[1]
                    ),
                    list(runtime_arg.values),
                )
                for runtime_arg in descriptor_resource_plan.runtime_args
            ]
            defines = list(descriptor_resource_plan.defines)

        descriptor_variants: List[_KernelDescriptorVariant]
        if not reconfiguration_args:
            kernel_compile_time_args = list(cb_indices)
            if spec.thread_type != "compute":
                kernel_compile_time_args.extend(tensor_accessor_args)
            descriptor_variants = [
                _KernelDescriptorVariant(
                    core_ranges=kernel_ranges,
                    compile_time_args=kernel_compile_time_args,
                    runtime_args=runtime_args,
                )
            ]
        else:
            descriptor_variants = _build_reconfiguration_descriptor_variants(
                kernel_ranges,
                cb_indices,
                tensor_accessor_args,
                spec.thread_type,
                runtime_args,
                reconfiguration_args,
            )

        for descriptor_variant in descriptor_variants:
            kernel_descriptor_args = dict(
                kernel_source=spec.path,
                core_ranges=descriptor_variant.core_ranges,
                compile_time_args=descriptor_variant.compile_time_args,
                defines=defines,
                common_runtime_args=common_runtime_args,
                config=spec.config,
                compiler_include_paths=spec.compiler_include_paths,
            )
            if descriptor_variant.runtime_args:
                kernel_descriptor_args["runtime_args"] = descriptor_variant.runtime_args
            kernel_descriptors.append(ttnn.KernelDescriptor(**kernel_descriptor_args))

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


def _device_identity(device: Any) -> Any:
    if device is None:
        return None
    device_id = getattr(device, "id", None)
    if callable(device_id):
        return ("device-id", device_id())
    return ("object-id", id(device))


def _same_device(lhs: Any, rhs: Any) -> bool:
    return _device_identity(lhs) == _device_identity(rhs)


def _allocate_l1_sharded_storage_tensor(
    core_ranges: Any, num_bytes: int, device: Any, *, zero_initialize: bool = False
):
    """Allocate row-major L1 storage with one 4-byte element per storage word."""
    aligned_bytes = _align_up(num_bytes, 32)
    elements_per_core = max(1, aligned_bytes // 4)
    num_cores = core_ranges.num_cores()
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
    allocator = ttnn.zeros if zero_initialize else ttnn.empty
    return allocator(
        (num_cores, elements_per_core),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _l1_buffer_addresses_by_core(
    tensor: Any, device: Any
) -> Dict[Tuple[int, int], int]:
    """Return each shard's physical L1 base indexed by logical core."""
    buffer_address = int(tensor.buffer_address())
    addresses = {}
    for page in ttnn._ttnn.reports.get_buffer_pages(device):
        if (
            page.buffer_type != ttnn.BufferType.L1
            or int(page.address) != buffer_address
        ):
            continue
        core = (int(page.core_x), int(page.core_y))
        page_address = int(page.page_address)
        addresses[core] = min(addresses.get(core, page_address), page_address)
    return addresses


def build_pipe_sram_scratch_tensors(
    tensors: List[Any],
    core_ranges: Any,
    scratch_bytes: int,
    device: Optional[Any] = None,
    *,
    zero_initialize: bool = False,
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
    return [
        _allocate_l1_sharded_storage_tensor(
            core_ranges,
            scratch_bytes,
            device,
            zero_initialize=zero_initialize,
        )
    ]


def build_pipe_global_semaphores(
    tensors: List[Any],
    core_ranges: Any,
    count: int,
    device: Optional[Any] = None,
) -> Tuple[List[Any], List[int]]:
    """Allocate GlobalSemaphores used by compiler-managed PipeNet counters.

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


def _get_pipe_computed_address_dfb_allocation_bytes(
    cb_configs: List[PhysicalDFBConfig],
    dfb_indices: List[int],
    plan: Optional[DFBReconfigurationPlan],
) -> Dict[int, int]:
    """Return maximum compiler-managed storage required by each receiver DFB."""
    if plan is not None and len(plan.dfb_epochs) != len(cb_configs):
        raise ValueError(
            "DFB reconfiguration plan must describe every physical DFB config"
        )

    allocation_bytes_by_index = {}
    for dfb_index in dfb_indices:
        if dfb_index < 0 or dfb_index >= len(cb_configs):
            raise ValueError(
                f"computed-address receiver DFB index {dfb_index} is invalid"
            )
        configurations = [cb_configs[dfb_index]]
        if plan is not None:
            configurations.extend(epoch.config for epoch in plan.dfb_epochs[dfb_index])

        max_allocation_bytes = 0
        for config in configurations:
            _validate_physical_dfb_config(config, dfb_index)
            if config.storage_segments and all(
                segment.is_tensor_backed for segment in config.storage_segments
            ):
                continue
            max_allocation_bytes = max(
                max_allocation_bytes, _get_dfb_allocation(config).total_size
            )
        if max_allocation_bytes == 0:
            raise ValueError(
                f"computed-address receiver DFB[{dfb_index}] has no "
                "compiler-managed storage"
            )
        allocation_bytes_by_index[dfb_index] = max_allocation_bytes
    return allocation_bytes_by_index


def build_pipe_computed_address_dfb_tensors(
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    pipe_computed_address_dfb_indices: Optional[List[int]] = None,
    device: Optional[Any] = None,
    kernel_specs: Optional[List[KernelSpec]] = None,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
) -> Dict[int, Any]:
    """Allocate hidden L1 backing tensors for computed pipe receiver DFBs."""
    dfb_indices = sorted(set(pipe_computed_address_dfb_indices or []))
    if not dfb_indices:
        return {}

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    device = device if device is not None else _first_device(tensors)
    allocation_bytes_by_index = _get_pipe_computed_address_dfb_allocation_bytes(
        cb_configs, dfb_indices, dfb_reconfiguration_plan
    )
    used_by_core = _used_dfb_indices_by_core(kernel_specs, core_ranges, len(cb_configs))
    backing_tensors = {}
    for dfb_index in dfb_indices:
        backing_core_ranges = core_ranges
        if used_by_core is not None:
            backing_cores = tuple(
                sorted(
                    core
                    for core, used_indices in used_by_core.items()
                    if dfb_index in used_indices
                )
            )
            if backing_cores:
                backing_core_ranges = _make_singleton_core_ranges(backing_cores)
        backing_tensors[dfb_index] = _allocate_l1_sharded_storage_tensor(
            backing_core_ranges, allocation_bytes_by_index[dfb_index], device
        )
    return backing_tensors


def build_pipe_runtime_resources(
    tensors: List[Any],
    core_ranges: Any,
    cb_configs: Optional[List[PhysicalDFBConfig]] = None,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    pipe_computed_address_dfb_indices: Optional[List[int]] = None,
    device: Optional[Any] = None,
    initialize_sram_scratch: bool = False,
    kernel_specs: Optional[List[KernelSpec]] = None,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
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
    computed_address_dfb_allocation_bytes = {}
    if computed_address_dfb_indices:
        if cb_configs is None:
            raise ValueError(
                "computed-address receiver DFB base allocation requires DFB configs"
            )
        computed_address_dfb_allocation_bytes = (
            _get_pipe_computed_address_dfb_allocation_bytes(
                cb_configs,
                sorted(set(computed_address_dfb_indices)),
                dfb_reconfiguration_plan,
            )
        )
        computed_address_dfb_tensors = build_pipe_computed_address_dfb_tensors(
            tensors=tensors,
            cb_configs=cb_configs,
            core_ranges=core_ranges,
            pipe_computed_address_dfb_indices=computed_address_dfb_indices,
            device=resource_device,
            kernel_specs=kernel_specs,
            dfb_reconfiguration_plan=dfb_reconfiguration_plan,
        )

    scratch_tensors = build_pipe_sram_scratch_tensors(
        tensors=tensors,
        core_ranges=core_ranges,
        scratch_bytes=pipe_sram_scratch_bytes,
        device=resource_device,
        zero_initialize=initialize_sram_scratch,
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
    l1_buffer_addresses = frozenset(
        [
            *(int(tensor.buffer_address()) for tensor in scratch_tensors),
            *global_semaphore_addresses,
            *computed_address_base_addresses.values(),
        ]
    )
    return PipeRuntimeResources(
        scratch_tensors=scratch_tensors,
        global_semaphores=global_semaphores,
        computed_address_dfb_tensors=computed_address_dfb_tensors,
        computed_address_dfb_allocation_bytes=(computed_address_dfb_allocation_bytes),
        computed_address_base_addresses=computed_address_base_addresses,
        extra_common_runtime_args=extra_common_runtime_args,
        expected_extra_common_runtime_args=expected_extra_common_runtime_args,
        l1_buffer_addresses=l1_buffer_addresses,
    )


def _runtime_resource_compatibility_key(
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    pipe_sram_scratch_bytes: int,
    num_pipe_global_semaphores: int,
    pipe_computed_address_dfb_indices: Tuple[int, ...],
    num_dfb_resets: int,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan],
    device: Optional[Any],
) -> Tuple[Tuple[Any, ...], Optional[Any]]:
    requires_device = (
        pipe_sram_scratch_bytes > 0
        or num_pipe_global_semaphores > 0
        or bool(pipe_computed_address_dfb_indices)
        or dfb_reconfiguration_plan is not None
    )
    resource_device = device
    if resource_device is None and requires_device:
        resource_device = _first_device(tensors)

    core_key = ()
    if requires_device:
        core_key = tuple(
            (int(core.x), int(core.y))
            for core in ttnn.corerange_to_cores(core_ranges, row_wise=True)
        )
    tensor_address_key = []
    if dfb_reconfiguration_plan is not None:
        _validate_dfb_reconfiguration_plan(tensors, dfb_reconfiguration_plan)
        tensor_indices = sorted(
            {
                segment.tensor_index
                for epochs in dfb_reconfiguration_plan.dfb_epochs
                for epoch in epochs
                for segment in epoch.config.storage_segments
                if segment.tensor_index is not None
            }
        )
        for tensor_index in tensor_indices:
            addresses = _l1_buffer_addresses_by_core(
                tensors[tensor_index], resource_device
            )
            tensor_address_key.append((tensor_index, tuple(sorted(addresses.items()))))
    compatibility_key = (
        _device_identity(resource_device),
        core_key,
        tuple(cb_configs),
        pipe_sram_scratch_bytes,
        num_pipe_global_semaphores,
        pipe_computed_address_dfb_indices,
        num_dfb_resets,
        dfb_reconfiguration_plan,
        tuple(tensor_address_key),
    )
    return compatibility_key, resource_device


def _get_cached_runtime_resources_impl(
    cache: Optional[KernelRuntimeResourceCache],
    *,
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    pipe_sram_scratch_bytes: int,
    num_pipe_global_semaphores: int,
    pipe_computed_address_dfb_indices: Tuple[int, ...],
    num_dfb_resets: int,
    device: Optional[Any],
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
    kernel_specs: Optional[List[KernelSpec]] = None,
) -> Tuple[PipeRuntimeResources, DFBReconfigurationRuntimeResources]:
    pipe_computed_address_dfb_indices = tuple(pipe_computed_address_dfb_indices)
    compatibility_key, resource_device = _runtime_resource_compatibility_key(
        tensors,
        cb_configs,
        core_ranges,
        pipe_sram_scratch_bytes,
        num_pipe_global_semaphores,
        pipe_computed_address_dfb_indices,
        num_dfb_resets,
        dfb_reconfiguration_plan,
        device,
    )
    if (
        cache is not None
        and cache.compatibility_key == compatibility_key
        and cache.pipe_resources is not None
        and cache.reconfiguration_resources is not None
    ):
        if num_pipe_global_semaphores == 0:
            return cache.pipe_resources, cache.reconfiguration_resources
        _release_cached_runtime_resources_impl(cache)

    if cache is not None and cache.compatibility_key is not None:
        _release_cached_runtime_resources_impl(cache)

    pipe_resources = build_pipe_runtime_resources(
        tensors=tensors,
        core_ranges=core_ranges,
        cb_configs=cb_configs,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
        pipe_computed_address_dfb_indices=list(pipe_computed_address_dfb_indices),
        device=resource_device,
        initialize_sram_scratch=num_dfb_resets > 0,
        kernel_specs=kernel_specs,
        dfb_reconfiguration_plan=dfb_reconfiguration_plan,
    )
    reconfiguration_resources = build_dfb_reconfiguration_runtime_resources(
        tensors=tensors,
        core_ranges=core_ranges,
        plan=dfb_reconfiguration_plan,
        existing_backing_tensors=pipe_resources.computed_address_dfb_tensors,
        existing_backing_allocation_bytes=(
            pipe_resources.computed_address_dfb_allocation_bytes
        ),
        device=resource_device,
    )
    if cache is not None:
        cache.compatibility_key = compatibility_key
        cache.device = resource_device
        cache.pipe_resources = pipe_resources
        cache.reconfiguration_resources = reconfiguration_resources
        cache.owned_l1_buffer_addresses = (
            pipe_resources.l1_buffer_addresses
            | reconfiguration_resources.l1_buffer_addresses
        )
    return pipe_resources, reconfiguration_resources


def get_min_remaining_l1_excluding_cached_resources(
    cache: KernelRuntimeResourceCache, device: Any
) -> int:
    """Return the current L1 budget with this cache's buffers excluded."""
    with cache.lock:
        excluded_addresses = (
            cache.owned_l1_buffer_addresses
            if _same_device(cache.device, device)
            else ()
        )
        return get_min_remaining_l1_for_device(device, excluded_addresses)


def get_cached_runtime_resources(
    cache: Optional[KernelRuntimeResourceCache],
    *,
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    pipe_sram_scratch_bytes: int,
    num_pipe_global_semaphores: int,
    pipe_computed_address_dfb_indices: Tuple[int, ...],
    num_dfb_resets: int,
    device: Optional[Any],
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
    kernel_specs: Optional[List[KernelSpec]] = None,
) -> Tuple[PipeRuntimeResources, DFBReconfigurationRuntimeResources]:
    """Return one compatible resource generation from a synchronized cache."""
    arguments = {
        "tensors": tensors,
        "cb_configs": cb_configs,
        "core_ranges": core_ranges,
        "pipe_sram_scratch_bytes": pipe_sram_scratch_bytes,
        "num_pipe_global_semaphores": num_pipe_global_semaphores,
        "pipe_computed_address_dfb_indices": pipe_computed_address_dfb_indices,
        "num_dfb_resets": num_dfb_resets,
        "dfb_reconfiguration_plan": dfb_reconfiguration_plan,
        "device": device,
        "kernel_specs": kernel_specs,
    }
    if cache is None:
        return _get_cached_runtime_resources_impl(None, **arguments)
    with cache.lock:
        return _get_cached_runtime_resources_impl(cache, **arguments)


def build_dfb_reconfiguration_runtime_resources(
    tensors: List[Any],
    core_ranges: Any,
    plan: Optional[DFBReconfigurationPlan],
    existing_backing_tensors: Optional[Dict[int, Any]] = None,
    existing_backing_allocation_bytes: Optional[Dict[int, int]] = None,
    device: Optional[Any] = None,
) -> DFBReconfigurationRuntimeResources:
    """Allocate scratch storage and one shared L1 configuration per boundary."""
    if plan is None:
        return DFBReconfigurationRuntimeResources({}, [], {}, device)

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    _validate_dfb_reconfiguration_plan(tensors, plan)
    resource_device = device if device is not None else _first_device(tensors)
    reusable_backing_tensors = dict(existing_backing_tensors or {})
    reusable_backing_allocation_bytes = dict(existing_backing_allocation_bytes or {})
    if set(reusable_backing_tensors) != set(reusable_backing_allocation_bytes):
        raise ValueError(
            "existing DFB backing tensors and allocation sizes must use the "
            "same physical indices"
        )
    all_cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
    core_keys = [(int(core.x), int(core.y)) for core in all_cores]
    scratch_bytes_by_index = {}
    scratch_nodes_by_index = {}
    for dfb_index, epochs in enumerate(plan.dfb_epochs):
        scratch_bytes = 0
        scratch_nodes = set()
        for epoch in epochs:
            config = epoch.config
            allocation = _get_dfb_allocation(config)
            if not config.storage_segments:
                scratch_bytes = max(scratch_bytes, allocation.total_size)
                scratch_nodes.update(core_keys)
                continue
            scratch_segments = tuple(
                segment
                for segment in config.storage_segments
                if not segment.is_tensor_backed
            )
            if scratch_segments:
                scratch_bytes = max(scratch_bytes, allocation.total_size)
                for segment in scratch_segments:
                    scratch_nodes.update(segment.nodes)
        if scratch_bytes > 0:
            scratch_bytes_by_index[dfb_index] = scratch_bytes
            scratch_nodes_by_index[dfb_index] = scratch_nodes

    scratch_tensors = {}
    for dfb_index, scratch_bytes in scratch_bytes_by_index.items():
        existing_tensor = reusable_backing_tensors.get(dfb_index)
        if existing_tensor is not None:
            backing_allocation_bytes = reusable_backing_allocation_bytes[dfb_index]
            if scratch_bytes > backing_allocation_bytes:
                raise ValueError(
                    f"DFB[{dfb_index}] PipeNet backing is smaller than its "
                    "reconfiguration scratch requirement"
                )
            scratch_tensors[dfb_index] = existing_tensor
            continue

    core_rows = {core: row for row, core in enumerate(core_keys)}
    if len(plan.dfb_epochs) > _DFB_RECONFIGURATION_MAX_INDICES:
        raise ValueError(
            "DFB reconfiguration supports at most "
            f"{_DFB_RECONFIGURATION_MAX_INDICES} physical indices"
        )
    for dfb_index, scratch_nodes in scratch_nodes_by_index.items():
        outside_nodes = scratch_nodes.difference(core_rows)
        if outside_nodes:
            outside_node = min(outside_nodes)
            raise ValueError(
                f"DFB[{dfb_index}] configuration references launch node "
                f"{outside_node} outside the kernel grid"
            )

    import torch

    host_configurations = {
        boundary_ordinal: torch.zeros(
            (len(core_keys), _DFB_RECONFIGURATION_WORDS_PER_CORE),
            dtype=torch.uint32,
        )
        for boundary_ordinal in plan.boundary_ordinals
    }

    for dfb_index, scratch_bytes in scratch_bytes_by_index.items():
        if dfb_index in scratch_tensors:
            continue
        scratch_tensors[dfb_index] = _allocate_l1_sharded_storage_tensor(
            _make_singleton_core_ranges(sorted(scratch_nodes_by_index[dfb_index])),
            scratch_bytes,
            resource_device,
        )

    configuration_runtime_args = {core: [] for core in core_keys}
    configuration_tensors = []
    tensor_addresses_by_core = {}
    scratch_addresses_by_index = {
        dfb_index: _l1_buffer_addresses_by_core(tensor, resource_device)
        for dfb_index, tensor in scratch_tensors.items()
    }
    owned_l1_buffer_addresses = {
        address
        for dfb_index, addresses_by_core in scratch_addresses_by_index.items()
        if dfb_index not in reusable_backing_tensors
        for address in addresses_by_core.values()
    }

    for boundary_ordinal in plan.boundary_ordinals:
        host_configuration = host_configurations[boundary_ordinal]
        for dfb_index, epochs in enumerate(plan.dfb_epochs):
            matching_epoch = next(
                (
                    epoch
                    for epoch in epochs
                    if epoch.entry_reconfiguration_ordinal == boundary_ordinal
                ),
                None,
            )
            if matching_epoch is None:
                continue
            config = matching_epoch.config
            allocation = _get_dfb_allocation(config)
            segments = config.storage_segments or (
                DFBStorageSegment(nodes=tuple(core_keys)),
            )
            records_by_core = {}
            for segment in segments:
                if segment.is_tensor_backed:
                    tensor_index = segment.tensor_index
                    assert tensor_index is not None
                    if tensor_index < 0 or tensor_index >= len(tensors):
                        raise ValueError(
                            f"DFB[{dfb_index}] references invalid tensor "
                            f"index {tensor_index}"
                        )
                    tensor = tensors[tensor_index]
                    if tensor_index not in tensor_addresses_by_core:
                        tensor_addresses_by_core[tensor_index] = (
                            _l1_buffer_addresses_by_core(tensor, resource_device)
                        )
                    addresses_by_core = tensor_addresses_by_core[tensor_index]
                else:
                    scratch_tensor = scratch_tensors.get(dfb_index)
                    if scratch_tensor is None:
                        raise ValueError(
                            f"DFB[{dfb_index}] configuration requires scratch storage"
                        )
                    addresses_by_core = scratch_addresses_by_index[dfb_index]
                for node in segment.nodes:
                    if node not in core_rows:
                        raise ValueError(
                            f"DFB[{dfb_index}] configuration references launch "
                            f"node {node} outside the kernel grid"
                        )
                    if node not in addresses_by_core:
                        raise RuntimeError(
                            f"DFB[{dfb_index}] storage has no L1 address for "
                            f"launch node {node}"
                        )
                    address = addresses_by_core[node] + int(segment.byte_offset)
                    records_by_core[node] = (
                        address,
                        allocation.total_size,
                        allocation.num_tiles * allocation.block_count,
                        allocation.page_size,
                    )

            mask_word = (
                _DFB_RECONFIGURATION_LOW_MASK_WORD
                if dfb_index < 32
                else _DFB_RECONFIGURATION_HIGH_MASK_WORD
            )
            mask_bit = 1 << (dfb_index % 32)
            configuration_offset = dfb_index * _DFB_RECONFIGURATION_WORDS_PER_DFB
            for core in core_keys:
                record = records_by_core.get(core)
                if record is None:
                    continue
                row = core_rows[core]
                host_configuration[row, mask_word] = (
                    int(host_configuration[row, mask_word]) | mask_bit
                )
                for record_offset, value in enumerate(record):
                    host_configuration[row, configuration_offset + record_offset] = (
                        value
                    )

        num_devices = (
            resource_device.get_num_devices()
            if hasattr(resource_device, "get_num_devices")
            else 1
        )
        mesh_mapper = (
            ttnn.ReplicateTensorToMesh(resource_device) if num_devices > 1 else None
        )
        shard_spec = ttnn.ShardSpec(
            core_ranges,
            (1, _DFB_RECONFIGURATION_WORDS_PER_CORE),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        configuration_tensor = ttnn.from_torch(
            host_configuration,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=resource_device,
            memory_config=memory_config,
            **({"mesh_mapper": mesh_mapper} if mesh_mapper is not None else {}),
        )
        configuration_tensors.append(configuration_tensor)
        configuration_addresses_by_core = _l1_buffer_addresses_by_core(
            configuration_tensor, resource_device
        )
        owned_l1_buffer_addresses.update(configuration_addresses_by_core.values())
        missing_configuration_cores = [
            core for core in core_keys if core not in configuration_addresses_by_core
        ]
        if missing_configuration_cores:
            raise RuntimeError(
                "DFB reconfiguration storage has no L1 address for launch "
                f"nodes {missing_configuration_cores}"
            )
        for core in core_keys:
            configuration_runtime_args[core].append(
                configuration_addresses_by_core[core]
            )

    return DFBReconfigurationRuntimeResources(
        scratch_tensors=scratch_tensors,
        configuration_tensors=configuration_tensors,
        configuration_runtime_args=configuration_runtime_args,
        device=resource_device,
        l1_buffer_addresses=frozenset(owned_l1_buffer_addresses),
    )


def build_pipe_sync_semaphore_descriptors(
    core_ranges: Any,
    count: int,
) -> List[Any]:
    """Build local semaphore descriptors referenced by PipeNet lowering."""
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


def combine_program_hash_with_runtime_resources(
    program_hash: Optional[int], structural_fingerprint: int
) -> Optional[int]:
    """Combine compiler and runtime-resource structure into one uint64 hash."""
    normalized_program_hash = normalize_program_hash(program_hash)
    if normalized_program_hash is None:
        return None
    return _digest_primitive_payload(
        (
            "operation-runtime-resource-program-hash",
            _RESOURCE_PLAN_VERSION,
            normalized_program_hash,
            structural_fingerprint,
        ),
        _RESOURCE_HASH_PERSONALIZATION,
    )


def _core_range_coordinates(core_ranges: Any, *, label: str) -> set[Tuple[int, int]]:
    """Expand a CoreRangeSet into logical ``(x, y)`` pairs."""
    if core_ranges is None:
        raise ValueError(f"{label} must be a CoreRangeSet")
    try:
        cores = ttnn.corerange_to_cores(core_ranges)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a CoreRangeSet") from exc

    coordinates = set()
    for core in cores:
        try:
            coordinates.add((int(core.x), int(core.y)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid core") from exc
    if not coordinates:
        raise ValueError(f"{label} must cover at least one core")
    return coordinates


def _used_dfb_indices_by_core(
    kernel_specs: Optional[List[KernelSpec]],
    program_core_ranges: Any,
    num_dfbs: int,
) -> Optional[Dict[Tuple[int, int], set[int]]]:
    """Union specialized kernel DFB use on each logical core."""
    if not kernel_specs or not any(
        spec.used_dfb_indices is not None for spec in kernel_specs
    ):
        return None

    program_cores = _core_range_coordinates(
        program_core_ranges, label="program core ranges"
    )
    used_by_core = {core: set() for core in program_cores}
    all_indices = set(range(num_dfbs))
    for spec_index, spec in enumerate(kernel_specs):
        spec_ranges = (
            spec.core_ranges if spec.core_ranges is not None else program_core_ranges
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
            if spec.used_dfb_indices is None
            else {int(index) for index in spec.used_dfb_indices}
        )
        invalid = sorted(index for index in indices if index < 0 or index >= num_dfbs)
        if invalid:
            raise ValueError(
                f"kernel spec {spec_index} uses DFB ids outside "
                f"[0, {num_dfbs}): {invalid}"
            )
        for core in spec_cores:
            used_by_core[core].update(indices)
    return used_by_core


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
    tensors: List[Any], cb_configs: Iterable[PhysicalDFBConfig]
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


def _specialized_dfb_placements(
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    backing_tensors: Dict[int, Any],
    kernel_specs: Optional[List[KernelSpec]],
) -> Optional[List[Dict[Tuple[int, int], Tuple[str, Optional[int]]]]]:
    """Resolve the storage source for each used ``(DFB, core)`` pair.

    A DFB with ``storage_segments`` must cover every core that still uses it.
    """
    used_by_core = _used_dfb_indices_by_core(kernel_specs, core_ranges, len(cb_configs))
    if used_by_core is None:
        return None

    program_cores = set(used_by_core)
    placements = []
    for dfb_index, config in enumerate(cb_configs):
        candidates: Dict[Tuple[int, int], Tuple[str, Optional[int]]] = {}
        if not config.storage_segments:
            storage_kind = "backing" if dfb_index in backing_tensors else "static"
            candidates = {core: (storage_kind, None) for core in program_cores}
        else:
            for segment_index, segment in enumerate(config.storage_segments):
                if segment.is_tensor_backed:
                    storage_kind = "tensor"
                elif dfb_index in backing_tensors:
                    storage_kind = "backing"
                else:
                    storage_kind = "static"
                source = (storage_kind, segment_index)
                for core in segment.nodes:
                    if core not in program_cores:
                        raise ValueError(
                            f"DFB[{dfb_index}] storage segment claims core "
                            f"{core} outside the program grid"
                        )
                    if core in candidates:
                        raise ValueError(
                            f"DFB[{dfb_index}] has overlapping storage segments "
                            f"on core {core}"
                        )
                    candidates[core] = source
        used_cores = {
            core for core, indices in used_by_core.items() if dfb_index in indices
        }
        if config.storage_segments:
            uncovered = sorted(core for core in used_cores if core not in candidates)
            if uncovered:
                raise ValueError(
                    f"DFB[{dfb_index}] is used on cores {uncovered} that are "
                    f"not covered by any storage segment"
                )
        placements.append(
            {core: candidates[core] for core in used_cores if core in candidates}
        )
    return placements


def _cb_format_descriptor(cb_index: int, allocation: _DFBAllocation) -> Any:
    tile_descriptor = (
        ttnn.TileDescriptor(ttnn.Tile(allocation.tile))
        if allocation.tile is not None
        else None
    )
    return ttnn.CBFormatDescriptor(
        buffer_index=cb_index,
        data_format=allocation.data_format,
        page_size=allocation.page_size,
        **({"tile": tile_descriptor} if tile_descriptor is not None else {}),
    )


def _build_specialized_dfb_descriptors(
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    allocations: List[_DFBAllocation],
    allocation_bytes: List[int],
    placements: List[Dict[Tuple[int, int], Tuple[str, Optional[int]]]],
    backing_tensors: Dict[int, Any],
    remaining_bytes_by_core: Dict[Tuple[int, int], int],
) -> List[Any]:
    """Build descriptors on disjoint core partitions after DFB-use filtering."""
    all_cores = sorted({core for placement in placements for core in placement})
    bytes_by_core = {core: 0 for core in all_cores}
    for dfb_index, placement in enumerate(placements):
        for core, (kind, _) in placement.items():
            if kind == "static":
                bytes_by_core[core] += allocation_bytes[dfb_index]

    overflow_cores = [
        core
        for core, static_bytes in bytes_by_core.items()
        if static_bytes > remaining_bytes_by_core[core]
    ]
    if overflow_cores:
        core = max(
            overflow_cores,
            key=lambda item: bytes_by_core[item] - remaining_bytes_by_core[item],
        )
        static_indices = [
            dfb_index
            for dfb_index, placement in enumerate(placements)
            if placement.get(core, (None, None))[0] == "static"
        ]
        breakdown = "\n".join(
            f"  DFB[{index}]: num_tiles={allocations[index].num_tiles} "
            f"block_count={allocations[index].block_count} "
            f"format={cb_configs[index].data_format} "
            f"tile={allocations[index].tile} -> "
            f"{allocation_bytes[index]} bytes"
            for index in sorted(static_indices)
        )
        raise ValueError(
            f"Per-core DFB allocation on core {core} ("
            f"{bytes_by_core[core]} bytes) exceeds L1 budget "
            f"({remaining_bytes_by_core[core]} bytes).\n"
            + breakdown
            + "\n  hint: reduce DFB shapes or block_count."
        )

    # Every descriptor uses one of these disjoint partitions. Equal ranges
    # share TT-Metal's allocation cursor; disjoint ranges cannot overlap.
    cores_by_signature: Dict[
        Tuple[Optional[Tuple[str, Optional[int]]], ...],
        set[Tuple[int, int]],
    ] = {}
    for core in all_cores:
        signature = tuple(placement.get(core) for placement in placements)
        cores_by_signature.setdefault(signature, set()).add(core)

    descriptors = []
    for signature, partition_cores in cores_by_signature.items():
        partition_ranges = _make_singleton_core_ranges(sorted(partition_cores))
        for dfb_index, source in enumerate(signature):
            if source is None:
                continue
            kind, segment_index = source
            allocation = allocations[dfb_index]
            cb_format = _cb_format_descriptor(dfb_index, allocation)
            if kind == "tensor":
                assert segment_index is not None
                segment = cb_configs[dfb_index].storage_segments[segment_index]
                tensor_index = segment.tensor_index
                assert tensor_index is not None
                descriptors.append(
                    ttnn.cb_descriptor_from_sharded_tensor(
                        dfb_index,
                        tensors[tensor_index],
                        address_offset=segment.byte_offset,
                        total_size=allocation.total_size,
                        core_ranges=partition_ranges,
                    )
                )
                continue

            descriptor = ttnn.CBDescriptor(
                total_size=allocation.total_size,
                core_ranges=partition_ranges,
                format_descriptors=[cb_format],
            )
            if kind == "backing":
                backing_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    dfb_index,
                    backing_tensors[dfb_index],
                    total_size=allocation.total_size,
                    core_ranges=partition_ranges,
                )
                descriptor.set_buffer_from_cb(backing_descriptor)
            descriptors.append(descriptor)
    return descriptors


def _validate_dfb_reconfiguration_plan(
    tensors: List[Any], plan: DFBReconfigurationPlan
) -> None:
    """Validate every configuration before allocating runtime resources."""
    boundary_ordinals = plan.boundary_ordinals
    if not boundary_ordinals:
        raise ValueError("DFB reconfiguration plan must contain a boundary")
    if len(set(boundary_ordinals)) != len(boundary_ordinals):
        raise ValueError("DFB reconfiguration boundary ordinals must be unique")

    configurations_by_entry = {None: {}}
    configurations_by_entry.update({ordinal: {} for ordinal in boundary_ordinals})
    for dfb_index, epochs in enumerate(plan.dfb_epochs):
        if not epochs:
            raise ValueError(
                f"DFB reconfiguration plan has no configurations for DFB[{dfb_index}]"
            )
        seen_entries = set()
        for epoch in epochs:
            entry_ordinal = epoch.entry_reconfiguration_ordinal
            if entry_ordinal in seen_entries:
                raise ValueError(
                    f"DFB[{dfb_index}] has duplicate reconfiguration epoch "
                    f"{entry_ordinal}"
                )
            if (
                entry_ordinal is not None
                and entry_ordinal not in configurations_by_entry
            ):
                raise ValueError(
                    f"DFB[{dfb_index}] configuration entry {entry_ordinal} "
                    "is not a reconfiguration boundary"
                )
            seen_entries.add(entry_ordinal)
            config = epoch.config
            if config.data_format in {"float16", "f16"}:
                raise ValueError(
                    "DFB reconfiguration does not support IEEE FP16 because "
                    "TTNN has no native FP16 tensor representation"
                )
            _get_dfb_allocation(config)
            _validate_physical_dfb_config(config, dfb_index)
            configurations_by_entry[entry_ordinal][dfb_index] = config

    current_tensor_configurations = {}

    def apply_configuration(config: PhysicalDFBConfig) -> None:
        if not config.storage_segments:
            for dfb_node in list(current_tensor_configurations):
                if dfb_node[0] == config.dfb_index:
                    del current_tensor_configurations[dfb_node]
            return
        for segment in config.storage_segments:
            for node in segment.nodes:
                dfb_node = (config.dfb_index, node)
                current_tensor_configurations.pop(dfb_node, None)
                if segment.is_tensor_backed:
                    node_segment = replace(segment, nodes=(node,))
                    current_tensor_configurations[dfb_node] = replace(
                        config, storage_segments=(node_segment,)
                    )

    for config in configurations_by_entry[None].values():
        apply_configuration(config)
    _validate_tensor_backing_aliases(tensors, current_tensor_configurations.values())
    for boundary_ordinal in boundary_ordinals:
        for config in configurations_by_entry[boundary_ordinal].values():
            apply_configuration(config)
        _validate_tensor_backing_aliases(
            tensors, current_tensor_configurations.values()
        )


def build_cb_descriptors(
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    pipe_computed_address_backing_tensors: Optional[Dict[int, Any]] = None,
    kernel_specs: Optional[List[KernelSpec]] = None,
    dfb_reconfiguration_scratch_tensors: Optional[Dict[int, Any]] = None,
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
        kernel_specs: Final per-kernel launch ranges and surviving DFB-use sets.
        dfb_reconfiguration_scratch_tensors: Maximum-capacity scratch storage
            retained across configuration epochs.

    Returns:
        List of ttnn.CBDescriptor objects. A configuration with storage
        segments produces one descriptor per segment.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    pipe_backing_tensors = dict(pipe_computed_address_backing_tensors or {})
    invalid_pipe_backing_indices = sorted(
        set(pipe_backing_tensors).difference(range(len(cb_configs)))
    )
    if invalid_pipe_backing_indices:
        raise ValueError(
            "computed-address backing tensors reference invalid DFB indices "
            f"{invalid_pipe_backing_indices}"
        )
    backing_tensors = dict(pipe_backing_tensors)
    for dfb_index, tensor in (dfb_reconfiguration_scratch_tensors or {}).items():
        if dfb_index < 0 or dfb_index >= len(cb_configs):
            raise ValueError(
                f"reconfiguration scratch references invalid DFB index {dfb_index}"
            )
        config = cb_configs[dfb_index]
        initial_uses_scratch = not config.storage_segments or any(
            not segment.is_tensor_backed for segment in config.storage_segments
        )
        if not initial_uses_scratch:
            continue
        existing = backing_tensors.get(dfb_index)
        if existing is not None and existing is not tensor:
            raise ValueError(f"DFB[{dfb_index}] has conflicting hidden backing tensors")
        backing_tensors[dfb_index] = tensor
    for dfb_index in pipe_backing_tensors:
        if cb_configs[dfb_index].storage_segments:
            raise ValueError(
                f"DFB[{dfb_index}] cannot combine PipeNet computed-address "
                "storage with finalized storage segments"
            )
    _validate_tensor_backing_aliases(tensors, cb_configs)

    device = None
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, "device"):
            device = tensor.device()
            if device is None:
                continue
            break
    allocation_quantum_bytes = _get_l1_allocation_quantum_bytes(device)

    # Compute sizes first so overflow is diagnosed before creating descriptors.
    allocations = []
    allocation_bytes = []
    static_cb_bytes = 0
    static_allocation_summaries = []
    for physical_index, config in enumerate(cb_configs):
        allocation = _get_dfb_allocation(config)
        _validate_physical_dfb_config(config, physical_index)
        aligned_bytes = _align_up(allocation.total_size, allocation_quantum_bytes)
        allocation_summary = (
            f"  DFB[{physical_index}]: num_tiles={allocation.num_tiles} "
            f"block_count={allocation.block_count} "
            f"format={config.data_format} tile={allocation.tile} -> "
            f"{aligned_bytes} bytes"
        )
        allocations.append(allocation)
        allocation_bytes.append(aligned_bytes)
        has_static_storage = not config.storage_segments or any(
            not segment.is_tensor_backed for segment in config.storage_segments
        )
        if physical_index not in backing_tensors and has_static_storage:
            static_cb_bytes += aligned_bytes
            static_allocation_summaries.append(allocation_summary)

    placements = _specialized_dfb_placements(
        cb_configs, core_ranges, backing_tensors, kernel_specs
    )
    if placements is not None:
        placement_cores = {
            core for placement in placements for core in placement.keys()
        }
        remaining_bytes_by_core = (
            _get_remaining_l1_by_core_for_device(device, placement_cores)
            if device is not None
            else {core: DEFAULT_L1_CB_BUDGET_BYTES for core in placement_cores}
        )
        return _build_specialized_dfb_descriptors(
            tensors,
            cb_configs,
            allocations,
            allocation_bytes,
            placements,
            backing_tensors,
            remaining_bytes_by_core,
        )

    remaining_bytes = (
        get_min_remaining_l1_for_device(device)
        if device is not None
        else DEFAULT_L1_CB_BUDGET_BYTES
    )

    # Must stay aligned with MLIR ttl-validate-cb-budget and the finalized DFB
    # page-size metadata. Computed-address backing tensors are allocated
    # separately before this check, so their L1 is already reflected in
    # remaining_bytes; counting them here would double-charge them.
    if static_cb_bytes > remaining_bytes:
        breakdown = "\n".join(static_allocation_summaries)
        raise ValueError(
            "Total DFB allocation ("
            f"{static_cb_bytes} bytes) exceeds L1 budget ({remaining_bytes} bytes). "
            "This checks static DFB backing store only (not all L1 on core).\n"
            + breakdown
            + "\n  hint: reduce DFB shapes or block_count."
        )

    cb_descriptors = []
    for cb_index, allocation in enumerate(allocations):
        config = cb_configs[cb_index]
        cb_format = _cb_format_descriptor(cb_index, allocation)
        if not config.storage_segments:
            descriptor = ttnn.CBDescriptor(
                total_size=allocation.total_size,
                core_ranges=core_ranges,
                format_descriptors=[cb_format],
            )
            backing_tensor = backing_tensors.get(cb_index)
            if backing_tensor is not None:
                backing_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    cb_index,
                    backing_tensor,
                    total_size=allocation.total_size,
                    core_ranges=core_ranges,
                )
                descriptor.set_buffer_from_cb(backing_descriptor)
            cb_descriptors.append(descriptor)
            continue

        for segment in config.storage_segments:
            segment_core_ranges = _make_singleton_core_ranges(segment.nodes)
            if not segment.is_tensor_backed:
                descriptor = ttnn.CBDescriptor(
                    total_size=allocation.total_size,
                    core_ranges=segment_core_ranges,
                    format_descriptors=[cb_format],
                )
                backing_tensor = backing_tensors.get(cb_index)
                if backing_tensor is not None:
                    backing_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        cb_index,
                        backing_tensor,
                        total_size=allocation.total_size,
                        core_ranges=segment_core_ranges,
                    )
                    descriptor.set_buffer_from_cb(backing_descriptor)
                cb_descriptors.append(descriptor)
                continue

            tensor_index = segment.tensor_index
            assert tensor_index is not None
            tensor = tensors[tensor_index]
            cb_descriptors.append(
                ttnn.cb_descriptor_from_sharded_tensor(
                    cb_index,
                    tensor,
                    address_offset=segment.byte_offset,
                    total_size=allocation.total_size,
                    core_ranges=segment_core_ranges,
                )
            )

    return cb_descriptors


def build_generic_op_io_tensors(
    tensors: List[Any],
    pipe_sram_scratch_tensors: List[Any],
    pipe_computed_address_dfb_tensors: Optional[Dict[int, Any]] = None,
    dfb_reconfiguration_scratch_tensors: Optional[Dict[int, Any]] = None,
    dfb_reconfiguration_configuration_tensors: Optional[List[Any]] = None,
) -> List[Any]:
    """Return io_tensors with the user-visible output in the final position."""
    if not tensors:
        raise ValueError("kernel must have at least one output tensor")

    computed_address_dfb_tensors = [
        pipe_computed_address_dfb_tensors[dfb_index]
        for dfb_index in sorted(pipe_computed_address_dfb_tensors or {})
    ]
    reconfiguration_scratch_tensors = [
        dfb_reconfiguration_scratch_tensors[dfb_index]
        for dfb_index in sorted(dfb_reconfiguration_scratch_tensors or {})
        if all(
            dfb_reconfiguration_scratch_tensors[dfb_index] is not tensor
            for tensor in computed_address_dfb_tensors
        )
    ]
    io_tensors = (
        list(pipe_sram_scratch_tensors)
        + computed_address_dfb_tensors
        + reconfiguration_scratch_tensors
        + list(dfb_reconfiguration_configuration_tensors or [])
        + list(tensors)
    )
    if len(io_tensors) < 2:
        io_tensors = [io_tensors[-1]] + io_tensors
    return io_tensors


def _run_kernel_on_device_impl(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
    program_hash: Optional[int] = None,
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    num_dfb_resets: int = 0,
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    operation_name: str = "<anonymous>",
    runtime_resource_cache: Optional[KernelRuntimeResourceCache] = None,
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
        dfb_reconfiguration_plan: Optional finalized configuration epochs.
        core_ranges: ttnn.CoreRangeSet for kernel execution.
        program_hash: Hash for tt-metal program cache.
        num_pipe_sync_semaphores: Number of pipe synchronization semaphores
            allocated by the compiler.
        pipe_sram_scratch_bytes: Per-core SRAM scratch bytes required by
            PipeNet metadata.
        num_pipe_global_semaphores: Number of GlobalSemaphore-backed PipeNet
            counters allocated by the compiler.
        num_dfb_resets: Number of synchronized DFB reset boundaries. A nonzero
            count requires zero-initialized compiler scratch state.
        runtime_resource_factory: Optional callback that returns declarative
            resources for the current invocation.
        operation_name: User-facing operation name for callback diagnostics.
        runtime_resource_cache: Optional cache owning persistent PipeNet, DFB
            reconfiguration, and declarative runtime resources.
        device: Optional explicit resource device. Defaults to the first input
            tensor's device.

    Returns:
        Result from ttnn.generic_op (typically None or output tensor).
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    if runtime_resource_cache is not None:
        _release_portable_runtime_resources_impl(runtime_resource_cache)

    semaphore_descriptors = build_pipe_sync_semaphore_descriptors(
        core_ranges=core_ranges,
        count=num_pipe_sync_semaphores,
    )
    if [descriptor.id for descriptor in semaphore_descriptors] != list(
        range(num_pipe_sync_semaphores)
    ):
        raise RuntimeError(
            "compiler-managed semaphore descriptors must use the dense ID range "
            f"[0, {num_pipe_sync_semaphores})"
        )

    resource_plan = None
    if runtime_resource_factory is not None:
        try:
            program_resources = runtime_resource_factory(
                tensors=tuple(tensors),
                core_ranges=core_ranges,
                first_free_semaphore_id=num_pipe_sync_semaphores,
            )
        except Exception as error:
            raise RuntimeError(
                f"@ttl.operation {operation_name!r}: runtime resource factory "
                f"failed: {error}"
            ) from error
        resource_plan = plan_program_runtime_resources(
            operation_name=operation_name,
            resources=program_resources,
            kernel_specs=kernel_specs,
            operation_core_ranges=core_ranges,
            first_free_semaphore_id=num_pipe_sync_semaphores,
        )

    # Build tensor accessor args.
    tensor_accessor_args = build_tensor_accessor_args(tensors)

    # Get grid dimensions from core_ranges.
    grid_size = core_ranges.bounding_box().grid_size()
    grid_cols = grid_size.x
    grid_rows = grid_size.y

    pipe_computed_address_dfb_indices = tuple(
        sorted(
            {
                dfb_index
                for spec in kernel_specs
                for dfb_index in spec.pipe_computed_address_dfb_indices
            }
        )
    )
    pipe_runtime_resources, reconfiguration_resources = get_cached_runtime_resources(
        runtime_resource_cache,
        tensors=tensors,
        core_ranges=core_ranges,
        cb_configs=cb_configs,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
        pipe_computed_address_dfb_indices=pipe_computed_address_dfb_indices,
        num_dfb_resets=num_dfb_resets,
        device=device,
        kernel_specs=kernel_specs,
        dfb_reconfiguration_plan=dfb_reconfiguration_plan,
    )

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
        descriptor_resource_plans=(
            resource_plan.kernel_descriptors if resource_plan is not None else None
        ),
        dfb_reconfiguration_runtime_args=(
            reconfiguration_resources.configuration_runtime_args
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
        kernel_specs=kernel_specs,
        dfb_reconfiguration_scratch_tensors=(reconfiguration_resources.scratch_tensors),
    )

    if resource_plan is not None:
        semaphore_descriptors.extend(resource_plan.semaphore_descriptors)

    # Build and execute program.
    program = ttnn.ProgramDescriptor(
        kernels=kernel_descriptors,
        cbs=cb_descriptors,
        semaphores=semaphore_descriptors,
    )
    normalized_program_hash = normalize_program_hash(program_hash)
    if resource_plan is not None:
        normalized_program_hash = combine_program_hash_with_runtime_resources(
            normalized_program_hash,
            resource_plan.structural_fingerprint,
        )
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
        pipe_computed_address_dfb_tensors=(
            pipe_runtime_resources.computed_address_dfb_tensors
        ),
        dfb_reconfiguration_scratch_tensors=(reconfiguration_resources.scratch_tensors),
        dfb_reconfiguration_configuration_tensors=(
            reconfiguration_resources.configuration_tensors
        ),
    )

    portable_resource_lifetimes = (
        resource_plan.lifetimes if resource_plan is not None else ()
    )
    if runtime_resource_cache is not None and portable_resource_lifetimes:
        runtime_resource_cache.portable_resource_lifetimes = portable_resource_lifetimes
        runtime_resource_cache.portable_resource_device = (
            device
            if device is not None
            else (
                runtime_resource_cache.device
                if runtime_resource_cache.device is not None
                else _first_device(tensors)
            )
        )

    uncached_portable_resource_lifetimes = (
        portable_resource_lifetimes if runtime_resource_cache is None else ()
    )
    owns_hidden_runtime_resources = bool(
        pipe_runtime_resources.scratch_tensors
        or pipe_runtime_resources.global_semaphores
        or pipe_runtime_resources.computed_address_dfb_tensors
        or reconfiguration_resources.scratch_tensors
        or reconfiguration_resources.configuration_tensors
    )
    synchronize_after_success = runtime_resource_cache is None and bool(
        owns_hidden_runtime_resources or uncached_portable_resource_lifetimes
    )
    synchronize_after_dispatch_error = bool(
        owns_hidden_runtime_resources or portable_resource_lifetimes
    )
    resource_device = None
    if synchronize_after_success:
        resource_device = reconfiguration_resources.device
        if resource_device is None:
            resource_device = device if device is not None else _first_device(tensors)
    try:
        result = ttnn.generic_op(io_tensors, program)
    except BaseException as dispatch_error:
        if synchronize_after_dispatch_error:
            try:
                if runtime_resource_cache is not None:
                    _invalidate_cached_runtime_resources_after_dispatch_error(
                        runtime_resource_cache
                    )
                else:
                    if resource_device is None:
                        resource_device = (
                            device if device is not None else _first_device(tensors)
                        )
                    _synchronize_or_retain_runtime_resources(
                        resource_device,
                        pipe_runtime_resources,
                        reconfiguration_resources,
                        portable_resource_lifetimes,
                    )
            except BaseException as synchronization_error:
                try:
                    dispatch_error.add_note(
                        "device synchronization also failed: "
                        f"{synchronization_error}"
                    )
                except BaseException:
                    pass
        raise
    if synchronize_after_success:
        _synchronize_or_retain_runtime_resources(
            resource_device,
            pipe_runtime_resources,
            reconfiguration_resources,
            uncached_portable_resource_lifetimes,
        )
    return result


def run_kernel_on_device(
    kernel_specs: List[KernelSpec],
    tensors: List[Any],
    cb_configs: List[PhysicalDFBConfig],
    core_ranges: Any,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
    program_hash: Optional[int] = None,
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    num_dfb_resets: int = 0,
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    operation_name: str = "<anonymous>",
    runtime_resource_cache: Optional[KernelRuntimeResourceCache] = None,
    device: Optional[Any] = None,
) -> Any:
    """Execute a kernel, serializing use of persistent runtime resources."""
    arguments = {
        "kernel_specs": kernel_specs,
        "tensors": tensors,
        "cb_configs": cb_configs,
        "core_ranges": core_ranges,
        "dfb_reconfiguration_plan": dfb_reconfiguration_plan,
        "program_hash": program_hash,
        "num_pipe_sync_semaphores": num_pipe_sync_semaphores,
        "pipe_sram_scratch_bytes": pipe_sram_scratch_bytes,
        "num_pipe_global_semaphores": num_pipe_global_semaphores,
        "num_dfb_resets": num_dfb_resets,
        "runtime_resource_factory": runtime_resource_factory,
        "operation_name": operation_name,
        "runtime_resource_cache": runtime_resource_cache,
        "device": device,
    }
    if runtime_resource_cache is None:
        return _run_kernel_on_device_impl(**arguments)

    requires_persistent_resources = bool(
        pipe_sram_scratch_bytes > 0
        or num_pipe_global_semaphores > 0
        or num_dfb_resets > 0
        or dfb_reconfiguration_plan is not None
        or runtime_resource_factory is not None
        or any(spec.pipe_computed_address_dfb_indices for spec in kernel_specs)
    )
    if not requires_persistent_resources:
        with runtime_resource_cache.lock:
            _release_cached_runtime_resources_impl(runtime_resource_cache)
        arguments["runtime_resource_cache"] = None
        return _run_kernel_on_device_impl(**arguments)

    with runtime_resource_cache.lock:
        return _run_kernel_on_device_impl(**arguments)


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


def _serialize_logical_kernel(
    spec: KernelSpec,
) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    if spec.logical_kernel is None:
        return None
    logical_kernel = _normalize_logical_kernel_selector(
        spec.logical_kernel,
        operation_name="<emitted runner>",
        source=f"kernel descriptor {spec.path!r}",
    )
    return (
        logical_kernel.kind.value,
        logical_kernel.name,
        logical_kernel.operation,
        logical_kernel.implicit_role,
    )


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


def _append_physical_dfb_config_source(
    lines: List[str],
    config: PhysicalDFBConfig,
    *,
    indent: str,
    closing_suffix: str = "",
) -> None:
    """Append one reconstructible physical DFB configuration."""
    lines.append(f"{indent}PhysicalDFBConfig(")
    lines.append(f"{indent}    dfb_index={config.dfb_index},")
    lines.append(f"{indent}    num_tiles={config.num_tiles},")
    lines.append(f"{indent}    data_format={config.data_format!r},")
    lines.append(f"{indent}    block_count={config.block_count},")
    lines.append(f"{indent}    page_size={config.page_size},")
    lines.append(f"{indent}    tile={config.tile!r},")
    if config.storage_segments:
        lines.append(f"{indent}    storage_segments=(")
        for segment in config.storage_segments:
            lines.append(f"{indent}        DFBStorageSegment(")
            lines.append(f"{indent}            nodes={segment.nodes!r},")
            lines.append(f"{indent}            tensor_index={segment.tensor_index!r},")
            lines.append(f"{indent}            byte_offset={segment.byte_offset},")
            lines.append(f"{indent}            byte_size={segment.byte_size!r},")
            lines.append(f"{indent}        ),")
        lines.append(f"{indent}    ),")
    lines.append(f"{indent}){closing_suffix}")


def _append_dfb_reconfiguration_plan_source(
    lines: List[str], plan: Optional[DFBReconfigurationPlan]
) -> None:
    """Append the finalized reconfiguration plan used by emitted runners."""
    if plan is None:
        lines.append("DFB_RECONFIGURATION_PLAN = None")
        return

    lines.append("DFB_RECONFIGURATION_PLAN = DFBReconfigurationPlan(")
    lines.append(f"    boundary_ordinals={plan.boundary_ordinals!r},")
    lines.append("    dfb_epochs=(")
    for epochs in plan.dfb_epochs:
        lines.append("        (")
        for epoch in epochs:
            lines.append("            DFBConfigurationEpoch(")
            lines.append(
                "                entry_reconfiguration_ordinal="
                f"{epoch.entry_reconfiguration_ordinal!r},"
            )
            lines.append("                config=")
            _append_physical_dfb_config_source(
                lines,
                epoch.config,
                indent="                ",
                closing_suffix=",",
            )
            lines.append("            ),")
        lines.append("        ),")
    lines.append("    ),")
    lines.append(")")


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
    num_dfb_resets: int = 0,
    program_hash: Optional[int] = None,
    requires_runtime_resource_factory: bool = False,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
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
    lines.append("from ttl.dataflow_buffer import PhysicalDFBConfig")
    lines.append("from ttl.dataflow_buffer import DFBStorageSegment")
    lines.append("from ttl.dataflow_buffer import DFBConfigurationEpoch")
    lines.append("from ttl.dataflow_buffer import DFBReconfigurationPlan")
    lines.append("from ttl.kernel import Kernel, KernelKind")
    lines.append("from ttl.kernel_runner import (")
    lines.append("    KernelRuntimeResourceCache,")
    lines.append("    KernelSpec,")
    lines.append("    attach_runtime_resource_finalizer,")
    lines.append("    run_kernel_on_device,")
    lines.append(")")
    lines.append("")

    lines.append(f"GRID_COLS = {grid_cols}")
    lines.append(f"GRID_ROWS = {grid_rows}")
    lines.append(f"NUM_TENSORS = {num_tensors}")
    lines.append(f"OPERATION_NAME = {kernel_name!r}")
    lines.append(f"PROGRAM_HASH = {normalize_program_hash(program_hash)!r}")
    lines.append(f"NUM_PIPE_SYNC_SEMAPHORES = {num_pipe_sync_semaphores}")
    lines.append(f"NUM_DFB_RESETS = {num_dfb_resets}")
    lines.append(f"PIPE_SRAM_SCRATCH_BYTES = {pipe_sram_scratch_bytes}")
    lines.append(f"NUM_PIPE_GLOBAL_SEMAPHORES = {num_pipe_global_semaphores}")
    lines.append("class _RuntimeResourceOwner:")
    lines.append("    pass")
    lines.append("")
    lines.append("_RUNTIME_RESOURCE_OWNER = _RuntimeResourceOwner()")
    lines.append("_RUNTIME_RESOURCE_CACHE = KernelRuntimeResourceCache()")
    lines.append("_RUNTIME_RESOURCE_FINALIZER = attach_runtime_resource_finalizer(")
    lines.append("    _RUNTIME_RESOURCE_OWNER, _RUNTIME_RESOURCE_CACHE")
    lines.append(")")
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

    lines.append("KERNEL_LOGICAL_IDENTITIES = [")
    for spec in kernel_specs:
        lines.append(f"    {_serialize_logical_kernel(spec)!r},")
    lines.append("]")
    lines.append("")

    lines.append("KERNEL_USED_DFB_INDICES = [")
    for spec in kernel_specs:
        lines.append(f"    {spec.used_dfb_indices!r},  # {spec.thread_type}")
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
    for physical_index, config in enumerate(cb_configs):
        _get_dfb_allocation(config)
        _validate_physical_dfb_config(config, physical_index)
        _append_physical_dfb_config_source(
            lines,
            config,
            indent="    ",
            closing_suffix=f",  # DFB {physical_index}",
        )
    lines.append("]")
    lines.append("")
    _append_dfb_reconfiguration_plan_source(lines, dfb_reconfiguration_plan)
    lines.append("")

    lines.append("")
    if requires_runtime_resource_factory:
        lines.append("def run(tensors, *, runtime_resource_factory, device=None):")
    else:
        lines.append("def run(tensors, device=None):")
    lines.append(f'    """Run the {kernel_name} on device."""')
    lines.append(
        f"    assert len(tensors) == {num_tensors}, f'Expected {num_tensors} tensors, got {{len(tensors)}}'"
    )
    if requires_runtime_resource_factory:
        lines.append("    if runtime_resource_factory is None:")
        lines.append(
            '        raise TypeError(f"emitted runner for {OPERATION_NAME!r} '
            'requires runtime_resource_factory")'
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
    lines.append("    def _logical_kernel_from_spec(identity_spec):")
    lines.append("        if identity_spec is None:")
    lines.append("            return None")
    lines.append("        kind, name, operation, implicit_role = identity_spec")
    lines.append("        kind = KernelKind(kind)")
    lines.append("        if name is None:")
    lines.append("            return kind")
    lines.append("        return Kernel._from_metadata(")
    lines.append("            kind, name, operation, implicit_role")
    lines.append("        )")
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
        "                logical_kernel=_logical_kernel_from_spec("
        "KERNEL_LOGICAL_IDENTITIES[kernel_idx]),"
    )
    lines.append(
        "                used_dfb_indices=KERNEL_USED_DFB_INDICES[kernel_idx],"
    )
    lines.append("            )")
    lines.append("        )")
    lines.append("    return run_kernel_on_device(")
    lines.append("        kernel_specs=kernel_specs,")
    lines.append("        tensors=tensors,")
    lines.append("        cb_configs=CB_CONFIGS,")
    lines.append("        dfb_reconfiguration_plan=DFB_RECONFIGURATION_PLAN,")
    lines.append("        core_ranges=core_ranges,")
    lines.append("        program_hash=PROGRAM_HASH,")
    lines.append("        num_pipe_sync_semaphores=NUM_PIPE_SYNC_SEMAPHORES,")
    lines.append("        num_dfb_resets=NUM_DFB_RESETS,")
    lines.append("        pipe_sram_scratch_bytes=PIPE_SRAM_SCRATCH_BYTES,")
    lines.append("        num_pipe_global_semaphores=NUM_PIPE_GLOBAL_SEMAPHORES,")
    if requires_runtime_resource_factory:
        lines.append("        runtime_resource_factory=runtime_resource_factory,")
    lines.append("        operation_name=OPERATION_NAME,")
    lines.append("        runtime_resource_cache=_RUNTIME_RESOURCE_CACHE,")
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
    num_dfb_resets: int = 0,
    program_hash: Optional[int] = None,
    requires_runtime_resource_factory: bool = False,
    dfb_reconfiguration_plan: Optional[DFBReconfigurationPlan] = None,
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
        num_dfb_resets=num_dfb_resets,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
        requires_runtime_resource_factory=requires_runtime_resource_factory,
        dfb_reconfiguration_plan=dfb_reconfiguration_plan,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(source)

    print(f"Runner written to {output_path}")
    return output_path


__all__ = [
    "KernelSpec",
    "LogicalKernelId",
    "ProgramResourcePlan",
    "PipeRuntimeResources",
    "KernelRuntimeResourceCache",
    "DFBReconfigurationRuntimeResources",
    "build_tensor_accessor_args",
    "build_kernel_descriptors",
    "build_cb_descriptors",
    "build_pipe_sram_scratch_tensors",
    "build_pipe_global_semaphores",
    "build_pipe_computed_address_dfb_tensors",
    "build_pipe_runtime_resources",
    "get_cached_runtime_resources",
    "build_dfb_reconfiguration_runtime_resources",
    "build_pipe_sync_semaphore_descriptors",
    "normalize_program_hash",
    "combine_program_hash_with_runtime_resources",
    "build_generic_op_io_tensors",
    "plan_program_runtime_resources",
    "attach_runtime_resource_finalizer",
    "release_cached_runtime_resources",
    "run_kernel_on_device",
    "emit_runner_source",
    "emit_runner_file",
]
