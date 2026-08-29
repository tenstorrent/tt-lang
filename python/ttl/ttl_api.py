# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Main API for the TTL dialect Python DSL."""

from __future__ import annotations

import ast
import functools
import inspect
import os
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

ttnn = None  # Lazy-loaded on first access via _ensure_ttnn()


def _forward_mlir_warning(diagnostic):
    """Print MLIR warnings while preserving the existing error handler."""
    if diagnostic.severity != DiagnosticSeverity.WARNING:
        return False
    print(f"warning: {diagnostic}", file=sys.stderr)
    return True


def _ensure_ttnn():
    """Lazy import of ttnn to avoid triggering heavy dependencies at module load.

    Returns the ttnn module or None if unavailable. Caches the result in the
    module-level ``ttnn`` variable so existing ``ttnn.Foo`` call sites work
    unchanged after a single ``ttnn = _ensure_ttnn()`` call.
    """
    global ttnn
    if ttnn is not None:
        return ttnn
    try:
        import ttnn as _ttnn

        ttnn = _ttnn
    except (ModuleNotFoundError, ImportError):
        pass
    return ttnn


import ttl._mlir_libs._ttlang  # Register tt-lang passes
from ttl._mlir_libs._ttlang import ttl_ir as _ttl_ir
from ttl.pykernel._src.utils import _cleanup_source_code
from ttl.dialects import ttcore, ttkernel, ttl as ttl_dialect
from ttl.ir import *
from ttl.ir import DenseI32ArrayAttr
from ttl.passes import (
    get_ttkernel_arg_spec,
    get_ttkernel_names,
    ttkernel_to_cpp_by_name,
)

from ttl.passmanager import PassManager


from ._src.auto_profile import (
    build_cb_wait_to_dma_map,
    build_dma_producer_to_cb_map,
    get_line_mapper,
    is_auto_profile_enabled,
    load_cb_flow_graph,
    parse_device_profile_csv,
    print_profile_report,
)
from ._src.perf_trace_server import serve_trace
from ._src.signpost_profile import is_signpost_profile_enabled
from ._src.tensor_registry import (
    get_tensor_global_index,
    get_tensor_source,
    register_tensor_name,
    register_tensor_source,
)
from ._src.global_semaphore import is_ttnn_global_semaphore
from ._src.ttl_ast import TTLGenericCompiler
from .dataflow_buffer import (
    CircularBuffer,
    DataflowBuffer,
    DFBConfigurationEpoch,
    DFBReconfigurationPlan,
    DFBStorageSegment,
    PhysicalDFBConfig,
    get_cb_count,
)
from .pipe import Pipe, PipeNet
from .scalar import ScalarType
from .condition import (
    _BoundDispatchCondition,
    DispatchCondition,
    _bind_current_dispatch_condition,
    _dispatch_condition_binding_scope,
)
from .dfb_reset import (
    DFBReset,
    _BoundDFBReset,
    _bind_current_dfb_reset,
    _dfb_reset_binding_scope,
)
from .dfb_allocation_group import _dfb_allocation_group_binding_scope
from .dfb_reconfiguration import (
    DFBReconfiguration,
    _BoundDFBReconfiguration,
    _bind_current_dfb_reconfiguration,
    _dfb_reconfiguration_binding_scope,
)
from .constants import SUPPORTED_MEMORY_SPACES, validate_math_fidelity
from .diagnostics import (
    TTLangCompileError,
    find_variable_assignment,
    format_mlir_error,
    format_python_error,
)
from .dtype_utils import (
    is_ttnn_tensor,
    torch_dtype_to_ttnn_datatype,
)
from .kernel_runner import (
    _detect_device_arch,
    _same_device,
    attach_runtime_resource_finalizer,
    KernelRuntimeResourceCache,
    KernelSpec,
    get_min_remaining_l1_excluding_cached_resources,
    get_min_remaining_l1_for_device,
    run_kernel_on_device,
    emit_runner_file,
)
from .kernel import (
    Kernel,
    KernelKind,
    KernelSelector,
    _PIPE_SOURCE_KERNEL_ROLE,
    _bind_kernel_declarations,
    _format_kernel_capacity_error,
    _operation_identity,
    _selector_implicit_role,
    _selector_kind,
)
from .runtime_resources import ProgramRuntimeResources
from .operators import (
    CopyTransferHandler,
    ReadyReceive,
    ReceiveRequest,
    TensorBlock,
    copy,
    wait_any,
)
from .compiler_options import CompilerOptions
from .ttl_utils import get_thread_type_string

_TTCORE_ARCH_BY_DEVICE_NAME = {
    "blackhole": ttcore.Arch.Blackhole,
    "wormhole_b0": ttcore.Arch.WormholeB0,
}


@dataclass(frozen=True)
class _BackendKernelSlot:
    kind: KernelKind
    kernel_type: str
    source_name: str
    implicit_role: Optional[str] = None


_COMMON_BACKEND_KERNEL_SLOTS = (
    _BackendKernelSlot(KernelKind.COMPUTE, "compute", "trisc"),
    _BackendKernelSlot(KernelKind.DATA_MOVEMENT, "datamovement", "ncrisc"),
    _BackendKernelSlot(
        KernelKind.DATA_MOVEMENT,
        "datamovement",
        "brisc",
        implicit_role=_PIPE_SOURCE_KERNEL_ROLE,
    ),
)
_BACKEND_KERNEL_SLOTS_BY_ARCH = {
    target_arch: _COMMON_BACKEND_KERNEL_SLOTS
    for target_arch in _TTCORE_ARCH_BY_DEVICE_NAME
}


def _backend_kernel_slots(
    target_arch: Optional[str] = None,
) -> tuple[_BackendKernelSlot, ...]:
    """Return the processor slots declared by the selected backend target."""
    if target_arch is None:
        return _COMMON_BACKEND_KERNEL_SLOTS
    try:
        return _BACKEND_KERNEL_SLOTS_BY_ARCH[target_arch]
    except KeyError:
        raise ValueError(f"unsupported target architecture {target_arch!r}") from None


def _backend_kernel_capacities(
    target_arch: Optional[str] = None,
) -> Mapping[KernelKind, int]:
    slots = _backend_kernel_slots(target_arch)
    return {kind: sum(slot.kind == kind for slot in slots) for kind in KernelKind}


def _slot_idle_kernel(slot: _BackendKernelSlot) -> KernelSelector:
    """Return the logical identity a slot carries when it holds no work.

    A slot the target reserves for a compiler-owned affinity keeps that role. Any
    other slot is the canonical kernel of its kind, which is unoccupied precisely
    when no selector of that kind was planned.
    """
    if slot.implicit_role is None:
        return slot.kind
    return Kernel._implicit(slot.kind, slot.implicit_role)


# Thread registry for automatic collection of @compute and @datamovement threads
_thread_registry: List[Callable] = []


def _register_thread(thread_fn: Callable) -> None:
    """Register a thread function during decoration."""
    _thread_registry.append(thread_fn)


def _clear_thread_registry() -> None:
    """Clear the thread registry before kernel execution."""
    _thread_registry.clear()


def _get_registered_threads() -> List[Callable]:
    """Get all registered threads and clear the registry."""
    threads = list(_thread_registry)
    _thread_registry.clear()
    return threads


def _validate_explicit_logical_kernel_uses(
    threads: List[Callable],
    kernel_capacities: Optional[Mapping[KernelKind, int]] = None,
) -> None:
    """Require each named logical kernel to identify one explicit thread."""
    thread_by_kernel: Dict[Kernel, Callable] = {}
    for thread in threads:
        logical_kernel = thread._logical_kernel
        if not isinstance(logical_kernel, Kernel):
            continue
        previous_thread = thread_by_kernel.get(logical_kernel)
        if previous_thread is not None:
            raise ValueError(
                f"logical Kernel {logical_kernel.identity!r} is selected by "
                "multiple explicit threads: "
                f"{previous_thread.__name__!r} and {thread.__name__!r}"
            )
        thread_by_kernel[logical_kernel] = thread

    if kernel_capacities is None:
        return
    selectors = tuple(thread._logical_kernel for thread in threads)
    for kind in KernelKind:
        capacity = kernel_capacities[kind]
        selected = tuple(
            selector for selector in selectors if _selector_kind(selector) == kind
        )
        if len(selected) > capacity:
            raise ValueError(_format_kernel_capacity_error(kind, selected, capacity))


def _captured_kernel_declarations(function: Callable) -> Dict[str, Kernel]:
    """Return logical kernels referenced by an explicit operation."""
    closure = inspect.getclosurevars(function)
    captures = dict(closure.globals)
    captures.update(closure.nonlocals)
    return {
        name: value
        for name, value in sorted(captures.items())
        if isinstance(value, Kernel)
    }


def _get_tensor_cache_info(tensor) -> tuple:
    """Extract tensor properties that affect compilation or DFB descriptors."""
    shape = tuple(tensor.shape)
    padded_shape = tuple(getattr(tensor, "padded_shape", tensor.shape))
    dtype = str(tensor.dtype)
    mem_config = tensor.memory_config()
    memory_space = (
        str(mem_config.buffer_type) if hasattr(mem_config, "buffer_type") else "unknown"
    )
    memory_layout = (
        str(mem_config.memory_layout)
        if hasattr(mem_config, "memory_layout")
        else "unknown"
    )
    layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
    tile = (
        tuple(tensor.get_tile().tile_shape)
        if "TILE" in layout and hasattr(tensor, "get_tile")
        else None
    )
    return (shape, padded_shape, dtype, memory_space, memory_layout, layout, tile)


def _make_cache_key(
    args: tuple,
    resolved_grid: Union[tuple, List[int]],
    fp32_dest_acc_en: Optional[bool],
    dst_full_sync_en: Optional[bool],
    math_fidelity: Optional[str],
    target_arch: Optional[str],
    compiler_options: CompilerOptions = CompilerOptions(),
    l1_budget_override: Any = 0,
) -> tuple:
    """Create cache key from tensor properties and runtime compute config parameters."""
    grid_key = tuple(resolved_grid)
    tensor_args = [arg for arg in args if is_ttnn_tensor(arg)]
    tensor_key = tuple(_get_tensor_cache_info(tensor) for tensor in tensor_args)
    first_position_by_identity = {}
    alias_partition = []
    for position, tensor in enumerate(tensor_args):
        identity = id(tensor)
        first_position_by_identity.setdefault(identity, position)
        alias_partition.append(first_position_by_identity[identity])
    # Include mesh shape so that single-device and multi-device compilations
    # with different shard shapes don't collide in the cache.
    mesh_key = None
    for tensor in tensor_args:
        if _is_mesh_tensor(tensor):
            mesh_key = tuple(tensor.device().shape)
            break
    return (
        tensor_key,
        tuple(alias_partition),
        mesh_key,
        grid_key,
        fp32_dest_acc_en,
        dst_full_sync_en,
        math_fidelity,
        target_arch,
        compiler_options,
        l1_budget_override,
    )


def _should_execute() -> bool:
    """Check if kernel execution should proceed (not compile-only mode)."""
    return os.environ.get("TTLANG_COMPILE_ONLY", "0") != "1"


def _run_profiling_pipeline(
    tensors: tuple,
    all_source_lines: Dict[str, List[str]],
    thread_to_kernel: Dict[str, str],
    kernel_line_offsets: Optional[Dict[str, int]] = None,
):
    """
    Read device profiler data and display profile report.

    Called after kernel execution when auto-profiling is enabled.

    Args:
        tensors: Tuple of tensor arguments passed to the kernel
        all_source_lines: Dict mapping kernel name to source lines
        thread_to_kernel: Dict mapping RISC thread name to kernel name
    """
    if not is_auto_profile_enabled():
        return

    _ensure_ttnn()
    if ttnn is None:
        print("[Auto-profile] ttnn not available, skipping profiling")
        return

    from pathlib import Path

    # Get device from first ttnn tensor
    device = None
    for tensor in tensors:
        if is_ttnn_tensor(tensor) and hasattr(tensor, "device"):
            device = tensor.device()
            break

    if device is None:
        print("[Auto-profile] No device found in tensors, skipping profiling")
        return

    # Read profiler data from device
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception as e:
        print(f"[Auto-profile] Failed to read device profiler: {e}")
        return

    # Find the profile CSV - default location is $TT_METAL_HOME/generated/profiler/.logs/
    if "TTLANG_PROFILE_CSV" in os.environ:
        csv_path = Path(os.environ["TTLANG_PROFILE_CSV"])
    else:
        tt_metal_home = os.environ.get("TT_METAL_HOME", "")
        if not tt_metal_home:
            print("[Auto-profile] TT_METAL_HOME not set, cannot find profile CSV")
            return
        csv_path = (
            Path(tt_metal_home) / "generated/profiler/.logs/profile_log_device.csv"
        )

    if not csv_path.exists():
        print(f"[Auto-profile] Profile CSV not found at {csv_path}")
        print("[Auto-profile] Ensure TT_METAL_DEVICE_PROFILER=1 is set before running")
        return

    # Parse and display results
    line_mapper = get_line_mapper()

    # Load CB flow graph for DMA attribution
    cb_flow = load_cb_flow_graph(csv_path)
    cb_wait_to_dma = build_cb_wait_to_dma_map(cb_flow)
    dma_producer_to_cb = build_dma_producer_to_cb_map(cb_flow)

    try:
        results = parse_device_profile_csv(csv_path, line_mapper)
        if results:
            print_profile_report(
                results,
                all_source_lines,
                thread_to_kernel,
                line_mapper,
                cb_wait_to_dma,
                dma_producer_to_cb,
                kernel_line_offsets,
            )
        else:
            print("[Auto-profile] No signpost results found in profile CSV")
    except Exception as e:
        print(f"[Auto-profile] Failed to parse profile CSV: {e}")


def _run_perf_dump(tensors: tuple, kernel_name: str):
    """
    Run NOC profiler summary and print CB flow / pipe graph after execution.

    Called after kernel execution when TTLANG_PERF_DUMP=1 is set.
    Reads NOC traces from $TT_METAL_HOME/generated/profiler/.logs/ and
    CB flow graph from /tmp/ttlang_cb_flow_graph.json (written by
    ttl-dump-cb-flow-graph pass).
    """
    _ensure_ttnn()
    from ._src.perf_summary import run as perf_summary_run

    # Flush profiler data from device (requires mid-run dump)
    if os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") != "1":
        print(
            "[perf_dump] WARNING: TT_METAL_PROFILER_MID_RUN_DUMP=1 not set, "
            "profiler data may be stale"
        )
    device = None
    for tensor in tensors:
        if is_ttnn_tensor(tensor) and hasattr(tensor, "device"):
            device = tensor.device()
            break
    if device is not None:
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception as e:
            print(f"[perf_dump] WARNING: Failed to read device profiler: {e}")

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    if not tt_metal_home:
        raise ValueError("TTLANG_PERF_DUMP=1 requires TT_METAL_HOME to be set")

    # NOC profiler summary
    logs_path = Path(tt_metal_home) / "generated" / "profiler" / ".logs"
    if not logs_path.exists():
        raise ValueError(
            f"Profiler logs directory not found: {logs_path}\n"
            "Ensure TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 is set"
        )
    result = perf_summary_run(logs_path, names=[kernel_name])
    if result:
        print(result)

    # CB flow graph (written by ttl-dump-cb-flow-graph pass)
    cb_flow_path = Path("/tmp/ttlang_cb_flow_graph.json")
    if not cb_flow_path.exists():
        raise ValueError(f"CB flow graph not found: {cb_flow_path}")
    print("=== CB FLOW GRAPH ===")
    print(cb_flow_path.read_text())


def _run_signpost_profile(tensors: tuple):
    """
    Run user-defined signpost profiler after execution.

    Called after kernel execution when TTLANG_SIGNPOST_PROFILE=1 is set.
    """
    from ._src.signpost_profile import run as signpost_profile_run

    # Flush profiler data from device (requires mid-run dump)
    if os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") != "1":
        print(
            "[signpost_profile] WARNING: TT_METAL_PROFILER_MID_RUN_DUMP=1 not set, "
            "profiler data may be stale"
        )
    device = None
    for tensor in tensors:
        if is_ttnn_tensor(tensor) and hasattr(tensor, "device"):
            device = tensor.device()
            break
    if device is not None:
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception as e:
            print(f"[signpost_profile] WARNING: Failed to read device profiler: {e}")

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    if not tt_metal_home:
        raise ValueError("TTLANG_SIGNPOST_PROFILE=1 requires TT_METAL_HOME to be set")

    logs_path = Path(tt_metal_home) / "generated" / "profiler" / ".logs"
    if not logs_path.exists():
        raise ValueError(
            f"Profiler logs directory not found: {logs_path}\n"
            "Ensure TT_METAL_DEVICE_PROFILER=1 is set"
        )

    result = signpost_profile_run(logs_path)
    if result:
        print(result)
    else:
        print("[signpost_profile] No user-defined signpost zones found")


def _is_mesh_tensor(tensor) -> bool:
    """Check if a ttnn tensor is distributed across a multi-device mesh."""
    if not is_ttnn_tensor(tensor):
        return False
    device = tensor.device()
    if device is None:
        return False
    shape = getattr(device, "shape", None)
    if shape is None:
        return False
    from math import prod

    return prod(shape) > 1


def _detect_memory_space_from_tensor(tensor, default: str) -> str:
    """Detect memory space (L1/DRAM) from a ttnn tensor's buffer type."""
    mem_config = tensor.memory_config()
    if hasattr(mem_config, "buffer_type"):
        buffer_type_str = str(mem_config.buffer_type)
        if "L1" in buffer_type_str:
            return "L1"
        elif "DRAM" in buffer_type_str:
            return "DRAM"
    return default


def _require_device(args):
    """Extract the device from tensor arguments, raising if none are on-device.

    Returns the first non-None device found after verifying that every
    on-device tensor shares the same device.  Raises ValueError when no
    tensor carries a device, or when tensors are on different devices.
    """
    first_device = None
    first_idx = None
    for i, arg in enumerate(args):
        if not is_ttnn_tensor(arg):
            continue
        device = arg.device()
        if device is None:
            continue
        if first_device is None:
            first_device = device
            first_idx = i
        elif not _same_device(first_device, device):
            raise ValueError(
                f"Tensor arguments are on different devices: "
                f"arg[{first_idx}] is on device {first_device}, "
                f"but arg[{i}] is on device {device}. "
                f"All on-device tensors must reside on the same device."
            )
    if first_device is not None:
        return first_device
    host_args = [
        f"  arg[{i}]: {arg.shape}" for i, arg in enumerate(args) if is_ttnn_tensor(arg)
    ]
    if not host_args:
        raise ValueError("No device found: no ttnn tensor arguments were provided.")
    raise ValueError(
        "No device found on any tensor argument. "
        "All ttnn tensor inputs are on host:\n"
        + "\n".join(host_args)
        + "\nPlace tensors on device before calling the operation, e.g.:\n"
        "  ttnn.to_device(tensor, device)\n"
        "  ttnn.from_torch(tensor, ..., device=device)"
    )


def _resolve_l1_budget(
    args: tuple,
    compiler_options: CompilerOptions,
    runtime_resource_cache: Optional[KernelRuntimeResourceCache] = None,
) -> int:
    """Return the explicit or device-derived L1 compilation budget."""
    if compiler_options.l1_budget != 0:
        return compiler_options.l1_budget
    if not any(is_ttnn_tensor(arg) for arg in args):
        return 0
    try:
        device = _require_device(args)
        if runtime_resource_cache is not None:
            return get_min_remaining_l1_excluding_cached_resources(
                runtime_resource_cache, device
            )
        return get_min_remaining_l1_for_device(device)
    except ValueError:
        return 0


def _device_target_arch(args) -> Optional[str]:
    """Return the common tensor device architecture, or None for host inputs."""
    target_arch = None
    for arg in args:
        if not is_ttnn_tensor(arg):
            continue
        try:
            device = arg.device()
        except Exception as error:
            raise ValueError(
                "Unsupported or undetectable TT device architecture"
            ) from error
        if device is None:
            continue
        arch = _detect_device_arch(device)
        if arch is None:
            raise ValueError("Unsupported or undetectable TT device architecture")
        if arch not in _TTCORE_ARCH_BY_DEVICE_NAME:
            raise ValueError(f"Unsupported TT device architecture: {arch}")
        if target_arch is None:
            target_arch = arch
        elif target_arch != arch:
            raise ValueError(
                "Tensor arguments use different TT device architectures: "
                f"{target_arch} and {arch}"
            )
    return target_arch


def _resolve_grid(grid, args, kwargs):
    """Resolve the compile-time grid: callable is evaluated; both "auto"
    and "full" expand to the device compute grid."""
    if callable(grid):
        return grid(*args, **kwargs)
    if grid in ("auto", "full"):
        device = _require_device(args)
        device_grid = device.compute_with_storage_grid_size()
        return (device_grid.x, device_grid.y)
    return grid


def _get_source_line_offset(f) -> int:
    """Get the line offset to convert parsed AST line numbers to actual file lines."""
    try:
        raw_lines, start_lineno = inspect.getsourcelines(f)
        # Count only leading decorator lines (before the def)
        num_decorator_lines = 0
        for line in raw_lines:
            stripped = line.strip()
            if stripped.startswith("@"):
                num_decorator_lines += 1
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                break
        return start_lineno + num_decorator_lines - 1
    except (TypeError, OSError):
        return 0


def _track_tensor_sources(f_params, args, source_file: str) -> None:
    """Track source locations for tensor arguments.

    Searches backwards from the kernel call site to find where each
    tensor variable was assigned, then registers that location.
    """
    if source_file == "<unknown>":
        return

    try:
        with open(source_file, "r") as sf:
            source_lines = sf.read().splitlines()
    except (IOError, OSError):
        return

    call_line = None
    for frame_info in inspect.stack():
        if frame_info.filename == source_file:
            call_line = frame_info.lineno
            break

    if not call_line:
        return

    for param_name, arg in zip(f_params, args):
        if not is_ttnn_tensor(arg):
            continue
        assign_line = find_variable_assignment(source_lines, param_name, call_line)
        if assign_line:
            register_tensor_source(arg, source_file, assign_line)


class CompiledTTNNKernel:
    """
    A compiled tt-lang kernel ready for execution via ttnn.generic_op.

    Caches compilation artifacts (kernel paths, CB descriptors) so the kernel
    can be executed multiple times with different tensors without recompiling.
    """

    def __init__(
        self,
        kernel_paths,
        kernel_configs,
        kernel_arg_specs,
        num_tensors,
        core_ranges,
        kernel_tensor_indices,
        kernel_core_ranges=None,
        cb_configs=None,
        dfb_reconfiguration_plan=None,
        program_hash=None,
        source_lines=None,
        all_source_lines=None,
        thread_to_kernel=None,
        kernel_line_offsets=None,
        num_pipe_sync_semaphores=0,
        pipe_sram_scratch_bytes=0,
        num_pipe_global_semaphores=0,
        num_dfb_resets=0,
        opaque_include_paths=None,
        kernel_pipe_computed_address_dfb_indices=None,
        kernel_logical_selectors=None,
        operation_name="<anonymous>",
        runtime_resource_factory: Optional[
            Callable[..., ProgramRuntimeResources]
        ] = None,
        runtime_resource_cache=None,
        kernel_used_dfb_indices=None,
    ):
        """
        Initialize with pre-compiled kernel artifacts.

        Args:
            kernel_paths: List of (path, thread_type) tuples for each kernel
            kernel_configs: List of config descriptors matching kernel_paths
            kernel_arg_specs: List of arg specs (rt_args list) for each kernel
            num_tensors: Number of input/output tensors
            core_ranges: CoreRangeSet for kernel execution
            kernel_tensor_indices: List of global tensor indices used by each kernel
            kernel_core_ranges: Optional list of per-kernel CoreRangeSet aligned
                with kernel_paths. Set by the per-core specialization path so
                each specialized clone is dispatched only to its own core; None
                entries fall back to the whole-grid core_ranges.
            cb_configs: Final physical DFB configurations indexed by cb_index
            dfb_reconfiguration_plan: Final boundary order and epoch configs.
            program_hash: Hash for tt-metal program cache
            source_lines: Source code lines for auto-profiling reports (deprecated)
            all_source_lines: Dict mapping kernel name to source lines
            thread_to_kernel: Dict mapping RISC thread name to kernel name
            kernel_line_offsets: Dict mapping kernel name to line offset
            num_pipe_sync_semaphores: Number of pipe synchronization
                semaphores used by this kernel
            pipe_sram_scratch_bytes: Per-core SRAM scratch bytes used by
                PipeNet metadata.
            num_pipe_global_semaphores: Number of GlobalSemaphore-backed
                PipeNet counters used by this kernel.
            num_dfb_resets: Number of synchronized DFB reset boundaries.
            kernel_pipe_computed_address_dfb_indices: Per-kernel receiver DFB indices whose
                L1 bases are supplied as common runtime args.
            kernel_logical_selectors: Logical selector for each compiled kernel.
            operation_name: User-facing operation name for runtime diagnostics.
            runtime_resource_factory: Optional per-invocation resource callback.
            runtime_resource_cache: Operation-owned persistent L1 resources.
            kernel_used_dfb_indices: Physical DFB indices referenced by each
                final specialized kernel. None entries are conservative.
        """
        self.kernel_paths = kernel_paths
        self.kernel_configs = kernel_configs
        self.kernel_arg_specs = kernel_arg_specs
        self.num_tensors = num_tensors
        self.core_ranges = core_ranges
        self.kernel_tensor_indices = kernel_tensor_indices
        self.kernel_core_ranges = kernel_core_ranges or [None] * len(kernel_paths)
        self.cb_configs = cb_configs or []
        self.dfb_reconfiguration_plan = dfb_reconfiguration_plan
        self.program_hash = program_hash
        self.source_lines = source_lines
        self.all_source_lines = all_source_lines or {}
        self.thread_to_kernel = thread_to_kernel or {}
        self.kernel_line_offsets = kernel_line_offsets or {}
        self.num_pipe_sync_semaphores = num_pipe_sync_semaphores
        self.num_dfb_resets = num_dfb_resets
        self.pipe_sram_scratch_bytes = pipe_sram_scratch_bytes
        self.num_pipe_global_semaphores = num_pipe_global_semaphores
        self.kernel_pipe_computed_address_dfb_indices = (
            kernel_pipe_computed_address_dfb_indices or [[] for _ in kernel_paths]
        )
        self.kernel_logical_selectors = (
            list(kernel_logical_selectors)
            if kernel_logical_selectors is not None
            else [None for _ in kernel_paths]
        )
        if runtime_resource_factory is not None:
            if len(self.kernel_logical_selectors) != len(kernel_paths):
                raise ValueError(
                    f"@ttl.operation {operation_name!r}: runtime_resource_factory "
                    "requires one logical-kernel selector per compiled kernel; "
                    f"got {len(self.kernel_logical_selectors)} selectors for "
                    f"{len(kernel_paths)} kernels"
                )
            missing_selector_indices = [
                kernel_index
                for kernel_index, selector in enumerate(self.kernel_logical_selectors)
                if selector is None
            ]
            if missing_selector_indices:
                raise ValueError(
                    f"@ttl.operation {operation_name!r}: runtime_resource_factory "
                    "requires logical-kernel selectors for compiled kernel indices "
                    f"{missing_selector_indices}"
                )
        self.kernel_used_dfb_indices = (
            kernel_used_dfb_indices
            if kernel_used_dfb_indices is not None
            else [None for _ in kernel_paths]
        )
        self.operation_name = operation_name
        self.runtime_resource_factory = runtime_resource_factory
        owns_runtime_resource_cache = runtime_resource_cache is None
        self._runtime_resource_cache = (
            KernelRuntimeResourceCache()
            if owns_runtime_resource_cache
            else runtime_resource_cache
        )
        if owns_runtime_resource_cache:
            self._runtime_resource_finalizer = attach_runtime_resource_finalizer(
                self, self._runtime_resource_cache
            )
        self.opaque_include_paths = opaque_include_paths or []

    def __call__(self, *args):
        """Execute the kernel with the given tensors."""
        if len(args) != self.num_tensors:
            raise ValueError(f"Expected {self.num_tensors} tensors, got {len(args)}")

        # Validate grid against device's compute grid.
        device = _require_device(args)
        device_grid = device.compute_with_storage_grid_size()
        kernel_grid = self.core_ranges.bounding_box().grid_size()
        if kernel_grid.x > device_grid.x or kernel_grid.y > device_grid.y:
            raise ValueError(
                f"Kernel grid ({kernel_grid.x}, {kernel_grid.y}) exceeds device "
                f"compute grid ({device_grid.x}, {device_grid.y}). "
                f"Reduce grid size to fit within available cores."
            )

        # Build kernel specs from stored kernel info.
        kernel_specs = []
        for kernel_idx, (kernel_path, thread_type) in enumerate(self.kernel_paths):
            tensor_indices = self.kernel_tensor_indices[kernel_idx]
            config = self.kernel_configs[kernel_idx]
            spec = KernelSpec(
                path=kernel_path,
                thread_type=thread_type,
                tensor_indices=tensor_indices,
                config=config,
                compiler_include_paths=self.opaque_include_paths,
                pipe_computed_address_dfb_indices=self.kernel_pipe_computed_address_dfb_indices[
                    kernel_idx
                ],
                core_ranges=self.kernel_core_ranges[kernel_idx],
                logical_kernel=self.kernel_logical_selectors[kernel_idx],
                used_dfb_indices=self.kernel_used_dfb_indices[kernel_idx],
            )
            kernel_specs.append(spec)

        # Use shared kernel execution logic.
        return run_kernel_on_device(
            kernel_specs=kernel_specs,
            tensors=list(args),
            cb_configs=self.cb_configs,
            dfb_reconfiguration_plan=self.dfb_reconfiguration_plan,
            core_ranges=self.core_ranges,
            program_hash=self.program_hash,
            num_pipe_sync_semaphores=self.num_pipe_sync_semaphores,
            num_dfb_resets=self.num_dfb_resets,
            pipe_sram_scratch_bytes=self.pipe_sram_scratch_bytes,
            num_pipe_global_semaphores=self.num_pipe_global_semaphores,
            runtime_resource_factory=self.runtime_resource_factory,
            operation_name=self.operation_name,
            runtime_resource_cache=self._runtime_resource_cache,
            device=device,
        )


def _write_kernel_to_tmp(name: str, source: str) -> str:
    """Write generated kernel source and return the path used by TT-Metal JIT."""
    import hashlib
    import os
    import tempfile

    content_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    user = os.environ.get("USER", "default")
    output_dir = Path("/tmp") / user
    worker_name = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_name:
        output_dir = output_dir / worker_name
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"ttlang_kernel_{name}_{content_hash}.cpp"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(source)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
    print(f"=== {name} kernel written to {path} ===")
    print(source)
    print("=" * 60)
    return str(path)


def _lookup_kernel_func_op(module, kernel_name: str):
    """Return the func.func operation for a kernel symbol."""
    for op_view in module.body.operations:
        operation = getattr(op_view, "operation", op_view)
        if operation.name != "func.func":
            continue
        sym_name = operation.attributes.get("sym_name", None)
        if sym_name is not None and str(sym_name).strip('"') == kernel_name:
            return operation
    raise RuntimeError(f"Could not find TTKernel function '{kernel_name}'")


def _set_unpack_to_dest_fp32(config, ttnn_mod, cb_indices) -> None:
    """Configure UnpackToDestFp32 for the listed CB indices.

    `cb_indices` is the set of circular buffer indices that the compiler
    determined need full-precision f32 unpack to DST (because at least one
    SFPU consumer reads an f32 tile from that CB directly into DST).
    """
    unpack_mode = ttnn_mod.UnpackToDestMode
    # The jit_build layer requires the vector size to be >= the number of CBs
    # for the target architecture (32 for WH B0, 64 for Blackhole). Use 64 to
    # cover both.
    num_cbs = 64
    cb_set = set(cb_indices)
    modes = config.unpack_to_dest_mode
    for i in range(num_cbs):
        modes.append(
            unpack_mode.UnpackToDestFp32 if i in cb_set else unpack_mode.Default
        )


def _set_math_fidelity(config, ttnn_mod, math_fidelity: str) -> None:
    try:
        config.math_fidelity = getattr(ttnn_mod.MathFidelity, math_fidelity)
    except AttributeError as error:
        raise RuntimeError(
            f"TTNN does not provide MathFidelity.{math_fidelity}"
        ) from error


def _get_kernel_bool_attr(module, kernel_name: str, attr_name: str) -> bool:
    """Read a boolean func.func attribute from a compiled kernel."""
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get(attr_name, None)
    if attr is None:
        raise ValueError(
            f"Required compiler-generated attribute '{attr_name}' is missing "
            f"from compute kernel '{kernel_name}'"
        )
    attr_text = str(attr).strip()
    if attr_text == "true":
        return True
    if attr_text == "false":
        return False
    raise ValueError(
        f"Expected boolean attribute '{attr_name}' on kernel '{kernel_name}', "
        f"got {attr_text!r}"
    )


def _get_kernel_i32_array_attr(module, kernel_name: str, attr_name: str):
    """Read a required `DenseI32ArrayAttr` kernel attribute."""
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get(attr_name, None)
    if attr is None:
        raise ValueError(
            f"Required compiler-generated attribute '{attr_name}' is missing "
            f"from compute kernel '{kernel_name}'"
        )
    if not isinstance(attr, DenseI32ArrayAttr):
        raise ValueError(
            f"Expected DenseI32ArrayAttr for '{attr_name}' on kernel "
            f"'{kernel_name}', got {attr}"
        )
    return list(attr)


def _get_kernel_optional_i32_array_attr(module, kernel_name: str, attr_name: str):
    """Read an optional `DenseI32ArrayAttr`. Missing returns None, empty returns []."""
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get(attr_name, None)
    if attr is None:
        return None
    if not isinstance(attr, DenseI32ArrayAttr):
        raise ValueError(
            f"Expected DenseI32ArrayAttr for '{attr_name}' on kernel "
            f"'{kernel_name}', got {attr}"
        )
    return list(attr)


def _get_kernel_core_coords(module, kernel_name: str):
    """Read the `ttl.core_coord` attribute set by `ttkernel-specialize-cores`.

    We expect the array to be of length 2 since node dim currently only supports 2D.
    The attribute is the launch coordinates a kernel is dispatched to.

    Returns the list of `(x, y)` launch coordinates for a specialized clone, or
    None when the kernel was not specialized (the whole-grid default path).
    """
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get("ttl.core_coord", None)
    if attr is None:
        return None
    if not isinstance(attr, ArrayAttr):
        raise ValueError(
            f"Expected an array for 'ttl.core_coord' on kernel "
            f"'{kernel_name}', got {attr}"
        )
    coords = []
    for pair in attr:
        pair = ArrayAttr(pair)
        if len(pair) != 2:
            raise ValueError(
                f"Expected length-2 [x, y] entries in 'ttl.core_coord' on "
                f"kernel '{kernel_name}', got {pair}"
            )
        coords.append(
            (int(IntegerAttr(pair[0]).value), int(IntegerAttr(pair[1]).value))
        )
    return coords


def _get_kernel_logical_selector(module, kernel_name: str) -> Optional[KernelSelector]:
    """Recover logical-kernel metadata retained by specialization clones."""
    operation = _lookup_kernel_func_op(module, kernel_name)
    raw_attribute = operation.attributes.get(_ttl_ir.LOGICAL_KERNEL_ATTR, None)
    if raw_attribute is None:
        return None
    attribute = ttl_dialect.LogicalKernelAttr.maybe_downcast(raw_attribute)
    if attribute is None:
        raise ValueError(
            f"Invalid '{_ttl_ir.LOGICAL_KERNEL_ATTR}' on kernel {kernel_name!r}"
        )

    if attribute.kind == ttl_dialect.ir.LogicalKernelKind.Compute:
        kind = KernelKind.COMPUTE
    elif attribute.kind == ttl_dialect.ir.LogicalKernelKind.DataMovement:
        kind = KernelKind.DATA_MOVEMENT
    else:
        raise ValueError(f"Unknown logical kernel kind on kernel {kernel_name!r}")

    if not attribute.identity:
        return kind
    return Kernel._from_metadata(
        kind,
        attribute.identity,
        operation_identity=attribute.operation,
        implicit_role=attribute.role,
    )


def _get_kernel_noc_index(module, kernel_name: str):
    """Read the `ttl.noc_index` attribute (0 = reader, 1 = writer).

    The frontend tags every datamovement thread with this attribute, required here so
    reader/writer role assignment is attribute-based rather than positional.
    """
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get("ttl.noc_index", None)
    if attr is None:
        raise ValueError(
            f"Missing 'ttl.noc_index' on datamovement kernel '{kernel_name}'"
        )
    return int(IntegerAttr(attr).value)


def _get_kernel_crta_indices(module, kernel_name: str):
    """Read the `ttl.crta_indices` attribute as a list of global tensor indices.

    Used by the per-core specialization path where clones cannot be aligned
    positionally with the original thread list. Raises an error if the attribute is missing.
    """
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get("ttl.crta_indices", None)
    if attr is None:
        raise ValueError(f"No CRTA indices found for kernel {kernel_name}")
    if not isinstance(attr, ArrayAttr):
        raise ValueError(
            f"Expected ArrayAttr for 'ttl.crta_indices' on kernel "
            f"'{kernel_name}', got {attr}"
        )
    return [int(IntegerAttr(idx).value) for idx in attr]


def _compile_ttnn_kernel(
    module,
    args,
    grid,
    num_outs,
    thread_tensor_indices,
    cb_configs=None,
    dfb_reconfiguration_plan=None,
    program_hash=None,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    math_fidelity: Optional[str] = None,
    verbose=True,
    source_lines=None,
    all_source_lines=None,
    kernel_line_offsets=None,
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
    num_dfb_resets: int = 0,
    opaque_include_paths: Optional[List[str]] = None,
    target_arch: Optional[str] = None,
    operation_name: str = "<anonymous>",
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    runtime_resource_cache: Optional[KernelRuntimeResourceCache] = None,
):
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.

    Args:
        module: MLIR module after TTL pipeline (with EmitC kernels)
        args: Input/output tensors (used for shape/dtype info)
        grid: Grid dimensions tuple
        num_outs: Number of output tensors
        program_hash: Hash for tt-metal program cache
        verbose: Print compilation info
        source_lines: Source code lines for auto-profiling reports

    Returns:
        CompiledTTNNKernel ready for execution
    """
    # Get kernel info from module
    kernel_info = get_ttkernel_names(module)

    # Validate tensor types: must be all TTNN or all torch, not mixed.
    # Mixed tensors would generate ToLayoutOps for host tensors, creating extra
    # bounce kernels that exceed the expected kernel count for core assignment.
    ttnn_count = sum(1 for arg in args if is_ttnn_tensor(arg))
    if ttnn_count > 0 and ttnn_count < len(args):
        raise ValueError(
            f"TTNN interop requires all tensors to be the same type. "
            f"Got {ttnn_count} TTNN tensors and {len(args) - ttnn_count} host tensors. "
            f"Mixed tensor types would generate extra bounce kernels."
        )

    # Validate TTNN tensors - must be L1 or DRAM and tilized
    for i, arg in enumerate(args):
        if is_ttnn_tensor(arg):
            mem_space = _detect_memory_space_from_tensor(arg, "unknown")
            if mem_space not in ("L1", "DRAM"):
                raise ValueError(
                    f"TTNN interop requires L1 or DRAM memory space, but tensor {i} is in {mem_space}."
                )
            if hasattr(arg, "layout") and "TILE" not in str(arg.layout):
                raise ValueError(
                    f"TTNN interop requires tilized tensors, but tensor {i} has layout {arg.layout}. "
                    f"Use ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) to convert."
                )

    # Detect the per-core specialization path: ttkernel-specialize-cores tags each
    # clone with ttl.core_coord (the list of coordinates the clone serves).
    # When present, get_ttkernel_names returns per-coordinate clones instead of
    # a single (compute + reader + writer) triple.
    kernel_coords = [_get_kernel_core_coords(module, name) for name, _ in kernel_info]
    kernel_logical_selectors = [
        _get_kernel_logical_selector(module, name) for name, _ in kernel_info
    ]
    specialize_cores = any(coords is not None for coords in kernel_coords)

    compute_count = sum(1 for _, t in kernel_info if t == "compute")
    dm_count = sum(1 for _, t in kernel_info if t == "noc")
    kernel_capacities = _backend_kernel_capacities(target_arch)
    kernel_counts = {
        KernelKind.COMPUTE: compute_count,
        KernelKind.DATA_MOVEMENT: dm_count,
    }
    if not specialize_cores:
        for kind in KernelKind:
            if kernel_counts[kind] > kernel_capacities[kind]:
                selected = tuple(
                    selector
                    for selector in kernel_logical_selectors
                    if selector is not None and _selector_kind(selector) == kind
                )
                if len(selected) != kernel_counts[kind]:
                    selected = (kind,) * kernel_counts[kind]
                raise ValueError(
                    _format_kernel_capacity_error(
                        kind, selected, kernel_capacities[kind]
                    )
                )
        if kernel_counts != kernel_capacities:
            required = ", ".join(
                f"{kernel_capacities[kind]} {kind.value}" for kind in KernelKind
            )
            provided = ", ".join(
                f"{kernel_counts[kind]} {kind.value}" for kind in KernelKind
            )
            raise ValueError(
                f"TTNN interop requires the target kernel set ({required}); "
                f"the operation provides {provided}"
            )
    else:
        # Validate every specialized core against the selected target.
        grid_cols, grid_rows = grid
        all_cores = [(x, y) for y in range(grid_rows) for x in range(grid_cols)]
        per_core_counts = {}
        for (name, thread_type), coords in zip(kernel_info, kernel_coords):
            covered = coords if coords is not None else all_cores
            for coord in covered:
                counts = per_core_counts.setdefault(tuple(coord), [0, 0])
                if thread_type == "compute":
                    counts[0] += 1
                elif thread_type == "noc":
                    counts[1] += 1
        for coord, (n_compute, n_noc) in per_core_counts.items():
            if (
                n_compute > kernel_capacities[KernelKind.COMPUTE]
                or n_noc > kernel_capacities[KernelKind.DATA_MOVEMENT]
            ):
                raise ValueError(
                    f"Per-core specialization assigned {n_compute} compute and "
                    f"{n_noc} data movement kernels to core {coord}. The target "
                    f"supports at most "
                    f"{kernel_capacities[KernelKind.COMPUTE]} compute and "
                    f"{kernel_capacities[KernelKind.DATA_MOVEMENT]} data movement "
                    "kernels per core."
                )

    if verbose:
        print("=" * 60)
        print("TTNN INTEROP: Compiling kernel")
        print("=" * 60)
        print(f"Found {len(kernel_info)} kernels:")

    if verbose:
        for name, thread_type in kernel_info:
            print(f"  - {name} ({thread_type})")

    _ensure_ttnn()
    if ttnn is None:
        print("\nttnn not available - cannot compile for ttnn.generic_op")
        return None

    # Build CoreRangeSet from grid dimensions
    # Grid is (cols, rows) = (x, y), matching tt-metal CoreCoord convention
    grid_cols, grid_rows = grid
    core_start = ttnn.CoreCoord(0, 0)
    core_end = ttnn.CoreCoord(grid_cols - 1, grid_rows - 1)
    core_range = ttnn.CoreRange(core_start, core_end)
    core_ranges = ttnn.CoreRangeSet([core_range])
    if verbose:
        print(f"\nCore range: {core_ranges}")

    kernel_paths = []
    kernel_configs = []
    kernel_arg_specs = []
    kernel_pipe_computed_address_dfb_indices = []
    kernel_used_dfb_indices = []
    # Per-kernel single-core ranges (specialization path) and tensor indices
    # read from ttl.crta_indices. Both stay aligned with kernel_info order.
    kernel_core_ranges = []
    specialized_tensor_indices = []
    kernel_config_attrs = {
        name: {
            "fp32_dest_acc_en": _get_kernel_bool_attr(module, name, "fp32_dest_acc_en"),
            "dst_full_sync_en": _get_kernel_bool_attr(module, name, "dst_full_sync_en"),
            "unpack_to_dest_fp32": _get_kernel_i32_array_attr(
                module, name, "ttl.unpack_to_dest_fp32"
            ),
        }
        for name, thread_type in kernel_info
        if thread_type == "compute"
    }

    # Build thread-to-kernel mapping for profiling
    # Maps RISC thread names to kernel names
    thread_to_kernel = {}

    for idx, (name, thread_type) in enumerate(kernel_info):
        cpp_source = ttkernel_to_cpp_by_name(module, name)
        kernel_path = _write_kernel_to_tmp(name, cpp_source)
        kernel_paths.append((kernel_path, thread_type))
        kernel_pipe_computed_address_dfb_indices.append(
            _get_kernel_optional_i32_array_attr(
                module, name, _ttl_ir.PIPE_COMPUTED_ADDRESS_DFB_INDICES_ATTR
            )
            or []
        )
        kernel_used_dfb_indices.append(
            _get_kernel_optional_i32_array_attr(
                module, name, _ttl_ir.USED_DFB_INDICES_ATTR
            )
        )

        # The specialized clone's launch coordinates (None on the default,
        # whole-grid path). Used to build the per-kernel dispatch range below.
        coords = kernel_coords[idx]

        if thread_type == "compute":
            config = ttnn.ComputeConfigDescriptor()
            if math_fidelity is not None:
                _set_math_fidelity(config, ttnn, math_fidelity)
            if fp32_dest_acc_en is not None:
                config.fp32_dest_acc_en = fp32_dest_acc_en
            elif kernel_config_attrs[name]["fp32_dest_acc_en"]:
                config.fp32_dest_acc_en = True
            if dst_full_sync_en is not None:
                config.dst_full_sync_en = dst_full_sync_en
            elif kernel_config_attrs[name]["dst_full_sync_en"]:
                config.dst_full_sync_en = True
            unpack_fp32_cbs = kernel_config_attrs[name]["unpack_to_dest_fp32"]
            if unpack_fp32_cbs:
                _set_unpack_to_dest_fp32(config, ttnn, unpack_fp32_cbs)
            # Compute kernels run on TRISC threads
            thread_to_kernel["TRISC_0"] = name
            thread_to_kernel["TRISC_1"] = name
            thread_to_kernel["TRISC_2"] = name
        elif thread_type == "noc":
            noc_role = _get_kernel_noc_index(module, name)
            if noc_role == 0:
                config = ttnn.ReaderConfigDescriptor()
                thread_to_kernel["NCRISC"] = name  # Reader
            else:
                config = ttnn.WriterConfigDescriptor()
                thread_to_kernel["BRISC"] = name  # Writer
        else:
            config = ttnn.ReaderConfigDescriptor()
        kernel_configs.append(config)

        # Turn the specialized clone's coordinates into a CoreRangeSet
        if coords is not None:
            kernel_core_ranges.append(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy))
                        for (cx, cy) in coords
                    ]
                )
            )
        else:
            kernel_core_ranges.append(None)
        # Clones cannot be aligned positionally with the original thread list,
        # so recover each kernel's global tensor indices from ttl.crta_indices.
        # Only needed on the specialization path; the default path uses the
        # positional thread_tensor_indices instead.
        if specialize_cores:
            specialized_tensor_indices.append(_get_kernel_crta_indices(module, name))

        # Extract runtime args from kernel's arg_spec attribute
        arg_spec = get_ttkernel_arg_spec(module, name)
        if arg_spec is not None:
            arg_spec = ttkernel.ir.ArgSpecAttr.maybe_downcast(arg_spec)
            kernel_arg_specs.append(arg_spec.rt_args if arg_spec else [])
        else:
            kernel_arg_specs.append([])

    # On the specialization path get_ttkernel_names returns 3*N clones, so the
    # positional thread_tensor_indices (one entry per original thread) no longer
    # lines up; use the per-clone indices recovered from ttl.crta_indices.
    kernel_tensor_indices = (
        specialized_tensor_indices if specialize_cores else thread_tensor_indices
    )

    compiled_kernel = CompiledTTNNKernel(
        kernel_paths=kernel_paths,
        kernel_configs=kernel_configs,
        kernel_arg_specs=kernel_arg_specs,
        num_tensors=len(args),
        core_ranges=core_ranges,
        kernel_tensor_indices=kernel_tensor_indices,
        kernel_core_ranges=kernel_core_ranges,
        cb_configs=cb_configs,
        dfb_reconfiguration_plan=dfb_reconfiguration_plan,
        program_hash=program_hash,
        source_lines=source_lines,
        all_source_lines=all_source_lines,
        thread_to_kernel=thread_to_kernel,
        kernel_line_offsets=kernel_line_offsets,
        num_pipe_sync_semaphores=num_pipe_sync_semaphores,
        num_dfb_resets=num_dfb_resets,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
        opaque_include_paths=opaque_include_paths or [],
        kernel_pipe_computed_address_dfb_indices=kernel_pipe_computed_address_dfb_indices,
        kernel_logical_selectors=kernel_logical_selectors,
        operation_name=operation_name,
        runtime_resource_factory=runtime_resource_factory,
        runtime_resource_cache=runtime_resource_cache,
        kernel_used_dfb_indices=kernel_used_dfb_indices,
    )

    if verbose:
        print(f"\nCompiled kernel ready (compiled {len(kernel_paths)} threads)")
        print("=" * 60)

    emit_runner_path = os.environ.get("TTLANG_EMIT_RUNNER")
    if emit_runner_path:
        kernel_specs_for_emit = []
        for kernel_idx, (kernel_path, thread_type) in enumerate(kernel_paths):
            tensor_indices = kernel_tensor_indices[kernel_idx]
            spec = KernelSpec(
                path=kernel_path,
                thread_type=thread_type,
                tensor_indices=tensor_indices,
                config=kernel_configs[kernel_idx],
                compiler_include_paths=opaque_include_paths or [],
                pipe_computed_address_dfb_indices=kernel_pipe_computed_address_dfb_indices[
                    kernel_idx
                ],
                core_ranges=kernel_core_ranges[kernel_idx],
                logical_kernel=kernel_logical_selectors[kernel_idx],
                used_dfb_indices=kernel_used_dfb_indices[kernel_idx],
            )
            kernel_specs_for_emit.append(spec)

        if emit_runner_path == "1":
            first_kernel_path = kernel_paths[0][0]
            runner_path = first_kernel_path.replace(".cpp", "_runner.py")
        else:
            runner_path = emit_runner_path

        emit_runner_file(
            kernel_specs=kernel_specs_for_emit,
            cb_configs=cb_configs,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            num_tensors=len(args),
            output_path=runner_path,
            program_hash=program_hash,
            kernel_name=operation_name,
            num_pipe_sync_semaphores=num_pipe_sync_semaphores,
            num_dfb_resets=num_dfb_resets,
            pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
            num_pipe_global_semaphores=num_pipe_global_semaphores,
            requires_runtime_resource_factory=runtime_resource_factory is not None,
            dfb_reconfiguration_plan=dfb_reconfiguration_plan,
        )

    return compiled_kernel


def _build_operation_pipenets(f: Callable, threads):
    """Discover PipeNets reachable from the operation and its threads, build
    the OperationPipeNets, validate it, and assign each Pipe its
    operation-local pipe-net id for AST emission.

    Discovery walks the operation function's closure plus each thread
    function's closure (matching the spec's "captured by the operation
    function" wording). PipeNets are deduplicated by `id()`, so a
    captured PipeNet referenced from multiple threads contributes one
    entry.
    """
    seen: Dict[int, PipeNet] = {}

    def visit(func):
        if func is None:
            return
        closure = getattr(func, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, PipeNet) and id(value) not in seen:
                seen[id(value)] = value
        fn_globals = getattr(func, "__globals__", None) or {}
        for value in fn_globals.values():
            if isinstance(value, PipeNet) and id(value) not in seen:
                seen[id(value)] = value

    visit(f)
    for thread in threads:
        visit(getattr(thread, "__wrapped__", None))

    return _build_pipenet_graph(seen.values())


def _build_pipenet_graph(nets):
    """Build the OperationPipeNets graph for a sequence of PipeNet objects,
    assigning each net (and its pipes) a dense operation-local id and
    validating the result. Shared by @ttl.operation (closure discovery)
    and unified @ttl.operation bodies (lifted PipeNet assigns)."""
    from ._pipenets import OperationPipeNets
    from .pipe import _pipe_to_pipe_use

    graph = OperationPipeNets()
    for net in nets:
        net_use = graph.add_pipe_net(_pipe_to_pipe_use(p) for p in net.pipes)
        net.pipe_net_id = net_use.id
        # Assign every Pipe in this net the operation-local id so the AST
        # visitor's create_pipe emission uses the same id space.
        for pipe in net.pipes:
            pipe.pipe_net_id = net_use.id

    graph.validate()
    return graph


def _collect_captures(
    f: Callable,
    bound_dispatch_conditions: Optional[Mapping[str, _BoundDispatchCondition]] = None,
    bound_dfb_resets: Optional[Mapping[str, _BoundDFBReset]] = None,
    bound_dfb_reconfigurations: Optional[Mapping[str, _BoundDFBReconfiguration]] = None,
) -> Dict[str, Any]:
    """
    Collect and convert captured variables from function closure.

    Args:
        f: Function with closure to inspect

    Returns:
        Dictionary mapping variable names to converted values

    Raises:
        TypeError: If closure contains unsupported variable types
    """
    if f.__closure__ is None:
        return {}

    def convert(name, val):
        if val is None:
            return val
        if isinstance(val, (int, float)):
            return val
        elif is_ttnn_global_semaphore(val):
            return val
        elif is_ttnn_tensor(val):
            return val
        elif isinstance(val, DataflowBuffer):
            return val
        elif isinstance(val, Pipe):
            return val
        elif isinstance(val, PipeNet):
            return val
        # A tuple or list of scalars is a compile-time shape or axis list. It
        # reaches the same consumers as the equivalent literal written inline,
        # so it stays a Python value rather than becoming an SSA operand.
        elif isinstance(val, (tuple, list)) and all(
            isinstance(elt, (int, float)) for elt in val
        ):
            return val
        elif val is ScalarType or isinstance(val, ScalarType):
            return val
        elif isinstance(val, DispatchCondition):
            bound_condition = (
                bound_dispatch_conditions.get(name)
                if bound_dispatch_conditions is not None
                else None
            )
            if bound_condition is not None and bound_condition.declaration is val:
                return bound_condition
            return _bind_current_dispatch_condition(val)
        elif isinstance(val, DFBReset):
            bound_reset = (
                bound_dfb_resets.get(name) if bound_dfb_resets is not None else None
            )
            if bound_reset is not None and bound_reset.declaration is val:
                return bound_reset
            return _bind_current_dfb_reset(val)
        elif isinstance(val, DFBReconfiguration):
            bound_reconfiguration = (
                bound_dfb_reconfigurations.get(name)
                if bound_dfb_reconfigurations is not None
                else None
            )
            if (
                bound_reconfiguration is not None
                and bound_reconfiguration.declaration is val
            ):
                return bound_reconfiguration
            return _bind_current_dfb_reconfiguration(val)
        else:
            raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }


# Map scalar MLIR element types to ttnn-compatible data format names.
_MLIR_SCALAR_TYPE_TO_FORMAT = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "i32": "int32",
    "si32": "int32",
    "ui8": "uint8",
    "ui32": "uint32",
    "ui16": "uint16",
}


_MLIR_TILE_DATA_TYPE_TO_FORMAT = {
    ttcore.DataType.Float32: "float32",
    ttcore.DataType.Float16: "float16",
    ttcore.DataType.BFloat16: "bfloat16",
    ttcore.DataType.BFP_BFloat8: "bfloat8_b",
    ttcore.DataType.BFP_BFloat4: "bfloat4_b",
    ttcore.DataType.UInt32: "uint32",
    ttcore.DataType.UInt16: "uint16",
    ttcore.DataType.UInt8: "uint8",
    ttcore.DataType.Int32: "int32",
}


def _parse_mlir_element_type(
    element_type_attr,
) -> tuple[str, Optional[tuple[int, int]]]:
    """Extract the data format and optional tile dimensions from a TypeAttr.

    The TypeAttr prints as e.g. "bf16" or "!ttcore.tile<32x32, bf16>".
    """
    if not isinstance(element_type_attr, TypeAttr):
        raise ValueError(
            "Physical DFB element_type metadata must be a TypeAttr, "
            f"got {element_type_attr}"
        )
    element_type = element_type_attr.value
    tile_type = ttcore.ir.TileType.maybe_downcast(element_type)
    if tile_type is not None:
        data_type = ttcore.DataType(tile_type.data_type_as_int)
        data_format = _MLIR_TILE_DATA_TYPE_TO_FORMAT.get(data_type)
        if data_format is None:
            raise ValueError(
                "Physical DFB tile data type "
                f"'{data_type.name}' is not supported by the ttnn runtime"
            )
        return data_format, tuple(tile_type.shape)

    type_str = str(element_type).strip()
    data_format = _MLIR_SCALAR_TYPE_TO_FORMAT.get(type_str)
    if data_format is not None:
        return data_format, None
    known_types = list(_MLIR_SCALAR_TYPE_TO_FORMAT.keys())
    raise ValueError(
        f"Unrecognized MLIR scalar element type '{type_str}'. "
        f"Known types: {known_types}"
    )


def _extract_dfb_node_coordinates(
    nodes_attr, *, context: str, allow_empty: bool
) -> tuple[tuple[int, int], ...]:
    nodes = []
    for node_position, node_attr in enumerate(nodes_attr):
        node = ArrayAttr(node_attr)
        if len(node) != 2:
            raise ValueError(f"{context}[{node_position}] must contain [x, y]")
        coordinate = tuple(int(IntegerAttr(component).value) for component in node)
        if coordinate[0] < 0 or coordinate[1] < 0:
            raise ValueError(f"{context} contains negative coordinate {coordinate}")
        if coordinate in nodes:
            raise ValueError(f"{context} contains duplicate coordinate {coordinate}")
        nodes.append(coordinate)
    if not allow_empty and not nodes:
        raise ValueError(f"{context} must not be empty")
    return tuple(sorted(nodes))


def _parse_physical_dfb_config(entry, *, dfb_index: int, context: str):
    """Parse and validate one compiler-emitted physical DFB configuration."""
    required_fields = ("num_tiles", "element_type", "block_count", "page_size")
    for field in required_fields:
        if field not in entry:
            raise ValueError(f"{context} is missing '{field}'")
    try:
        num_tiles = int(entry["num_tiles"])
        block_count = int(entry["block_count"])
        page_size = int(entry["page_size"])
        data_format, tile = _parse_mlir_element_type(entry["element_type"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {context}: {error}") from None
    for field, value in (
        ("num_tiles", num_tiles),
        ("block_count", block_count),
        ("page_size", page_size),
    ):
        if value <= 0:
            raise ValueError(f"{context}.{field} must be positive, got {value}")

    allocation_nodes = None
    if "allocation_nodes" in entry:
        allocation_nodes = _extract_dfb_node_coordinates(
            entry["allocation_nodes"],
            context=f"{context}.allocation_nodes",
            allow_empty=True,
        )

    storage_segments = []
    seen_nodes = set()
    storage_segment_entries = (
        entry["storage_segments"] if "storage_segments" in entry else []
    )
    for segment_position, segment in enumerate(storage_segment_entries):
        segment_context = f"{context}.storage_segments[{segment_position}]"
        if "nodes" not in segment:
            raise ValueError(f"{segment_context} is missing 'nodes'")
        nodes = _extract_dfb_node_coordinates(
            segment["nodes"],
            context=f"{segment_context}.nodes",
            allow_empty=False,
        )
        for coordinate in nodes:
            if coordinate in seen_nodes:
                raise ValueError(
                    f"{context} assigns launch node {coordinate} to multiple segments"
                )
            seen_nodes.add(coordinate)

        tensor_index = None
        byte_offset = 0
        byte_size = None
        if "tensor_backing" in segment:
            backing = ttl_dialect.TensorBackingAttr.maybe_downcast(
                segment["tensor_backing"]
            )
            if backing is None:
                raise ValueError(f"{segment_context}.tensor_backing has the wrong type")
            tensor_index = backing.tensor_index
            byte_offset = backing.byte_offset
            byte_size = backing.byte_size
            expected_size = num_tiles * block_count * page_size
            if byte_size != expected_size:
                raise ValueError(
                    f"{context} tensor backing byte_size must equal "
                    f"{expected_size}, got {byte_size}"
                )
        storage_segments.append(
            DFBStorageSegment(
                nodes=nodes,
                tensor_index=tensor_index,
                byte_offset=byte_offset,
                byte_size=byte_size,
            )
        )
    return PhysicalDFBConfig(
        dfb_index=dfb_index,
        num_tiles=num_tiles,
        data_format=data_format,
        block_count=block_count,
        page_size=page_size,
        tile=tile,
        storage_segments=tuple(storage_segments),
        allocation_nodes=allocation_nodes,
    )


def _extract_dfb_allocations(module):
    """Read `ttl.dfb_allocations` and require dense physical indices."""
    attribute_name = "ttl.dfb_allocations"
    attr = module.operation.attributes.get(attribute_name, None)
    if attr is None:
        return None

    configs = []
    seen_indices = set()
    for position, entry in enumerate(attr):
        context = f"{attribute_name}[{position}]"
        if "dfb_index" not in entry:
            raise ValueError(f"{context} is missing 'dfb_index'")
        dfb_index = int(entry["dfb_index"])
        if dfb_index < 0:
            raise ValueError(f"{context}.dfb_index must be non-negative")
        if dfb_index in seen_indices:
            raise ValueError(
                f"{attribute_name} contains duplicate dfb_index {dfb_index}"
            )
        seen_indices.add(dfb_index)
        configs.append(
            _parse_physical_dfb_config(entry, dfb_index=dfb_index, context=context)
        )

    configs.sort(key=lambda config: config.dfb_index)
    indices = [config.dfb_index for config in configs]
    expected_indices = list(range(len(configs)))
    if indices != expected_indices:
        raise ValueError(
            f"{attribute_name} must contain a dense physical index range "
            f"{expected_indices}, got {indices}"
        )
    return configs


def _extract_dfb_reconfiguration_plan(module, physical_configs):
    """Read finalized configuration epochs and boundary order."""
    attribute_name = "ttl.dfb_reconfiguration_plan"
    plan_attr = module.operation.attributes.get(attribute_name, None)
    if plan_attr is None:
        return None
    for field in ("boundary_ordinals", "dfbs"):
        if field not in plan_attr:
            raise ValueError(f"{attribute_name} is missing '{field}'")
    boundary_ordinals = tuple(int(value) for value in plan_attr["boundary_ordinals"])
    if not boundary_ordinals or any(ordinal < 0 for ordinal in boundary_ordinals):
        raise ValueError(f"{attribute_name}.boundary_ordinals must be non-empty")
    if len(set(boundary_ordinals)) != len(boundary_ordinals):
        raise ValueError(f"{attribute_name}.boundary_ordinals must be unique")

    dfb_epochs_by_index = {}
    for position, dfb_entry in enumerate(plan_attr["dfbs"]):
        context = f"{attribute_name}.dfbs[{position}]"
        for field in ("dfb_index", "configurations"):
            if field not in dfb_entry:
                raise ValueError(f"{context} is missing '{field}'")
        dfb_index = int(dfb_entry["dfb_index"])
        if dfb_index in dfb_epochs_by_index:
            raise ValueError(f"{attribute_name} contains duplicate index {dfb_index}")
        epochs = []
        seen_entries = set()
        for epoch_position, epoch_entry in enumerate(dfb_entry["configurations"]):
            epoch_context = f"{context}.configurations[{epoch_position}]"
            entry_ordinal = (
                int(epoch_entry["entry_reconfiguration"])
                if "entry_reconfiguration" in epoch_entry
                else None
            )
            if entry_ordinal in seen_entries:
                raise ValueError(f"{context} contains a duplicate configuration epoch")
            if entry_ordinal is not None and entry_ordinal not in boundary_ordinals:
                raise ValueError(
                    f"{epoch_context}.entry_reconfiguration is not a boundary"
                )
            seen_entries.add(entry_ordinal)
            epochs.append(
                DFBConfigurationEpoch(
                    entry_reconfiguration_ordinal=entry_ordinal,
                    config=_parse_physical_dfb_config(
                        epoch_entry, dfb_index=dfb_index, context=epoch_context
                    ),
                )
            )
        if not epochs:
            raise ValueError(f"{context}.configurations must not be empty")
        dfb_epochs_by_index[dfb_index] = tuple(epochs)

    expected_indices = list(range(len(physical_configs)))
    if sorted(dfb_epochs_by_index) != expected_indices:
        raise ValueError(
            f"{attribute_name}.dfbs must contain indices {expected_indices}"
        )
    for dfb_index, physical_config in enumerate(physical_configs):
        epochs = dfb_epochs_by_index[dfb_index]
        initial_epoch = next(
            (epoch for epoch in epochs if epoch.entry_reconfiguration_ordinal is None),
            epochs[0],
        )
        initial_config = initial_epoch.config
        same_geometry = (
            initial_config.dfb_index == physical_config.dfb_index
            and initial_config.num_tiles == physical_config.num_tiles
            and initial_config.data_format == physical_config.data_format
            and initial_config.block_count == physical_config.block_count
            and initial_config.page_size == physical_config.page_size
            and initial_config.tile == physical_config.tile
        )
        initial_tensor_segments = tuple(
            segment
            for segment in initial_config.storage_segments
            if segment.is_tensor_backed
        )
        physical_tensor_segments = tuple(
            segment
            for segment in physical_config.storage_segments
            if segment.is_tensor_backed
        )
        if not same_geometry or initial_tensor_segments != physical_tensor_segments:
            raise ValueError(
                f"{attribute_name}.dfbs[{dfb_index}] initial configuration "
                "does not match ttl.dfb_allocations"
            )
    return DFBReconfigurationPlan(
        boundary_ordinals=boundary_ordinals,
        dfb_epochs=tuple(dfb_epochs_by_index[index] for index in expected_indices),
    )


def _extract_pipe_sync_semaphore_count(module) -> Optional[int]:
    """Read the semaphore count selected by pipe lowering."""
    attr = module.operation.attributes.get(_ttl_ir.PIPE_SYNC_SEMAPHORE_COUNT_ATTR, None)
    if attr is None:
        return None
    return int(attr)


def _extract_dfb_reset_count(module) -> int:
    """Read the number of synchronized DFB reset boundaries."""
    attr = module.operation.attributes.get(_ttl_ir.DFB_RESET_COUNT_ATTR, None)
    if attr is None:
        return 0
    return int(attr)


def _extract_pipe_sram_scratch_bytes(module) -> int:
    """Read the per-core SRAM scratch bytes selected by pipe lowering."""
    attr = module.operation.attributes.get(_ttl_ir.PIPE_SRAM_SCRATCH_BYTES_ATTR, None)
    if attr is None:
        return 0
    return int(attr)


def _extract_pipe_global_semaphore_count(module) -> int:
    """Read the GlobalSemaphore count selected by pipe lowering."""
    attr = module.operation.attributes.get(
        _ttl_ir.PIPE_GLOBAL_SEMAPHORE_COUNT_ATTR, None
    )
    if attr is None:
        return 0
    return int(attr)


def _resolve_dfb_configs(module):
    """Return finalized physical DFB configurations from required metadata."""
    physical_allocations = _extract_dfb_allocations(module)
    if physical_allocations is None:
        raise ValueError(
            "compiled module is missing ttl.dfb_allocations; "
            "ttl-finalize-dfb-indices must run before runtime construction"
        )
    return physical_allocations


def _run_thread_compiler(
    name,
    kernel_type,
    captures,
    globals_,
    args,
    kwargs,
    module_ast,
    source_lines,
    source_file,
    verbose=False,
):
    """Construct a TTLGenericCompiler for one thread, visit its AST, and
    verify. Returns the compiler instance (the compiled thread).

    Shared by @ttl.operation thread wrappers (which build module_ast from
    the thread function source) and unified @ttl.operation (which feeds a
    synthesized per-thread function AST), so the per-thread lowering entry
    is in one place.
    """
    b = TTLGenericCompiler(
        name,
        kernel_type,
        captures,
        *args,
        _globals=globals_,
        **kwargs,
    )
    if verbose:
        print(ast.dump(module_ast, indent=4) + "\n")
    b.visit(module_ast)
    if verbose:
        print(b.module)
    try:
        b.module.operation.verify()
    except Exception as e:
        formatted = format_mlir_error(str(e), source_lines, source_file)
        raise RuntimeError(formatted) from None
    return b


def _compile(
    kernel_type: Optional[str] = None,
    verbose: bool = False,
    logical_kernel: Optional[KernelSelector] = None,
) -> Callable:
    """
    Internal decorator for compiling kernel threads.

    Args:
        kernel_type: Type of kernel ("compute" or "datamovement")
        verbose: Enable verbose compilation output
        logical_kernel: Target-independent logical kernel selector

    Returns:
        Decorator function for kernel compilation
    """

    def _decorator(f):
        expected_kind = {
            "compute": KernelKind.COMPUTE,
            "datamovement": KernelKind.DATA_MOVEMENT,
        }[kernel_type]
        selected_kernel = expected_kind if logical_kernel is None else logical_kernel
        if not isinstance(selected_kernel, (KernelKind, Kernel)):
            raise TypeError(
                "kernel must be a KernelKind or Kernel, got "
                f"{type(selected_kernel).__name__}"
            )
        if _selector_kind(selected_kernel) != expected_kind:
            raise ValueError(
                f"{kernel_type} thread kernel kind must be "
                f"{expected_kind.value}, got {_selector_kind(selected_kernel).value}"
            )
        if isinstance(selected_kernel, Kernel) and selected_kernel._identity is None:
            raise ValueError(
                "kernel handle must be captured by the enclosing @ttl.operation "
                "before it is used by a thread decorator"
            )

        # Capture source file at decoration time
        try:
            source_file = inspect.getfile(f)
        except (TypeError, OSError):
            source_file = "<unknown>"

        bound_dispatch_conditions = {
            name: _bind_current_dispatch_condition(cell.cell_contents)
            for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ())
            if isinstance(cell.cell_contents, DispatchCondition)
        }
        bound_dfb_resets = {
            name: _bind_current_dfb_reset(cell.cell_contents)
            for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ())
            if isinstance(cell.cell_contents, DFBReset)
        }
        bound_dfb_reconfigurations = {
            name: _bind_current_dfb_reconfiguration(cell.cell_contents)
            for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ())
            if isinstance(cell.cell_contents, DFBReconfiguration)
        }

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            source_code = _cleanup_source_code(f)
            source_lines = source_code.splitlines()

            if verbose:
                kwargs["_source_code"] = source_lines
                kwargs["_verbose"] = True

            # Pass source info for debug locations (always enabled for error messages)
            kwargs["_source_file"] = source_file
            kwargs["_source_lines"] = source_lines
            kwargs["_line_offset"] = _get_source_line_offset(f)
            kwargs["debug_locations"] = True

            m = ast.parse(source_code)
            return _run_thread_compiler(
                f.__name__,
                kernel_type,
                _collect_captures(
                    f,
                    bound_dispatch_conditions,
                    bound_dfb_resets,
                    bound_dfb_reconfigurations,
                ),
                f.__globals__,
                args,
                kwargs,
                m,
                source_lines,
                source_file,
                verbose=verbose,
            )

        _wrapper._decorator_name = kernel_type + "_thread"
        _wrapper._source_file = source_file
        _wrapper._logical_kernel = selected_kernel
        # Register thread for automatic collection
        _register_thread(_wrapper)
        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def compute(
    verbose: bool = False, *, kernel: Optional[KernelSelector] = None
) -> Callable:
    """
    Decorator for compute thread functions.

    Compute threads execute on Tensix cores and perform mathematical operations.

    Args:
        verbose: Enable verbose compilation output
        kernel: Logical compute kernel selected for this thread

    Returns:
        Decorator for compute kernel compilation
    """
    return _compile(
        kernel_type="compute",
        verbose=verbose,
        logical_kernel=kernel,
    )


def datamovement(
    verbose: bool = False, *, kernel: Optional[KernelSelector] = None
) -> Callable:
    """
    Decorator for data movement thread functions.

    Data movement threads handle DMA operations between memory hierarchies.

    Args:
        verbose: Enable verbose compilation output
        kernel: Logical data-movement kernel selected for this thread

    Returns:
        Decorator for data movement kernel compilation
    """
    return _compile(
        kernel_type="datamovement",
        verbose=verbose,
        logical_kernel=kernel,
    )


class Program:
    """
    Immutable container for kernel threads and their arguments.

    A Program encapsulates compute and data movement threads along with
    the arguments to be passed during execution. After construction, all
    fields should be treated as read-only.
    """

    def __init__(self, *threads, args=(), kwargs=None):
        self._threads = threads
        self._args = args
        self._kwargs = kwargs if kwargs is not None else {}

    @property
    def threads(self) -> tuple:
        return self._threads

    @property
    def args(self) -> tuple:
        return self._args

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    def __call__(self, *args, **kwargs):
        return Program(*self.threads, args=args, kwargs={**self.kwargs, **kwargs})


def _compile_kernel(
    f: Callable,
    args: tuple,
    kwargs: dict,
    grid: Union[tuple, List[int]],
    indexing_maps: List[Callable],
    iterator_types: List[str],
    num_outs: int,
    memory_space: str,
    tiled: bool,
    program_hash: int,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    math_fidelity: Optional[str] = None,
    target_arch: Optional[str] = None,
    compiler_options: CompilerOptions = CompilerOptions(),
    l1_budget_override: int = 0,
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    runtime_resource_cache: Optional[KernelRuntimeResourceCache] = None,
) -> Optional[CompiledTTNNKernel]:
    """
    Compile kernel function to MLIR and return CompiledTTNNKernel.

    Args:
        f: User kernel function
        args: Positional arguments for the kernel
        kwargs: Keyword arguments for the kernel
        grid: Grid dimensions
        indexing_maps: List of lambda functions for indexing
        iterator_types: List of iterator type strings
        num_outs: Number of output arguments
        memory_space: "L1" or "DRAM"
        tiled: Whether to use tiled layout
        program_hash: Hash for tt-metal program cache
        fp32_dest_acc_en: Optional override for fp32_dest_acc_en
        dst_full_sync_en: Optional override for dst_full_sync_en
        math_fidelity: Optional TTNN compute math fidelity
        target_arch: Optional TT device architecture for target-specific lowering
        compiler_options: Compiler pipeline options
        l1_budget_override: Explicit or device-derived L1 allocation budget
        runtime_resource_cache: Persistent resources shared by operation variants

    Returns:
        CompiledTTNNKernel ready for execution
    """
    f_params = inspect.signature(f).parameters

    # Get kernel source location for error reporting
    try:
        kernel_source_file = inspect.getfile(f)
        kernel_line_offset = _get_source_line_offset(f)
    except (TypeError, OSError):
        kernel_source_file = "<unknown>"
        kernel_line_offset = 0

    has_ttnn_tensors = any(is_ttnn_tensor(arg) for arg in args)

    # For mesh tensors, tensor.shape already returns the per-device shard
    # dimensions, so no wrapping is needed.
    is_mesh = has_ttnn_tensors and any(_is_mesh_tensor(arg) for arg in args)
    compile_args = args

    # For TTNN tensors, detect memory space from tensor's buffer type.
    # L1 tensors use simple NOC addressing, DRAM uses bank-aware addressing.
    # TODO: Check all tensors and handle mixed memory spaces.
    if has_ttnn_tensors:
        first_ttnn_tensor = next((arg for arg in args if is_ttnn_tensor(arg)), None)
        if first_ttnn_tensor is not None:
            memory_space = _detect_memory_space_from_tensor(
                first_ttnn_tensor, memory_space
            )
            print(f"[TTNN interop] Detected {memory_space} memory space")

    for idx, (param_name, arg) in enumerate(zip(f_params, compile_args)):
        register_tensor_name(arg, param_name, index=idx)

    # For pretty error printing only:
    _track_tensor_sources(f_params, args, kernel_source_file)

    inject_kwargs = [
        ("grid", grid),
        ("memory_space", memory_space),
        ("tiled", tiled),
    ]
    for injected_kwarg, val in inject_kwargs:
        if injected_kwarg in f_params:
            kwargs[injected_kwarg] = val

    from .dataflow_buffer import _reset_cb_counter
    from .operators import _set_current_grid

    _reset_cb_counter()
    _set_current_grid(grid)

    _clear_thread_registry()
    call_args = []
    call_kwargs = dict(kwargs)
    for param, value in zip(f_params.values(), compile_args):
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            call_kwargs[param.name] = value
        else:
            call_args.append(value)
    with (
        _dispatch_condition_binding_scope(),
        _dfb_allocation_group_binding_scope(),
        _dfb_reset_binding_scope(),
        _dfb_reconfiguration_binding_scope(),
    ):
        f(*call_args, **call_kwargs)
    threads = _get_registered_threads()

    if not threads:
        raise ValueError(
            "No threads found. Define at least one @ttl.compute() or "
            "@ttl.datamovement() function inside your kernel."
        )

    _validate_explicit_logical_kernel_uses(
        threads, _backend_kernel_capacities(target_arch)
    )

    pipenets = _build_operation_pipenets(f, threads)

    launch_grid = grid

    injected_program_kwargs = {
        "grid": grid,
        "memory_space": memory_space,
        "tiled": tiled,
        "debug_locations": True,  # Always generate locations for error messages
    }
    program = Program(
        *threads,
        args=compile_args,
        kwargs=injected_program_kwargs,
    )

    return _lower_program_to_kernel(
        program=program,
        args=args,
        launch_grid=launch_grid,
        num_outs=num_outs,
        pipenets=pipenets,
        target_arch=target_arch,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        math_fidelity=math_fidelity,
        compiler_options=compiler_options,
        program_hash=program_hash,
        l1_budget_override=l1_budget_override,
        kernel_source_file=kernel_source_file,
        kernel_line_offset=kernel_line_offset,
        logical_kernels=[thread._logical_kernel for thread in threads],
        operation_name=f.__name__,
        runtime_resource_factory=runtime_resource_factory,
        runtime_resource_cache=runtime_resource_cache,
    )


def _lower_program_to_kernel(
    *,
    program,
    args,
    launch_grid,
    num_outs,
    pipenets,
    target_arch,
    fp32_dest_acc_en,
    dst_full_sync_en,
    math_fidelity,
    compiler_options,
    program_hash,
    l1_budget_override,
    kernel_source_file,
    kernel_line_offset,
    logical_kernels=None,
    operation_name="<anonymous>",
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    runtime_resource_cache: Optional[KernelRuntimeResourceCache] = None,
):
    """Lower compiled threads to a CompiledTTNNKernel.

    Assembles the per-thread MLIR funcs into a module, runs the TTL pass
    pipeline, and builds the runner. Shared by both @ttl.operation forms
    so the compiler pipeline lives in one place.
    """
    # Always generate source locations for error messages
    # TTLANG_DEBUG_LOCATIONS only controls whether locations are printed in MLIR output
    print_debug_locations = os.environ.get("TTLANG_DEBUG_LOCATIONS", "0") == "1"
    if logical_kernels is not None and len(logical_kernels) != len(program.threads):
        raise ValueError(
            "logical kernel metadata must align one-to-one with program threads"
        )

    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        compiled_threads = []
        # Track which global tensor indices each thread uses (for building common_runtime_args)
        thread_tensor_indices = []
        # Collect source info for error formatting
        all_source_lines = {}
        all_source_files = {}

        # Track per-kernel line offsets for correct display
        kernel_line_offsets = {}
        noc_kernel_idx = 0

        for thread_index, compile_thread in enumerate(program.threads):
            try:
                ct = compile_thread(*program.args, **program.kwargs)
            except TTLangCompileError as e:
                # Thread-level error with embedded source location - use it
                raise type(e)(e.format()) from None
            except (ValueError, TypeError) as e:
                # Kernel-level error (no embedded location) - use kernel decorator
                formatted = format_python_error(
                    e, kernel_source_file, kernel_line_offset
                )
                raise type(e)(formatted) from None
            compiled_threads.append(ct)
            thread_tensor_indices.append(ct._tensor_accessor_global_indices)

            # Set TensorAccessor indexing attributes for C++ lowering
            base_cta = get_cb_count()
            ct.func_entry.attributes["ttl.base_cta_index"] = IntegerAttr.get(
                IntegerType.get_signless(32, ctx), base_cta
            )
            crta_indices = ct._tensor_accessor_global_indices
            ct.func_entry.attributes["ttl.crta_indices"] = ArrayAttr.get(
                [
                    IntegerAttr.get(IntegerType.get_signless(32, ctx), idx)
                    for idx in crta_indices
                ],
                ctx,
            )

            # Tag noc functions with their index so pipe semaphore allocation
            # and TTNN reader/writer role assignment can distinguish threads.
            if ct.kernel_type == "datamovement":
                ct.func_entry.attributes["ttl.noc_index"] = IntegerAttr.get(
                    IntegerType.get_signless(32, ctx), noc_kernel_idx
                )
                noc_kernel_idx += 1

            logical_kernel = (
                logical_kernels[thread_index] if logical_kernels is not None else None
            )
            if logical_kernel is not None:
                kind = _selector_kind(logical_kernel)
                ir_kind = {
                    KernelKind.COMPUTE: ttl_dialect.ir.LogicalKernelKind.Compute,
                    KernelKind.DATA_MOVEMENT: (
                        ttl_dialect.ir.LogicalKernelKind.DataMovement
                    ),
                }[kind]
                identity = None
                operation_identity = None
                implicit_role = None
                if isinstance(logical_kernel, Kernel):
                    identity = logical_kernel.identity
                    operation_identity = logical_kernel._operation_identity
                    implicit_role = _selector_implicit_role(logical_kernel)
                ct.func_entry.attributes[_ttl_ir.LOGICAL_KERNEL_ATTR] = (
                    ttl_dialect.LogicalKernelAttr.get(
                        ctx,
                        ir_kind,
                        identity,
                        operation_identity,
                        implicit_role,
                    )
                )

            # Collect source info for error reporting
            if hasattr(ct, "source_file") and hasattr(ct, "source_lines"):
                all_source_files[ct.name] = ct.source_file
                all_source_lines[ct.name] = ct.source_lines
            # Track per-kernel line offset
            if hasattr(ct, "line_offset"):
                kernel_line_offsets[ct.name] = ct.line_offset

        # Collect include paths from call_extern_func across all threads.
        opaque_include_paths = []
        for ct in compiled_threads:
            opaque_include_paths.extend(getattr(ct, "_opaque_include_paths", []))

        module = Module.create(loc)
        module.operation.attributes["ttl.launch_grid"] = ArrayAttr.get(
            [
                IntegerAttr.get(IntegerType.get_signless(64, ctx), dim)
                for dim in launch_grid
            ],
            ctx,
        )
        if target_arch is not None:
            module.operation.attributes["ttl.target_arch"] = ttcore.ir.ArchAttr.get(
                ctx, int(_TTCORE_ARCH_BY_DEVICE_NAME[target_arch])
            )

        # Insert standalone thread functions directly into module
        with InsertionPoint(module.body):
            for ct in compiled_threads:
                ct.func_entry.operation.detach_from_parent()
                module.body.append(ct.func_entry)

        initial_mlir_path = os.environ.get("TTLANG_INITIAL_MLIR")
        if initial_mlir_path:
            with open(initial_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED INITIAL TO {initial_mlir_path}")

        verify = True

        # fmt: off
        set_compute_config_pass = "ttl-set-compute-kernel-config"
        config_options = []
        if fp32_dest_acc_en is not None:
            config_options.append(
                "fp32-dest-acc-en="
                + ("enabled" if fp32_dest_acc_en else "disabled")
            )
        if dst_full_sync_en is not None:
            config_options.append(
                "dst-full-sync-en="
                + ("enabled" if dst_full_sync_en else "disabled")
            )
        config_options.append(
            f"reduce-full-fp32={int(compiler_options.reduce_full_fp32)}"
        )
        config_options.append(
            f"matmul-full-fp32={int(compiler_options.matmul_full_fp32)}"
        )
        config_options.append(
            f"enable-fpu-binary-ops={int(compiler_options.enable_fpu_binary_ops)}"
        )
        if config_options:
            set_compute_config_pass = (
                "ttl-set-compute-kernel-config{"
                + " ".join(config_options)
                + "}"
            )

        # NOTE: Pipeline pass ordering mirrors
        # lib/Dialect/TTL/Pipelines/TTLPipelines.cpp.
        assign_dst_pass = "ttl-assign-dst"

        compiler_dfbs_flag = int(compiler_options.compiler_dfbs)
        accumulation_strategy = compiler_options.accumulation_strategy
        pipe_batch_tiles = compiler_options.pipe_batch_tiles
        pipe_transport_options = [f"group-size={pipe_batch_tiles}"]
        if l1_budget_override > 0:
            pipe_transport_options.append(
                f"l1-budget-override={l1_budget_override}"
            )
        pipe_transport_pass = (
            "ttl-form-pipe-transports{" + " ".join(pipe_transport_options) + "}"
        )
        reuse_user_dfbs_flag = int(compiler_options.reuse_user_dfbs)
        unsafe_assume_allocation_groups_flag = int(
            compiler_options.unsafe_assume_dfb_allocation_groups
        )
        exact_coloring_search_limit = (
            compiler_options.dfb_exact_coloring_search_limit
        )
        tensor_recurrence_pipeline = (
            "ttl-form-accumulation-scopes{"
            f"strategy={accumulation_strategy}"
            "},"
            f"ttl-lower-accumulation-scopes{{strategy={accumulation_strategy}}},"
            "ttl-materialize-loop-state"
        )
        pipeline_passes = [
            f"func.func({tensor_recurrence_pipeline})",
            "func.func(ttl-insert-copy-wait)",
            "func.func(ttl-auto-sync)",
            "func.func(ttl-insert-accumulation-scopes{kind=dfb})",
            "func.func(ttl-lower-accumulation-scopes{kind=dfb})",
            "func.func(ttl-create-producer-compute)",
            f"func.func(ttl-insert-intermediate-dfbs{{enable={compiler_dfbs_flag}}})",
            "func.func(convert-ttl-to-compute)",
            "func.func(ttl-insert-cb-sync)",
            "ttl-verify-pipenet",
            pipe_transport_pass,
            "func.func(ttl-coalesce-dfb-acquires)",
            "ttl-finalize-dfb-indices{"
            f"reuse-user-dfbs={reuse_user_dfbs_flag} "
            "unsafe-assume-allocation-groups="
            f"{unsafe_assume_allocation_groups_flag} "
            f"exact-coloring-search-limit={exact_coloring_search_limit} "
            f"l1-budget-override={l1_budget_override}"
            "}",
            set_compute_config_pass,
            f"func.func({assign_dst_pass})",
        ]
        if compiler_options.maximize_dst:
            subblock_sync = "true" if compiler_options.subblock_sync else "false"
            strict_f32 = "true" if compiler_options.strict_f32_acc else "false"
            pipeline_passes.append(
                f"func.func(ttl-subblock-compute-for-dst{{subblock-sync={subblock_sync} strict-f32-acc={strict_f32}}})"
            )
        dst_acc_str = "true" if compiler_options.maximize_dst else "false"
        block_mm_str = "true" if compiler_options.use_block_matmul else "false"
        pipeline_passes.append(
            f"func.func(ttl-lower-to-loops{{dst-accumulation={dst_acc_str} use-block-matmul={block_mm_str}}})"
        )
        if compiler_options.maximize_dst:
            pipeline_passes.append("func.func(ttl-schedule-operations)")
        pipeline_passes.append("func.func(ttl-annotate-cb-associations)")
        pipeline_passes.append("ttl-verify-dfb-spsc")
        pipeline_passes.append("ttl-erase-pipenet-scopes")
        if l1_budget_override > 0:
            pipeline_passes.append(
                f"ttl-validate-cb-budget{{l1-budget-override={l1_budget_override}}}"
            )
        else:
            pipeline_passes.append("ttl-validate-cb-budget")
        # Add CB flow graph dump if auto-profiling or perf dump is enabled
        perf_dump = os.environ.get("TTLANG_PERF_DUMP") == "1"
        if perf_dump:
            # Remove stale outputs from previous runs
            for stale in ("/tmp/ttlang_cb_flow_graph.json",):
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass
        if perf_dump:
            pipeline_passes.append(
                'ttl-dump-cb-flow-graph{output="/tmp/ttlang_cb_flow_graph.json"}')
        if is_auto_profile_enabled():
            if "TTLANG_PROFILE_CSV" in os.environ:
                cb_flow_json = str(Path(os.environ["TTLANG_PROFILE_CSV"]).parent / "cb_flow_graph.json")
            else:
                tt_metal_home = os.environ.get("TT_METAL_HOME", "")
                if not tt_metal_home:
                    raise ValueError("TTLANG_AUTO_PROFILE=1 requires TT_METAL_HOME or TTLANG_PROFILE_CSV to be set")
                cb_flow_json = f"{tt_metal_home}/generated/profiler/.logs/cb_flow_graph.json"
            pipeline_passes.append(f'ttl-dump-cb-flow-graph{{output="{cb_flow_json}"}}')

        reduce_fp32_flag = int(compiler_options.reduce_full_fp32)
        pipe_computed_flag = int(compiler_options.pipe_computed_addresses)
        pipe_capacity_sync_flag = int(compiler_options.pipe_capacity_sync)
        pipe_global_semaphores_only_flag = int(
            compiler_options.pipe_global_semaphores_only
        )
        pipeline_passes += [
            "ttl-lower-dprint-to-emitc",
            (
                f"convert-ttl-to-ttkernel{{reduce-full-fp32={reduce_fp32_flag} "
                f"pipe-computed-addresses={pipe_computed_flag} "
                f"pipe-capacity-sync={pipe_capacity_sync_flag} "
                f"pipe-global-semaphores-only={pipe_global_semaphores_only_flag} "
                f"l1-budget-override={l1_budget_override}}}"
            ),
            "func.func(ttkernel-lower-scalar-fp-types)",
            "ttkernel-insert-inits",
            "ttkernel-insert-l1-accumulation",
        ]
        if compiler_options.combine_pack_tiles:
            pipeline_passes.append("func.func(ttkernel-combine-pack-tiles)")
        pipeline_passes += [
            "canonicalize",
            "cse",
            "lower-affine",
            "ttl-lower-signpost-to-emitc",
        ]
        if compiler_options.specialize_cores:
            pipeline_passes.append("ttkernel-specialize-and-annotate-dfb-use")
        pipeline_passes += [
            "convert-ttkernel-to-emitc",
            "symbol-dce",
        ]

        pipeline = ",".join(pipeline_passes)

        pipeline_str = f"builtin.module({pipeline})"
        # fmt: on
        pm = PassManager.parse(pipeline_str)
        pm.enable_verifier(verify)

        try:
            from ttl._mlir_libs._ttlang import enable_pretty_stack_traces

            enable_pretty_stack_traces(pm._CAPIPtr)
        except Exception:
            # Pretty stack traces are optional, silently continue if unavailable
            pass

        if os.environ.get("TTLANG_VERBOSE_PASSES"):
            from ttl.version import __version__

            print(f"ttlang {__version__}")
            print("Running custom pipeline:", pm)
            ctx.enable_multithreading(False)
            pm.enable_ir_printing(
                print_after_all=True,
                print_before_all=True,
                print_after_failure=True,
                enable_debug_info=True,
            )

        try:
            # Run the pass manager with error handling for source-aware diagnostics
            with ctx.attach_diagnostic_handler(_forward_mlir_warning):
                pm.run(module.operation)
        except Exception as e:
            error_msg = str(e)
            # Try to format error with source context
            # Use the first thread's source as fallback
            source_lines = None
            source_file = None
            if all_source_lines:
                first_thread = next(iter(all_source_lines.keys()))
                source_lines = all_source_lines[first_thread]
                source_file = all_source_files.get(first_thread)
            from ttl.version import __version__

            formatted = f"ttlang {__version__}\n{format_mlir_error(error_msg, source_lines, source_file)}"
            raise RuntimeError(formatted) from None

        final_mlir_path = os.environ.get("TTLANG_FINAL_MLIR")
        if final_mlir_path:
            with open(final_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED FINAL TO {final_mlir_path}")

        # Extract source lines for auto-profiling (use first thread's source)
        profile_source_lines = None
        if all_source_lines:
            first_thread = next(iter(all_source_lines.keys()))
            profile_source_lines = all_source_lines[first_thread]

        cb_configs = _resolve_dfb_configs(module)
        dfb_reconfiguration_plan = _extract_dfb_reconfiguration_plan(module, cb_configs)
        pipe_sync_semaphore_count = _extract_pipe_sync_semaphore_count(module)
        if pipe_sync_semaphore_count is None:
            raise RuntimeError(
                "compiled module is missing "
                f"{_ttl_ir.PIPE_SYNC_SEMAPHORE_COUNT_ATTR}"
            )
        dfb_reset_count = _extract_dfb_reset_count(module)
        pipe_sram_scratch_bytes = _extract_pipe_sram_scratch_bytes(module)
        pipe_global_semaphore_count = _extract_pipe_global_semaphore_count(module)

        # Compile to CompiledTTNNKernel for ttnn.generic_op.
        # `launch_grid` may be smaller than `grid` when grid="full" reduces
        # the launch to the PipeNet work extent; only core_ranges uses it.
        compiled_kernel = _compile_ttnn_kernel(
            module,
            args,
            launch_grid,
            num_outs,
            thread_tensor_indices,
            cb_configs,
            dfb_reconfiguration_plan=dfb_reconfiguration_plan,
            program_hash=program_hash,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            math_fidelity=math_fidelity,
            source_lines=profile_source_lines,
            all_source_lines=all_source_lines,
            kernel_line_offsets=kernel_line_offsets,
            num_pipe_sync_semaphores=pipe_sync_semaphore_count,
            num_dfb_resets=dfb_reset_count,
            pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
            num_pipe_global_semaphores=pipe_global_semaphore_count,
            opaque_include_paths=opaque_include_paths,
            target_arch=target_arch,
            operation_name=operation_name,
            runtime_resource_factory=runtime_resource_factory,
            runtime_resource_cache=runtime_resource_cache,
        )
        return compiled_kernel


def _canonical_tensor_args(
    function: Callable,
    args: tuple,
    kwargs: dict,
    *,
    expand_only_params=(),
) -> tuple:
    """Bind a call and return tensor arguments in signature order."""
    if expand_only_params:
        names = ", ".join(repr(name) for name in expand_only_params)
        raise ValueError(
            f"@ttl.operation {function.__name__!r} is expand-only because it has "
            f"DFB or PipeNet parameter(s): {names}; it cannot be called directly"
        )

    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    runtime_args = tuple(bound.arguments[name] for name in signature.parameters)
    for name, value in bound.arguments.items():
        if not is_ttnn_tensor(value):
            raise TypeError(
                f"@ttl.operation runtime argument {name!r} must be a TT-NN "
                f"tensor, got {type(value).__name__}"
            )
    return runtime_args


def _make_operation_wrapper(
    function: Callable,
    compile_callback: Callable,
    *,
    grid,
    fp32_dest_acc_en: Optional[bool],
    dst_full_sync_en: Optional[bool],
    math_fidelity: Optional[str],
    options: Optional[str],
    prepare_call: Optional[Callable] = None,
) -> Callable:
    """Build the shared top-level operation cache and execution wrapper."""
    kernel_id = random.getrandbits(64)
    cache: Dict[tuple, CompiledTTNNKernel] = {}
    cache_lock = threading.RLock()
    runtime_resource_cache = KernelRuntimeResourceCache()

    @functools.wraps(function)
    def _wrapper(*args, **kwargs):
        kwargs = dict(kwargs)
        opts_str = kwargs.pop("options", options)
        runtime_args = args
        grid_kwargs = kwargs
        if prepare_call is not None:
            runtime_args = prepare_call(args, kwargs)
            grid_kwargs = {}
        resolved_grid = _resolve_grid(grid, runtime_args, grid_kwargs)

        env_opts = os.environ.get("TTLANG_COMPILER_OPTIONS")
        if env_opts:
            opts_str = f"{opts_str or ''} {env_opts}".strip() or None
        compiler_options = CompilerOptions.from_string(opts_str).merge(
            CompilerOptions.from_argv()
        )
        target_arch = _device_target_arch(runtime_args)
        with cache_lock:
            l1_budget_override = _resolve_l1_budget(
                runtime_args, compiler_options, runtime_resource_cache
            )
            cache_key = _make_cache_key(
                runtime_args,
                resolved_grid=resolved_grid,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=dst_full_sync_en,
                math_fidelity=math_fidelity,
                target_arch=target_arch,
                compiler_options=compiler_options,
                l1_budget_override=l1_budget_override,
            )
            compiled_kernel = cache.get(cache_key)
            if compiled_kernel is None:
                compiled_kernel = compile_callback(
                    runtime_args,
                    kwargs,
                    resolved_grid,
                    hash((kernel_id, cache_key)),
                    target_arch,
                    compiler_options,
                    l1_budget_override,
                    runtime_resource_cache,
                )
                if compiled_kernel is not None:
                    cache[cache_key] = compiled_kernel

        if compiled_kernel is None or not _should_execute():
            return None

        result = compiled_kernel(*runtime_args)

        if is_auto_profile_enabled() and compiled_kernel.all_source_lines:
            _run_profiling_pipeline(
                runtime_args,
                compiled_kernel.all_source_lines,
                compiled_kernel.thread_to_kernel,
                compiled_kernel.kernel_line_offsets,
            )

        if os.environ.get("TTLANG_PERF_DUMP") == "1":
            _run_perf_dump(runtime_args, function.__name__)

        if is_signpost_profile_enabled():
            _run_signpost_profile(runtime_args)

        if os.environ.get("TTLANG_PERF_SERV") == "1":
            tt_metal_home = os.environ.get("TT_METAL_HOME", "")
            if not tt_metal_home:
                raise ValueError("TTLANG_PERF_SERV=1 requires TT_METAL_HOME")
            csv_path = (
                Path(tt_metal_home)
                / "generated"
                / "profiler"
                / ".logs"
                / "profile_log_device.csv"
            )
            if csv_path.exists():
                serve_trace(csv_path)
            del os.environ["TTLANG_PERF_SERV"]

        return result

    attach_runtime_resource_finalizer(_wrapper, runtime_resource_cache)
    return _wrapper


def _validate_operation_options(
    num_outs, memory_space, tiled, math_fidelity: Optional[str]
) -> None:
    if num_outs != 1:
        raise ValueError(f"num_outs must be 1, got {num_outs}")
    if memory_space not in SUPPORTED_MEMORY_SPACES:
        raise ValueError(
            f"Invalid memory_space: {memory_space!r}. "
            f"Must be one of: {', '.join(sorted(SUPPORTED_MEMORY_SPACES))}"
        )
    if not isinstance(tiled, bool):
        raise TypeError(f"tiled must be a boolean, got {type(tiled).__name__}")
    validate_math_fidelity(math_fidelity)


def pykernel_gen(
    grid: Optional[Union[tuple, Callable]] = None,
    indexing_maps: Optional[List[Callable]] = None,
    iterator_types: Optional[List[str]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    math_fidelity: Optional[str] = None,
    options: Optional[str] = None,
    runtime_resource_factory: Optional[Callable[..., ProgramRuntimeResources]] = None,
    _prepare_call: Optional[Callable] = None,
) -> Callable:
    """
    Decorator for generating TTL kernels from Python functions.

    This decorator compiles Python functions into TTL dialect operations,
    handling thread compilation, stream creation, and pipeline execution.
    Kernels are compiled to C++ for execution via ttnn.generic_op.

    Args:
        grid: Grid dimensions as tuple (e.g., (2, 2)) or callable
        indexing_maps: List of lambda functions for indexing (optional)
        iterator_types: List of iterator types ("parallel", "reduction")
        num_outs: Number of output arguments
        memory_space: "L1" or "DRAM"
        tiled: Whether to use tiled layout
        fp32_dest_acc_en: Optional override for fp32_dest_acc_en
        dst_full_sync_en: Optional override for dst_full_sync_en
        math_fidelity: Optional TTNN compute math fidelity
        options: Compiler option string (e.g., "--no-ttl-maximize-dst")
        runtime_resource_factory: Optional per-invocation resource callback

    Returns:
        Decorated function that compiles and executes the kernel

    Raises:
        ValueError: If required parameters or compute configuration are invalid
    """
    if grid is None:
        raise ValueError("grid parameter is required")
    _validate_operation_options(num_outs, memory_space, tiled, math_fidelity)
    if iterator_types is not None and indexing_maps is None:
        raise ValueError("indexing_maps must be set when iterator_types is set")

    if indexing_maps is None:
        indexing_maps = []

    if indexing_maps:
        for indexing_map in indexing_maps:
            num_dims = list(tuple(inspect.signature(indexing_map).parameters))
            if iterator_types is not None:
                if num_dims != len(iterator_types):
                    raise ValueError(
                        f"Number of dimensions ({num_dims}) must match iterator_types length ({len(iterator_types)})"
                    )

    if iterator_types is None:
        iterator_types = []

    def _decorator(f):
        _bind_kernel_declarations(
            _captured_kernel_declarations(f), _operation_identity(f)
        )

        def _compile_explicit(
            runtime_args,
            runtime_kwargs,
            resolved_grid,
            program_hash,
            target_arch,
            compiler_options,
            l1_budget_override,
            runtime_resource_cache,
        ):
            compile_kwargs = runtime_kwargs
            if _prepare_call is not None:
                compile_kwargs = {}
            return _compile_kernel(
                f,
                runtime_args,
                compile_kwargs,
                resolved_grid,
                indexing_maps,
                iterator_types,
                num_outs,
                memory_space,
                tiled,
                program_hash,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=dst_full_sync_en,
                math_fidelity=math_fidelity,
                target_arch=target_arch,
                compiler_options=compiler_options,
                l1_budget_override=l1_budget_override,
                runtime_resource_factory=runtime_resource_factory,
                runtime_resource_cache=runtime_resource_cache,
            )

        return _make_operation_wrapper(
            f,
            _compile_explicit,
            grid=grid,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            math_fidelity=math_fidelity,
            options=options,
            prepare_call=_prepare_call,
        )

    return _decorator


# Alias for backward compatibility
operation = pykernel_gen


__all__ = [
    "pykernel_gen",
    "operation",
    "Program",
    "compute",
    "datamovement",
    "TensorBlock",
    "DataflowBuffer",
    "CircularBuffer",
    "CopyTransferHandler",
    "ReceiveRequest",
    "ReadyReceive",
    "copy",
    "wait_any",
    "CompiledTTNNKernel",
]
