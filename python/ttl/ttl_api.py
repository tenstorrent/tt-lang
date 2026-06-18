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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

ttnn = None  # Lazy-loaded on first access via _ensure_ttnn()


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
from ttl.dialects import ttkernel
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
from ._src.ttl_ast import TTLGenericCompiler
from .dataflow_buffer import (
    CircularBuffer,
    CompilerAllocatedDFBConfig,
    DataflowBuffer,
    get_cb_count,
)
from .pipe import Pipe, PipeNet
from .constants import SUPPORTED_MEMORY_SPACES
from .diagnostics import (
    TTLangCompileError,
    find_variable_assignment,
    format_mlir_error,
    format_python_error,
)
from .dtype_utils import (
    is_ttnn_tensor,
    tile_bytes_from_dtype,
    torch_dtype_to_ttnn_datatype,
)
from .kernel_runner import (
    KernelSpec,
    get_min_remaining_l1_for_device,
    run_kernel_on_device,
    emit_runner_file,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .compiler_options import CompilerOptions
from .ttl_utils import get_thread_type_string

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


def _get_tensor_cache_info(tensor) -> tuple:
    """Extract cache-relevant info from a tensor: (shape, dtype, memory_space, layout)."""
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    mem_config = tensor.memory_config()
    memory_space = (
        str(mem_config.buffer_type) if hasattr(mem_config, "buffer_type") else "unknown"
    )
    layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
    return (shape, dtype, memory_space, layout)


def _make_cache_key(
    args: tuple,
    fp32_dest_acc_en: Optional[bool],
    dst_full_sync_en: Optional[bool],
    target_arch: Optional[str],
    compiler_options: CompilerOptions = CompilerOptions(),
) -> tuple:
    """Create cache key from tensor properties and runtime compute config parameters."""
    tensor_key = tuple(
        _get_tensor_cache_info(arg) for arg in args if is_ttnn_tensor(arg)
    )
    # Include mesh shape so that single-device and multi-device compilations
    # with different shard shapes don't collide in the cache.
    mesh_key = None
    for arg in args:
        if is_ttnn_tensor(arg) and _is_mesh_tensor(arg):
            mesh_key = tuple(arg.device().shape)
            break
    return (
        tensor_key,
        mesh_key,
        fp32_dest_acc_en,
        dst_full_sync_en,
        target_arch,
        compiler_options,
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


def _same_device(a, b) -> bool:
    """Return True when *a* and *b* refer to the same TTNN device."""
    if a is b:
        return True
    a_id = getattr(a, "id", None)
    b_id = getattr(b, "id", None)
    if callable(a_id) and callable(b_id):
        return a_id() == b_id()
    return False


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


def _detect_device_arch(device) -> Optional[str]:
    """Return a normalized architecture string from a TTNN device if present."""
    arch_attrs = (
        "arch",
        "architecture",
        "chip_type",
        "device_type",
        "_arch",
        "_architecture",
    )
    for attr in arch_attrs:
        # Properties on device handles may raise for reasons other than
        # AttributeError (e.g., closed handle); guard both attribute access
        # and the optional method call.
        try:
            arch_value = getattr(device, attr)
        except Exception:
            continue
        if callable(arch_value):
            try:
                arch_value = arch_value()
            except Exception:
                continue
        return str(arch_value).lower().rsplit(".", maxsplit=1)[-1]
    return None


def _device_target_arch(args) -> Optional[str]:
    """Return the first detected tensor device architecture, or None."""
    for arg in args:
        if not is_ttnn_tensor(arg) or not hasattr(arg, "device"):
            continue
        device = arg.device()
        if device is None:
            continue
        arch = _detect_device_arch(device)
        if arch is None:
            continue
        return arch
    return None


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
        cb_configs=None,
        program_hash=None,
        source_lines=None,
        all_source_lines=None,
        thread_to_kernel=None,
        kernel_line_offsets=None,
        num_pipe_sync_semaphores=0,
        pipe_sram_scratch_bytes=0,
        num_pipe_global_semaphores=0,
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
            cb_configs: List of (shape, block_count) tuples for each CB, indexed by cb_index
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
                PipeNet ready counters used by this kernel.
        """
        self.kernel_paths = kernel_paths
        self.kernel_configs = kernel_configs
        self.kernel_arg_specs = kernel_arg_specs
        self.num_tensors = num_tensors
        self.core_ranges = core_ranges
        self.kernel_tensor_indices = kernel_tensor_indices
        self.cb_configs = cb_configs or []
        self.program_hash = program_hash
        self.source_lines = source_lines
        self.all_source_lines = all_source_lines or {}
        self.thread_to_kernel = thread_to_kernel or {}
        self.kernel_line_offsets = kernel_line_offsets or {}
        self.num_pipe_sync_semaphores = num_pipe_sync_semaphores
        self.pipe_sram_scratch_bytes = pipe_sram_scratch_bytes
        self.num_pipe_global_semaphores = num_pipe_global_semaphores
        self._pipe_global_semaphore_lifetime = []

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
            )
            kernel_specs.append(spec)

        # Use shared kernel execution logic.
        return run_kernel_on_device(
            kernel_specs=kernel_specs,
            tensors=list(args),
            cb_configs=self.cb_configs,
            core_ranges=self.core_ranges,
            program_hash=self.program_hash,
            num_pipe_sync_semaphores=self.num_pipe_sync_semaphores,
            pipe_sram_scratch_bytes=self.pipe_sram_scratch_bytes,
            num_pipe_global_semaphores=self.num_pipe_global_semaphores,
            pipe_global_semaphore_lifetime=self._pipe_global_semaphore_lifetime,
        )


def _write_kernel_to_tmp(name: str, source: str) -> str:
    """Write kernel source to /tmp and return the file path."""
    import hashlib
    import os

    content_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    user = os.environ.get("USER", "default")
    path = f"/tmp/{user}/ttlang_kernel_{name}_{content_hash}.cpp"
    os.makedirs(f"/tmp/{user}", exist_ok=True)
    with open(path, "w") as f:
        f.write(source)
    print(f"=== {name} kernel written to {path} ===")
    print(source)
    print("=" * 60)
    return path


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


def _get_kernel_bool_attr(module, kernel_name: str, attr_name: str) -> bool:
    """Read a boolean func.func attribute from a compiled kernel."""
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get(attr_name, None)
    if attr is None:
        return False
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
    """Read a `DenseI32ArrayAttr` func.func attribute as a list of ints.

    Returns an empty list when the attribute is missing. Used by the runtime
    bridge to consume the per-CB UnpackToDestFp32 selection emitted by
    `ttl-set-compute-kernel-config`.
    """
    operation = _lookup_kernel_func_op(module, kernel_name)
    attr = operation.attributes.get(attr_name, None)
    if attr is None:
        return []
    if not isinstance(attr, DenseI32ArrayAttr):
        raise ValueError(
            f"Expected DenseI32ArrayAttr for '{attr_name}' on kernel "
            f"'{kernel_name}', got {attr}"
        )
    return list(attr)


def _compile_ttnn_kernel(
    module,
    args,
    grid,
    num_outs,
    thread_tensor_indices,
    cb_configs=None,
    program_hash=None,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    verbose=True,
    source_lines=None,
    all_source_lines=None,
    kernel_line_offsets=None,
    num_pipe_sync_semaphores: int = 0,
    pipe_sram_scratch_bytes: int = 0,
    num_pipe_global_semaphores: int = 0,
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

    # Validate kernel count: for now we must have exactly 3 kernels (1 compute + 2 data movement).
    # Each core has only 2 NOCs, so more than 2 DM kernels causes NOC conflicts.
    # TODO: in the future we should figure out how to map arbitrary kernels.
    if len(kernel_info) != 3:
        compute_count = sum(1 for _, t in kernel_info if t == "compute")
        dm_count = sum(1 for _, t in kernel_info if t == "noc")
        raise ValueError(
            f"TTNN interop requires exactly 3 kernels (1 compute + 2 data movement), "
            f"got {len(kernel_info)} kernels ({compute_count} compute, {dm_count} data movement). "
            f"Each core has only 2 NOCs, so more than 2 DM kernels causes NOC conflicts."
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
    noc_kernel_idx = 0
    kernel_config_attrs = {
        name: {
            "fp32_dest_acc_en": _get_kernel_bool_attr(module, name, "fp32_dest_acc_en"),
            "dst_full_sync_en": _get_kernel_bool_attr(module, name, "dst_full_sync_en"),
            "unpack_to_dest_fp32": _get_kernel_i32_array_attr(
                module, name, "ttl.unpack_to_dest_fp32"
            ),
        }
        for name, _ in kernel_info
    }

    # Build thread-to-kernel mapping for profiling
    # Maps RISC thread names to kernel names
    thread_to_kernel = {}

    for name, thread_type in kernel_info:
        cpp_source = ttkernel_to_cpp_by_name(module, name)
        kernel_path = _write_kernel_to_tmp(name, cpp_source)
        kernel_paths.append((kernel_path, thread_type))

        if thread_type == "compute":
            config = ttnn.ComputeConfigDescriptor()
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
            if noc_kernel_idx == 0:
                config = ttnn.ReaderConfigDescriptor()
                thread_to_kernel["NCRISC"] = name  # Reader
            else:
                config = ttnn.WriterConfigDescriptor()
                thread_to_kernel["BRISC"] = name  # Writer
            noc_kernel_idx += 1
        else:
            config = ttnn.ReaderConfigDescriptor()
        kernel_configs.append(config)

        # Extract runtime args from kernel's arg_spec attribute
        arg_spec = get_ttkernel_arg_spec(module, name)
        if arg_spec is not None:
            arg_spec = ttkernel.ir.ArgSpecAttr.maybe_downcast(arg_spec)
            kernel_arg_specs.append(arg_spec.rt_args if arg_spec else [])
        else:
            kernel_arg_specs.append([])

    compiled_kernel = CompiledTTNNKernel(
        kernel_paths=kernel_paths,
        kernel_configs=kernel_configs,
        kernel_arg_specs=kernel_arg_specs,
        num_tensors=len(args),
        core_ranges=core_ranges,
        kernel_tensor_indices=thread_tensor_indices,
        cb_configs=cb_configs,
        program_hash=program_hash,
        source_lines=source_lines,
        all_source_lines=all_source_lines,
        thread_to_kernel=thread_to_kernel,
        kernel_line_offsets=kernel_line_offsets,
        num_pipe_sync_semaphores=num_pipe_sync_semaphores,
        pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
        num_pipe_global_semaphores=num_pipe_global_semaphores,
    )

    if verbose:
        print(f"\nCompiled kernel ready (compiled {len(kernel_paths)} threads)")
        print("=" * 60)

    emit_runner_path = os.environ.get("TTLANG_EMIT_RUNNER")
    if emit_runner_path:
        kernel_specs_for_emit = []
        for kernel_idx, (kernel_path, thread_type) in enumerate(kernel_paths):
            tensor_indices = thread_tensor_indices[kernel_idx]
            spec = KernelSpec(
                path=kernel_path,
                thread_type=thread_type,
                tensor_indices=tensor_indices,
                config=kernel_configs[kernel_idx],
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
            kernel_name="ttlang_kernel",
            num_pipe_sync_semaphores=num_pipe_sync_semaphores,
            pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
            num_pipe_global_semaphores=num_pipe_global_semaphores,
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
    from ._pipenets import OperationPipeNets
    from .pipe import _pipe_to_pipe_use

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

    graph = OperationPipeNets()
    for net in seen.values():
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
) -> Dict[str, Union[int, DataflowBuffer, Pipe]]:
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
        if isinstance(val, (int, float)):
            return val
        elif is_ttnn_tensor(val):
            return val
        elif isinstance(val, DataflowBuffer):
            return val
        elif isinstance(val, Pipe):
            return val
        elif isinstance(val, PipeNet):
            return val
        else:
            raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }


def _collect_cb_configs(threads):
    """Extract DataflowBuffer objects from thread closures, indexed by dfb index.

    Returns a list of DataflowBuffer objects indexed by dfb index. Each DFB has
    shape, block_count, tensor (for dtype), and _cb_index attributes.
    """
    cb_configs_dict = {}
    for thread_fn in threads:
        wrapped = getattr(thread_fn, "__wrapped__", None)
        closure = getattr(wrapped, "__closure__", None) if wrapped else None
        if not closure:
            continue
        for cell in closure:
            val = cell.cell_contents
            if isinstance(val, DataflowBuffer):
                cb_configs_dict[val._cb_index] = val

    if not cb_configs_dict:
        return []
    max_idx = max(cb_configs_dict.keys())
    return [cb_configs_dict.get(i) for i in range(max_idx + 1)]


# Map MLIR element type names to ttnn-compatible data format names.
# Keyed by exact MLIR type mnemonic (no substring matching).
_MLIR_TYPE_TO_FORMAT = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "i32": "int32",
    "ui32": "uint32",
    "ui16": "uint16",
}


def _parse_mlir_element_type(type_str: str) -> str:
    """Extract the base data format name from an MLIR TypeAttr string.

    The TypeAttr prints as e.g. "bf16" or "!ttcore.tile<32x32, bf16>".
    This function extracts the trailing type mnemonic and maps it to a
    ttnn-compatible format name.
    """
    # For compound types like "!ttcore.tile<32x32, bf16>", extract the
    # type after the last comma. For bare types like "bf16", use as-is.
    token = type_str.strip()
    if "," in token:
        token = token.rsplit(",", 1)[1].strip().rstrip(">").strip()
    fmt = _MLIR_TYPE_TO_FORMAT.get(token)
    if fmt is not None:
        return fmt
    raise ValueError(
        f"Unrecognized MLIR element type '{token}' (from '{type_str}'). "
        f"Known types: {list(_MLIR_TYPE_TO_FORMAT.keys())}"
    )


def _extract_compiler_allocated_dfbs(module):
    """Read ttl.compiler_allocated_dfbs module attribute.

    Returns an empty list when the attribute is absent (no compiler-allocated
    DFBs). Each entry is a DictionaryAttr with dfb_index, num_tiles,
    element_type, and block_count.
    """
    attr = module.operation.attributes.get("ttl.compiler_allocated_dfbs", None)
    if attr is None:
        return []

    configs = []
    for entry in attr:
        dfb_index = int(entry["dfb_index"])
        num_tiles = int(entry["num_tiles"])
        block_count = int(entry["block_count"])
        data_format = _parse_mlir_element_type(str(entry["element_type"]))

        configs.append(
            CompilerAllocatedDFBConfig(
                dfb_index=dfb_index,
                num_tiles=num_tiles,
                data_format=data_format,
                block_count=block_count,
            )
        )
    return configs


def _extract_pipe_sync_semaphore_count(module) -> Optional[int]:
    """Read the semaphore count selected by pipe lowering."""
    attr = module.operation.attributes.get(_ttl_ir.PIPE_SYNC_SEMAPHORE_COUNT_ATTR, None)
    if attr is None:
        return None
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


def _merge_dfb_configs(cb_configs, compiler_allocated_dfbs):
    """Merge compiler-allocated DFBs into the CB config list.

    Extends cb_configs to cover all DFB indices. Compiler-allocated DFBs
    are placed at their dfb_index positions.
    """
    if not compiler_allocated_dfbs:
        return cb_configs

    user_max = len(cb_configs) - 1 if cb_configs else -1
    alloc_max = max(dfb.dfb_index for dfb in compiler_allocated_dfbs)
    total = max(user_max, alloc_max) + 1

    merged = list(cb_configs) + [None] * (total - len(cb_configs))
    for dfb in compiler_allocated_dfbs:
        if merged[dfb.dfb_index] is not None:
            raise ValueError(
                f"Compiler-allocated DFB index {dfb.dfb_index} collides with "
                f"an existing DFB."
            )
        merged[dfb.dfb_index] = dfb
    return merged


def _compile(
    kernel_type: Optional[str] = None,
    verbose: bool = False,
) -> Callable:
    """
    Internal decorator for compiling kernel threads.

    Args:
        kernel_type: Type of kernel ("compute" or "datamovement")
        verbose: Enable verbose compilation output

    Returns:
        Decorator function for kernel compilation
    """

    def _decorator(f):
        # Capture source file at decoration time
        try:
            source_file = inspect.getfile(f)
        except (TypeError, OSError):
            source_file = "<unknown>"

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
            line_offset = kwargs.get("_line_offset", 0)

            b = TTLGenericCompiler(
                f.__name__,
                kernel_type,
                _collect_captures(f),
                *args,
                _globals=f.__globals__,
                **kwargs,
            )

            if verbose:
                print(ast.dump(m, indent=4) + "\n")

            b.visit(m)

            if verbose:
                print(b.module)

            try:
                b.module.operation.verify()
            except Exception as e:
                formatted = format_mlir_error(str(e), source_lines, source_file)
                raise RuntimeError(formatted) from None

            return b

        _wrapper._decorator_name = kernel_type + "_thread"
        _wrapper._source_file = source_file
        # Register thread for automatic collection
        _register_thread(_wrapper)
        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def compute(verbose: bool = False) -> Callable:
    """
    Decorator for compute thread functions.

    Compute threads execute on Tensix cores and perform mathematical operations.

    Args:
        verbose: Enable verbose compilation output

    Returns:
        Decorator for compute kernel compilation
    """
    return _compile(
        kernel_type="compute",
        verbose=verbose,
    )


def datamovement(verbose: bool = False) -> Callable:
    """
    Decorator for data movement thread functions.

    Data movement threads handle DMA operations between memory hierarchies.

    Args:
        verbose: Enable verbose compilation output

    Returns:
        Decorator for data movement kernel compilation
    """
    return _compile(
        kernel_type="datamovement",
        verbose=verbose,
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
    target_arch: Optional[str] = None,
    compiler_options: CompilerOptions = CompilerOptions(),
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
        target_arch: Optional TT device architecture for target-specific lowering
        compiler_options: Compiler pipeline options

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

    l1_budget_override = compiler_options.l1_budget
    if l1_budget_override == 0 and has_ttnn_tensors:
        try:
            device = _require_device(args)
            l1_budget_override = get_min_remaining_l1_for_device(device)
        except ValueError:
            pass

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
    f(*compile_args, **kwargs)
    threads = _get_registered_threads()

    if not threads:
        raise ValueError(
            "No threads found. Define at least one @ttl.compute() or "
            "@ttl.datamovement() function inside your kernel."
        )

    pipenets = _build_operation_pipenets(f, threads)

    launch_grid = grid

    cb_configs = _collect_cb_configs(threads)

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

    # Always generate source locations for error messages
    # TTLANG_DEBUG_LOCATIONS only controls whether locations are printed in MLIR output
    print_debug_locations = os.environ.get("TTLANG_DEBUG_LOCATIONS", "0") == "1"

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

        for compile_thread in program.threads:
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

            # Tag noc functions with their index so pipe semaphore
            # allocation can distinguish threads.
            if ct.kernel_type == "datamovement":
                ct.func_entry.attributes["ttl.noc_index"] = IntegerAttr.get(
                    IntegerType.get_signless(32, ctx), noc_kernel_idx
                )
                noc_kernel_idx += 1

            # Collect source info for error reporting
            if hasattr(ct, "source_file") and hasattr(ct, "source_lines"):
                all_source_files[ct.name] = ct.source_file
                all_source_lines[ct.name] = ct.source_lines
            # Track per-kernel line offset
            if hasattr(ct, "line_offset"):
                kernel_line_offsets[ct.name] = ct.line_offset

        module = Module.create(loc)
        module.operation.attributes["ttl.launch_grid"] = ArrayAttr.get(
            [
                IntegerAttr.get(IntegerType.get_signless(64, ctx), dim)
                for dim in launch_grid
            ],
            ctx,
        )
        if target_arch is not None:
            module.operation.attributes["ttl.target_arch"] = StringAttr.get(target_arch)

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
        set_compute_config_pass = "func.func(ttl-set-compute-kernel-config)"
        config_options = []
        if fp32_dest_acc_en is not None:
            config_options.append(
                f"fp32-dest-acc-en={1 if fp32_dest_acc_en else 0}"
            )
        if dst_full_sync_en is not None:
            config_options.append(
                f"dst-full-sync-en={1 if dst_full_sync_en else 0}"
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
                "func.func(ttl-set-compute-kernel-config{"
                + " ".join(config_options)
                + "})"
            )

        # NOTE: Pipeline pass ordering is mirrored in
        # test/me2e/builder/pipeline.py and lib/Dialect/TTL/Pipelines/TTLPipelines.cpp.
        assign_dst_pass = "ttl-assign-dst"

        compiler_dfbs_flag = int(compiler_options.compiler_dfbs)
        pipeline_passes = [
            f"func.func(ttl-insert-intermediate-dfbs{{enable={compiler_dfbs_flag}}})",
            "func.func(ttl-insert-copy-wait)",
            "func.func(ttl-auto-sync)",
            "func.func(ttl-annotate-l1-acc-loops)",
            "func.func(convert-ttl-to-compute)",
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
        pipeline_passes.append("ttl-finalize-dfb-indices")
        pipeline_passes.append("func.func(ttl-annotate-cb-associations)")
        pipeline_passes.append("ttl-verify-pipenet-guards")
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
        pipeline_passes += [
            "ttl-lower-dprint-to-emitc",
            f"convert-ttl-to-ttkernel{{reduce-full-fp32={reduce_fp32_flag}}}",
            "ttl-lower-scalar-cmpf",
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
            "convert-ttkernel-to-emitc",
            "symbol-dce",
        ]

        pipeline = ",".join(pipeline_passes)

        pipeline_str = f"builtin.module({pipeline})"
        # fmt: on
        pm = PassManager.parse(pipeline_str)
        pm.enable_verifier(verify)

        try:
            from ttl._mlir_libs._ttmlir import enable_pretty_stack_traces

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

        # Merge compiler-allocated DFBs into the CB config list.
        compiler_allocated_dfbs = _extract_compiler_allocated_dfbs(module)
        cb_configs = _merge_dfb_configs(cb_configs, compiler_allocated_dfbs)
        pipe_sync_semaphore_count = _extract_pipe_sync_semaphore_count(module)
        if pipe_sync_semaphore_count is None:
            raise RuntimeError(
                "compiled module is missing "
                f"{_ttl_ir.PIPE_SYNC_SEMAPHORE_COUNT_ATTR}"
            )
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
            program_hash=program_hash,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            source_lines=profile_source_lines,
            all_source_lines=all_source_lines,
            kernel_line_offsets=kernel_line_offsets,
            num_pipe_sync_semaphores=pipe_sync_semaphore_count,
            pipe_sram_scratch_bytes=pipe_sram_scratch_bytes,
            num_pipe_global_semaphores=pipe_global_semaphore_count,
        )
        return compiled_kernel


def pykernel_gen(
    grid: Optional[Union[tuple, Callable]] = None,
    indexing_maps: Optional[List[Callable]] = None,
    iterator_types: Optional[List[str]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    options: Optional[str] = None,
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
        options: Compiler option string (e.g., "--no-ttl-maximize-dst")

    Returns:
        Decorated function that compiles and executes the kernel

    Raises:
        AssertionError: If required parameters are missing or invalid
    """
    if grid is None:
        raise ValueError("grid parameter is required")
    if num_outs != 1:
        raise ValueError(f"num_outs must be 1, got {num_outs}")
    if memory_space not in SUPPORTED_MEMORY_SPACES:
        raise ValueError(
            f"Invalid memory_space: {memory_space!r}. "
            f"Must be one of: {', '.join(sorted(SUPPORTED_MEMORY_SPACES))}"
        )
    if not isinstance(tiled, bool):
        raise TypeError(f"tiled must be a boolean, got {type(tiled).__name__}")
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
        # Per-kernel state: random ID and cache
        kernel_id = random.getrandbits(64)
        cache: Dict[tuple, CompiledTTNNKernel] = {}

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            resolved_grid = _resolve_grid(grid, args, kwargs)
            fp32_override = fp32_dest_acc_en
            dst_sync_override = dst_full_sync_en

            # Extract runtime options (allow per-call override via kwarg).
            # Priority: sys.argv > env var > decorator options=
            # Env var is appended to decorator string (later tokens win),
            # then sys.argv is merged on top as the highest-priority override.
            opts_str = kwargs.pop("options", options)
            env_opts = os.environ.get("TTLANG_COMPILER_OPTIONS")
            if env_opts:
                # Env var tokens appended after explicit options.
                opts_str = f"{opts_str or ''} {env_opts}".strip() or None
            base = CompilerOptions.from_string(opts_str)
            argv_overrides = CompilerOptions.from_argv()
            compiler_options = base.merge(argv_overrides)
            target_arch = _device_target_arch(args)

            # Build cache key from tensor properties
            cache_key = _make_cache_key(
                args,
                # Runtime options:
                fp32_dest_acc_en=fp32_override,
                dst_full_sync_en=dst_sync_override,
                target_arch=target_arch,
                compiler_options=compiler_options,
            )

            # Check cache for previously compiled kernel
            if cache_key in cache:
                compiled_kernel = cache[cache_key]
            else:
                # Compute program_hash for tt-metal cache
                program_hash = hash((kernel_id, cache_key))

                # Compile kernel
                compiled_kernel = _compile_kernel(
                    f,
                    args,
                    kwargs,
                    resolved_grid,
                    indexing_maps,
                    iterator_types,
                    num_outs,
                    memory_space,
                    tiled,
                    program_hash,
                    fp32_dest_acc_en=fp32_override,
                    dst_full_sync_en=dst_sync_override,
                    target_arch=target_arch,
                    compiler_options=compiler_options,
                )

                if compiled_kernel is not None:
                    cache[cache_key] = compiled_kernel

            # Execute (unless compile-only mode)
            if compiled_kernel is not None and _should_execute():
                result = compiled_kernel(*args)

                # Run auto-profiling after execution
                if is_auto_profile_enabled() and compiled_kernel.all_source_lines:
                    _run_profiling_pipeline(
                        args,
                        compiled_kernel.all_source_lines,
                        compiled_kernel.thread_to_kernel,
                        compiled_kernel.kernel_line_offsets,
                    )

                if os.environ.get("TTLANG_PERF_DUMP") == "1":
                    _run_perf_dump(args, f.__name__)

                if is_signpost_profile_enabled():
                    _run_signpost_profile(args)

                # Serve profiler data as Perfetto trace (runs last,
                # after other profilers have dumped their data)
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

        return _wrapper

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
    "copy",
    "CompiledTTNNKernel",
]
