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

try:
    import ttnn
except (ModuleNotFoundError, ImportError):
    ttnn = None

import ttl._mlir_libs._ttlang  # Register tt-lang passes
from pykernel._src.utils import _cleanup_source_code
from ttmlir.dialects import ttkernel
from ttmlir.ir import *
from ttmlir.passes import (
    get_ttkernel_arg_spec,
    get_ttkernel_names,
    ttkernel_to_cpp_by_name,
)
from ttmlir.passmanager import PassManager

from ._src.auto_profile import (
    build_cb_wait_to_dma_map,
    build_dma_producer_to_cb_map,
    get_line_mapper,
    is_auto_profile_enabled,
    load_cb_flow_graph,
    parse_device_profile_csv,
    print_profile_report,
)
from ._src.tensor_registry import (
    get_tensor_global_index,
    get_tensor_source,
    register_tensor_name,
    register_tensor_source,
)
from ._src.ttl_ast import TTLGenericCompiler
from .circular_buffer import CircularBuffer, get_cb_count
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
    run_kernel_on_device,
)
from .operators import CopyTransferHandler, TensorBlock, copy
from .ttl_utils import get_thread_type_string
from .config import HAS_TT_DEVICE

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
) -> tuple:
    """Create cache key from tensor properties and runtime compute config parameters."""
    tensor_key = tuple(
        _get_tensor_cache_info(arg) for arg in args if is_ttnn_tensor(arg)
    )
    return (tensor_key, fp32_dest_acc_en, dst_full_sync_en)


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


def _is_interleaved_tensor(tensor) -> bool:
    """Check if a ttnn tensor has interleaved memory layout."""
    mem_config = tensor.memory_config()
    if hasattr(mem_config, "memory_layout"):
        return "INTERLEAVED" in str(mem_config.memory_layout)
    return False


def _has_float32_args(args) -> bool:
    """
    Check if any input tensor uses float32 dtype.

    Inspects the tensor arguments to detect float32. This is used to
    automatically enable fp32_dest_acc_en configuration for compute kernels.

    Args:
        args: List of tensor arguments (torch or ttnn)

    Returns:
        True if any tensor uses float32 dtype, False otherwise
    """
    try:
        for tensor in args:
            if tensor is None:
                continue

            # Check ttnn tensor
            if is_ttnn_tensor(tensor):
                tensor_dtype = tensor.dtype
                # ttnn.float32
                if (
                    hasattr(tensor_dtype, "name")
                    and "float32" in str(tensor_dtype.name).lower()
                ):
                    return True
                if "float32" in str(tensor_dtype).lower():
                    return True
            # Check torch tensor
            elif hasattr(tensor, "dtype"):
                import torch

                if tensor.dtype == torch.float32:
                    return True
    except (AttributeError, TypeError, ImportError):
        pass

    return False


def _resolve_grid(grid, args, kwargs):
    """Resolve grid, evaluating callable or 'auto' if needed."""
    if callable(grid):
        return grid(*args, **kwargs)
    if grid == "auto":
        for arg in args:
            if is_ttnn_tensor(arg) and hasattr(arg, "device"):
                device = arg.device()
                device_grid = device.compute_with_storage_grid_size()
                return (device_grid.x, device_grid.y)
        raise ValueError(
            "grid='auto' requires at least one ttnn tensor argument "
            "to determine device compute grid"
        )
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
            cb_configs: List of (shape, buffer_factor) tuples for each CB, indexed by cb_index
            program_hash: Hash for tt-metal program cache
            source_lines: Source code lines for auto-profiling reports (deprecated)
            all_source_lines: Dict mapping kernel name to source lines
            thread_to_kernel: Dict mapping RISC thread name to kernel name
            kernel_line_offsets: Dict mapping kernel name to line offset
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

    def __call__(self, *args):
        """Execute the kernel with the given tensors."""
        if len(args) != self.num_tensors:
            raise ValueError(f"Expected {self.num_tensors} tensors, got {len(args)}")

        # Validate grid against device's compute grid.
        device = args[0].device()
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
        )


def _write_kernel_to_tmp(name: str, source: str) -> str:
    """Write kernel source to /tmp and return the file path."""
    import hashlib
    import re
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
):
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.

    Args:
        module: MLIR module after D2M pipeline (with EmitC kernels)
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

    # Validate TTNN tensors - must be interleaved (L1 or DRAM) and tilized
    for i, arg in enumerate(args):
        if is_ttnn_tensor(arg):
            mem_space = _detect_memory_space_from_tensor(arg, "unknown")
            if mem_space not in ("L1", "DRAM"):
                raise ValueError(
                    f"TTNN interop requires L1 or DRAM memory space, but tensor {i} is in {mem_space}."
                )
            if not _is_interleaved_tensor(arg):
                raise ValueError(
                    f"TTNN interop requires interleaved tensors, but tensor {i} is not. "
                    f"Use ttnn.DRAM_MEMORY_CONFIG or ttnn.L1_MEMORY_CONFIG for interleaved tensors."
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

    # Check if input args use f32 to auto-configure compute kernels
    has_f32 = _has_float32_args(args)

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
            if dst_full_sync_en is not None:
                config.dst_full_sync_en = dst_full_sync_en
            if fp32_dest_acc_en is None and has_f32:
                config.fp32_dest_acc_en = True
                if verbose:
                    print(
                        "  [fp32 detected] Enabling fp32_dest_acc_en for compute kernel"
                    )
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
    )

    if verbose:
        print(f"\nCompiled kernel ready (compiled {len(kernel_paths)} threads)")
        print("=" * 60)

    return compiled_kernel


def _collect_captures(
    f: Callable,
) -> Dict[str, Union[int, CircularBuffer]]:
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
        if isinstance(val, int):
            return val
        elif is_ttnn_tensor(val):
            return val
        elif isinstance(val, CircularBuffer):
            return val
        else:
            raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }


def _collect_cb_configs(threads):
    """Extract CircularBuffer objects from thread closures, indexed by cb_index.

    Returns a list of CircularBuffer objects indexed by cb_index. Each CB has
    shape, buffer_factor, tensor (for dtype), and _cb_index attributes.
    """
    cb_configs_dict = {}
    for thread_fn in threads:
        wrapped = getattr(thread_fn, "__wrapped__", None)
        closure = getattr(wrapped, "__closure__", None) if wrapped else None
        if not closure:
            continue
        for cell in closure:
            val = cell.cell_contents
            if isinstance(val, CircularBuffer):
                cb_configs_dict[val._cb_index] = val

    if not cb_configs_dict:
        return []
    max_idx = max(cb_configs_dict.keys())
    return [cb_configs_dict.get(i) for i in range(max_idx + 1)]


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

    for idx, (param_name, arg) in enumerate(zip(f_params, args)):
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

    from .circular_buffer import _reset_cb_counter, CircularBuffer
    from .operators import _set_current_grid

    _reset_cb_counter()
    _set_current_grid(grid)

    _clear_thread_registry()
    f(*args, **kwargs)
    threads = _get_registered_threads()

    if not threads:
        raise ValueError(
            "No threads found. Define at least one @ttl.compute() or "
            "@ttl.datamovement() function inside your kernel."
        )

    cb_configs = _collect_cb_configs(threads)

    injected_program_kwargs = {
        "grid": grid,
        "memory_space": memory_space,
        "tiled": tiled,
        "debug_locations": True,  # Always generate locations for error messages
    }
    program = Program(
        *threads,
        args=args,
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

            # Collect source info for error reporting
            if hasattr(ct, "source_file") and hasattr(ct, "source_lines"):
                all_source_files[ct.name] = ct.source_file
                all_source_lines[ct.name] = ct.source_lines
            # Track per-kernel line offset
            if hasattr(ct, "line_offset"):
                kernel_line_offsets[ct.name] = ct.line_offset

        module = Module.create(loc)

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
        if config_options:
            set_compute_config_pass = (
                "func.func(ttl-set-compute-kernel-config{"
                + " ".join(config_options)
                + "})"
            )

        pipeline_passes = [
            "func.func(convert-ttl-to-compute)",
            set_compute_config_pass,
            "func.func(ttl-assign-dst)",
            "func.func(ttl-insert-tile-regs-sync)",
            "func.func(ttl-lower-to-loops)",
            "func.func(ttl-annotate-cb-associations)",
        ]

        # Add auto-profiling passes if enabled
        if is_auto_profile_enabled():
            if "TTLANG_PROFILE_CSV" in os.environ:
                cb_flow_json = str(Path(os.environ["TTLANG_PROFILE_CSV"]).parent / "cb_flow_graph.json")
            else:
                tt_metal_home = os.environ.get("TT_METAL_HOME", "")
                if not tt_metal_home:
                    raise ValueError("TTLANG_AUTO_PROFILE=1 requires TT_METAL_HOME or TTLANG_PROFILE_CSV to be set")
                cb_flow_json = f"{tt_metal_home}/generated/profiler/.logs/cb_flow_graph.json"
            pipeline_passes.append(f'ttl-dump-cb-flow-graph{{output="{cb_flow_json}"}}')

        pipeline_passes += [
            "convert-ttl-to-ttkernel",
        ]

        if is_auto_profile_enabled():
            pipeline_passes.append("ttl-lower-signpost-to-emitc")

        pipeline_passes += [
            "canonicalize",
            "cse",
            "lower-affine",
            "convert-ttkernel-to-emitc",
            "symbol-dce",
        ]

        if HAS_TT_DEVICE:
            pipeline_passes.insert(0, "ttcore-register-device")

        pipeline = ",".join(pipeline_passes)

        pipeline_str = f"builtin.module({pipeline})"
        # fmt: on
        pm = PassManager.parse(pipeline_str)
        pm.enable_verifier(verify)

        try:
            from ttmlir._mlir_libs._ttmlir import enable_pretty_stack_traces

            enable_pretty_stack_traces(pm._CAPIPtr)
        except Exception:
            # Pretty stack traces are optional, silently continue if unavailable
            pass

        if os.environ.get("TTLANG_VERBOSE_PASSES"):
            print("Running custom pipeline:", pm)
            ctx.enable_multithreading(False)
            pm.enable_ir_printing(
                print_after_all=True,
                print_before_all=True,
                print_after_failure=True,
                enable_debug_info=True,
            )

        # Run the pass manager with error handling for source-aware diagnostics
        try:
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
            formatted = format_mlir_error(error_msg, source_lines, source_file)
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

        # Compile to CompiledTTNNKernel for ttnn.generic_op
        compiled_kernel = _compile_ttnn_kernel(
            module,
            args,
            grid,
            num_outs,
            thread_tensor_indices,
            cb_configs,
            program_hash=program_hash,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
            source_lines=profile_source_lines,
            all_source_lines=all_source_lines,
            kernel_line_offsets=kernel_line_offsets,
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

            # Build cache key from tensor properties
            cache_key = _make_cache_key(
                args,
                # Runtime options:
                fp32_dest_acc_en=fp32_override,
                dst_full_sync_en=dst_sync_override,
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

                return result

        return _wrapper

    return _decorator


# Alias for backward compatibility
kernel = pykernel_gen


__all__ = [
    "pykernel_gen",
    "kernel",
    "Program",
    "compute",
    "datamovement",
    "TensorBlock",
    "CircularBuffer",
    "CopyTransferHandler",
    "copy",
    "CompiledTTNNKernel",
]
