# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Main API for the TTL dialect Python DSL."""

from __future__ import annotations

import ast
import inspect
import functools
import os
from typing import List, Optional, Callable, Dict, Union

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None


from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.dialects import ttkernel
from ttmlir.passes import (
    ttkernel_to_cpp_by_name,
    get_ttkernel_names,
    get_ttkernel_arg_spec,
)


from .ttl_utils import get_thread_type_string


from pykernel._src.utils import _cleanup_source_code
from ._src.tensor_accessor import TensorAccessor
from ._src.tensor_registry import register_tensor_name

from ._src.ttl_ast import TTLGenericCompiler


def _extract_cb_param_names(f: Callable) -> List[str]:
    """
    Extract CircularBuffer parameter names from a thread function.

    Args:
        f: Thread function (decorated with @ttl.compute or @ttl.datamovement)

    Returns:
        List of parameter names annotated as CircularBuffer
    """
    source_code = _cleanup_source_code(f)
    m = ast.parse(source_code)

    cb_names = []
    for node in ast.walk(m):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    if arg.annotation.id == "CircularBuffer":
                        cb_names.append(arg.arg)
            break  # Only process the first function definition

    return cb_names


def _build_kernel_cb_symbol_table(threads: tuple) -> Dict[str, int]:
    """
    Build a kernel-level CB symbol table mapping CB names to global indices.

    Iterates through all threads and collects unique CB names, assigning
    global indices in order of first appearance.

    Args:
        threads: Tuple of thread functions (decorated with @ttl.compute or @ttl.datamovement)

    Returns:
        Dictionary mapping CB parameter names to global CB indices
    """
    cb_symbol_table = {}
    next_cb_index = 0

    for thread in threads:
        # Get the original function from the wrapper
        original_fn = thread.__wrapped__
        cb_names = _extract_cb_param_names(original_fn)

        for cb_name in cb_names:
            if cb_name not in cb_symbol_table:
                cb_symbol_table[cb_name] = next_cb_index
                next_cb_index += 1

    return cb_symbol_table


from .operators import TensorBlock, CopyTransferHandler, copy
from .circular_buffer import CircularBuffer
from .semaphore import Semaphore
from .dtype_utils import (
    torch_dtype_to_ttnn_datatype,
    tile_bytes_from_dtype,
    is_ttnn_tensor,
)
from .constants import SUPPORTED_MEMORY_SPACES

_TTLANG_DIALECTS_REGISTERED = False


def _get_global_registry() -> DialectRegistry:
    """Return the shared DialectRegistry with ttlang dialects registered.

    ttmlir dialects are registered via site initialization (_site_initialize_0)
    when Context is created. However, ttlang's site init is in a different
    namespace (ttlang._mlir_libs) and is not discovered by MLIR's site init.
    We explicitly register ttlang dialects here.
    """
    global _TTLANG_DIALECTS_REGISTERED
    from ttmlir._mlir_libs import get_dialect_registry

    registry = get_dialect_registry()

    # Register ttlang dialects once (ttmlir dialects already registered via site init)
    if not _TTLANG_DIALECTS_REGISTERED:
        import ttlang._mlir_libs._ttlang as _ttlang_lib

        _ttlang_lib.register_dialects(registry)
        _TTLANG_DIALECTS_REGISTERED = True

    return registry


class CompilerConfig:
    """
    Configuration for the compiler pipeline and runtime execution.

    Compiles to C++ for ttnn.generic_op.
    """

    def __init__(self, compile_only: bool = False):
        self._compile_only = (
            compile_only or os.environ.get("TTLANG_COMPILE_ONLY", "0") == "1"
        )

    def should_execute(self) -> bool:
        """Check if runtime execution should proceed."""
        return not self._compile_only


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


def _resolve_grid(grid, args, kwargs):
    """Resolve grid, evaluating callable if needed."""
    return grid(*args, **kwargs) if callable(grid) else grid


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
        kernel_cb_indices,
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
            kernel_cb_indices: List of CB indices used by each kernel
        """
        self.kernel_paths = kernel_paths
        self.kernel_configs = kernel_configs
        self.kernel_arg_specs = kernel_arg_specs
        self.num_tensors = num_tensors
        self.core_ranges = core_ranges
        self.kernel_tensor_indices = kernel_tensor_indices
        self.kernel_cb_indices = kernel_cb_indices

    def __call__(self, *args):
        """Execute the kernel with the given tensors."""
        if len(args) != self.num_tensors:
            raise ValueError(f"Expected {self.num_tensors} tensors, got {len(args)}")

        # Build TensorAccessorArgs config for each tensor (indexed by global tensor index)
        all_tensor_accessor_args = {}
        for idx, tensor in enumerate(args):
            all_tensor_accessor_args[idx] = ttnn.TensorAccessorArgs(
                tensor
            ).get_compile_time_args()

        # Build kernel descriptors with current tensor addresses
        kernel_descriptors = []

        for kernel_idx, ((kernel_path, thread_type), config, rt_args) in enumerate(
            zip(self.kernel_paths, self.kernel_configs, self.kernel_arg_specs)
        ):
            # runtime_args structure: [cores][core_ranges][args_per_core]
            # For single-core execution, this is one empty list per core range.
            # TODO: Support per-core runtime args for multi-core grids
            runtime_args = [[[]]]

            # Build common_runtime_args using kernel_tensor_indices
            # C++ indexes by function-local position, we provide addresses in that order
            tensor_indices = self.kernel_tensor_indices[kernel_idx]
            common_runtime_args = [args[idx].buffer_address() for idx in tensor_indices]

            # CB indices for THIS kernel (not all tensors)
            cb_indices = self.kernel_cb_indices[kernel_idx]

            # Compute kernels only need CB indices
            # DM kernels need CB indices + TensorAccessorArgs for tensors THIS kernel accesses
            if thread_type == "compute":
                kernel_compile_time_args = cb_indices
            else:
                # Build TensorAccessorArgs for only the tensors this kernel uses
                kernel_tensor_accessor_args = []
                for global_idx in tensor_indices:
                    kernel_tensor_accessor_args.extend(
                        all_tensor_accessor_args[global_idx]
                    )
                kernel_compile_time_args = cb_indices + kernel_tensor_accessor_args

            kernel_desc = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                core_ranges=self.core_ranges,
                compile_time_args=kernel_compile_time_args,
                runtime_args=runtime_args,
                common_runtime_args=common_runtime_args,
                config=config,
            )
            kernel_descriptors.append(kernel_desc)

        # Build CB descriptors
        cb_descriptors = []
        for i, tensor in enumerate(args):
            if hasattr(tensor, "dtype") and hasattr(tensor.dtype, "name"):
                data_format = tensor.dtype
            else:
                data_format = torch_dtype_to_ttnn_datatype(tensor.dtype)

            page_size = tile_bytes_from_dtype(data_format)
            if hasattr(tensor, "volume"):
                num_tiles = max(1, tensor.volume() // 1024)
            else:
                num_tiles = 1
            total_size = num_tiles * page_size

            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=i,
                data_format=data_format,
                page_size=page_size,
            )
            cb_desc = ttnn.CBDescriptor(
                total_size=total_size,
                core_ranges=self.core_ranges,
                format_descriptors=[cb_format],
            )
            cb_descriptors.append(cb_desc)

        # Build and execute program
        # TODO: Extract semaphore info from kernel IR for synchronization
        program = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=[],
        )

        return ttnn.generic_op(list(args), program)


def _write_kernel_to_tmp(name: str, source: str) -> str:
    """Write kernel source to /tmp and return the file path."""
    import hashlib

    # Use content hash in path to avoid JIT cache collisions between different kernels
    content_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    path = f"/tmp/ttlang_kernel_{name}_{content_hash}.cpp"
    with open(path, "w") as f:
        f.write(source)
    print(f"=== {name} kernel written to {path} ===")
    print(source)
    print("=" * 60)
    return path


def _compile_ttnn_kernel(
    module, args, grid, num_outs, thread_tensor_indices, thread_cb_indices, verbose=True
):
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.

    Args:
        module: MLIR module after D2M pipeline (with EmitC kernels)
        args: Input/output tensors (used for shape/dtype info)
        grid: Grid dimensions tuple
        num_outs: Number of output tensors
        thread_tensor_indices: List of global tensor indices used by each thread
        thread_cb_indices: List of CB indices used by each thread
        verbose: Print compilation info

    Returns:
        CompiledTTNNKernel ready for execution
    """
    # Get kernel info from module
    kernel_info = get_ttkernel_names(module)

    # Validate grid is single core
    if grid != (1, 1) and grid != [1, 1]:
        raise ValueError(
            f"TTNN interop only supports single-core grid (1, 1), got {grid}"
        )

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

    # TODO: Support multi-core grids. Currently hardcoded to single core (0,0).
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRange(core, core)
    core_ranges = ttnn.CoreRangeSet([core_range])

    if verbose:
        print(f"\nCore range: {core_ranges}")

    num_tensors = len(args)

    # Write all kernels to /tmp for debugging
    for _, (name, thread_type) in enumerate(kernel_info):
        cpp_source = ttkernel_to_cpp_by_name(module, name)
        _write_kernel_to_tmp(name, cpp_source)

    kernel_paths = []
    kernel_configs = []
    kernel_arg_specs = []
    noc_kernel_idx = 0

    for _, (name, thread_type) in enumerate(kernel_info):
        cpp_source = ttkernel_to_cpp_by_name(module, name)
        kernel_path = _write_kernel_to_tmp(name, cpp_source)
        kernel_paths.append((kernel_path, thread_type))

        if thread_type == "compute":
            config = ttnn.ComputeConfigDescriptor()
        elif thread_type == "noc":
            if noc_kernel_idx == 0:
                config = ttnn.ReaderConfigDescriptor()
            else:
                config = ttnn.WriterConfigDescriptor()
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
        kernel_cb_indices=thread_cb_indices,
    )

    if verbose:
        print(f"\nCompiled kernel ready (will execute on {len(kernel_paths)} kernels)")
        print("=" * 60)

    return compiled_kernel


def _collect_captures(f: Callable) -> Dict[str, Union[int, TensorAccessor]]:
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
        elif isinstance(val, TensorAccessor):
            return val
        else:
            raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }


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
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            source_code = _cleanup_source_code(f)

            if verbose:
                kwargs["_source_code"] = source_code.splitlines()
                kwargs["_verbose"] = True

            m = ast.parse(source_code)
            b = TTLGenericCompiler(
                f.__name__,
                kernel_type,
                _collect_captures(f),
                *args,
                **kwargs,
            )

            if verbose:
                print(ast.dump(m, indent=4) + "\n")

            b.visit(m)

            if verbose:
                print(b.module)

            b.module.operation.verify()

            return b

        _wrapper._decorator_name = kernel_type + "_thread"
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


def _compile_and_run_kernel(
    f: Callable,
    args: tuple,
    kwargs: dict,
    grid: Union[tuple, List[int]],
    indexing_maps: List[Callable],
    iterator_types: List[str],
    num_outs: int,
    memory_space: str,
    tiled: bool,
    compile_only: bool = False,
) -> Optional[CompiledTTNNKernel]:
    """
    Compile kernel function to MLIR and execute compilation pipeline.

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
        compile_only: If True, return compiled kernel without executing

    Returns:
        CompiledTTNNKernel ready for execution
    """
    f_params = inspect.signature(f).parameters

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

    inject_kwargs = [
        ("grid", grid),
        ("memory_space", memory_space),
        ("tiled", tiled),
    ]
    for injected_kwarg, val in inject_kwargs:
        if injected_kwarg in f_params:
            kwargs[injected_kwarg] = val

    program = f(*args, **kwargs)
    if not isinstance(program, Program):
        raise TypeError(
            f"Kernel function must return a Program, got {type(program).__name__}"
        )

    injected_program_kwargs = {
        "grid": grid,
        "memory_space": memory_space,
        "tiled": tiled,
    }
    program = Program(
        *program.threads,
        args=program.args,
        kwargs={**injected_program_kwargs, **program.kwargs},
    )

    registry = _get_global_registry()

    ctx = Context()
    ctx.append_dialect_registry(registry)
    ctx.load_all_available_dialects()

    # Make sure key dialects are loaded before running pipelines that materialize
    # their operations inside C++ passes.
    _ = ctx.dialects["ttkernel"]
    _ = ctx.dialects["ttl"]
    loc = Location.unknown(ctx)
    with ctx, loc:
        # Build kernel-level CB symbol table before compiling any threads.
        # This ensures all threads share the same CB index numbering.
        kernel_cb_symbol_table = _build_kernel_cb_symbol_table(program.threads)

        compiled_threads = []
        # Track which global tensor indices each thread uses (for building common_runtime_args)
        thread_tensor_indices = []
        # Track which CB indices each thread uses (for compile-time args)
        thread_cb_indices = []

        # Add kernel_cb_symbol_table to kwargs for thread compilation
        program_kwargs_with_cb_table = {
            **program.kwargs,
            "kernel_cb_symbol_table": kernel_cb_symbol_table,
        }

        # Total number of CBs in the kernel (used for TensorAccessorArgs offset)
        total_num_cbs = len(kernel_cb_symbol_table)

        for compile_thread in program.threads:
            ct = compile_thread(*program.args, **program_kwargs_with_cb_table)
            compiled_threads.append(ct)
            thread_tensor_indices.append(ct._tensor_accessor_global_indices)
            # Extract CB indices from _cb_info
            thread_cb_indices.append([cb["cb_index"] for cb in ct._cb_info])

        module = Module.create(loc)

        # Insert standalone thread functions directly into module
        with InsertionPoint(module.body):
            for ct in compiled_threads:
                ct.func_entry.operation.detach_from_parent()
                module.body.append(ct.func_entry)

        initial_mlir_path = os.environ.get("TTLANG_INITIAL_MLIR")
        if initial_mlir_path:
            with open(initial_mlir_path, "w") as fd:
                print(module, file=fd)
            print(f"SAVED INITIAL TO {initial_mlir_path}")

        verify = True
        config = CompilerConfig(compile_only)

        # fmt: off
        pipeline_passes = [
            "func.func(convert-ttl-to-compute)",
            "func.func(ttl-tile-and-assign-dst)",
            "func.func(ttl-insert-tile-regs-sync)",
            "func.func(ttl-lower-to-loops)",
            "func.func(ttl-annotate-cb-associations)",
            "convert-ttl-to-ttkernel",
            "canonicalize",
            "cse",
            "lower-affine",
            "convert-ttkernel-to-emitc",
            "symbol-dce",
        ]


        pipeline_str = f"builtin.module({','.join(pipeline_passes)})"
        # fmt: on
        pm = PassManager.parse(pipeline_str, context=ctx)
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

        pm.run(module.operation)

        final_mlir_path = os.environ.get("TTLANG_FINAL_MLIR")
        if final_mlir_path:
            with open(final_mlir_path, "w") as fd:
                print(module, file=fd)
            print(f"SAVED FINAL TO {final_mlir_path}")

        # Compile to CompiledTTNNKernel for ttnn.generic_op
        compiled_kernel = _compile_ttnn_kernel(
            module, args, grid, num_outs, thread_tensor_indices, thread_cb_indices
        )
        if compiled_kernel is not None and config.should_execute():
            compiled_kernel(*args)
        return compiled_kernel


def pykernel_gen(
    grid: Optional[Union[tuple, Callable]] = None,
    indexing_maps: Optional[List[Callable]] = None,
    iterator_types: Optional[List[str]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
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
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            resolved_grid = _resolve_grid(grid, args, kwargs)

            _compile_and_run_kernel(
                f,
                args,
                kwargs,
                resolved_grid,
                indexing_maps,
                iterator_types,
                num_outs,
                memory_space,
                tiled,
            )

        def _compile(*args, **kwargs):
            """
            Compile the kernel without executing, returning a CompiledTTNNKernel.

            Use this when you want to compile once and execute many times.

            Args:
                *args: Sample tensors (same shape/dtype as actual inputs)

            Returns:
                CompiledTTNNKernel that can be called multiple times
            """
            resolved_grid = _resolve_grid(grid, args, kwargs)

            return _compile_and_run_kernel(
                f,
                args,
                kwargs,
                resolved_grid,
                indexing_maps,
                iterator_types,
                num_outs,
                memory_space,
                tiled,
                compile_only=True,
            )

        _wrapper.compile = _compile
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
    "Semaphore",
    "copy",
    "TensorAccessor",
    "CompiledTTNNKernel",
]
