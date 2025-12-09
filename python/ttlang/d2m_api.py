# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Main API for the D2M dialect Python DSL."""

from __future__ import annotations

import ast
import inspect
import functools
import os
import platform
from typing import List, Optional, Callable, Dict, Union

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None

try:
    from _ttmlir_runtime import runtime, binary
except ModuleNotFoundError:
    runtime = None
    binary = None

from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.dialects import ttcore, ttkernel
from ttmlir.passes import (
    ttmetal_to_flatbuffer_bin,
    ttkernel_to_cpp_by_name,
    get_ttkernel_names,
    get_ttkernel_arg_spec,
)

import ttlang._mlir_libs._ttlang  # Register tt-lang passes

from pykernel._src.utils import _cleanup_source_code
from ._src.tensor_accessor import TensorAccessor
from ._src.tensor_registry import register_tensor_name

from ._src.d2m_ast import D2MGenericCompiler

from .operators import TensorBlock, MemTx, dma
from .circular_buffer import CircularBuffer
from .semaphore import Semaphore
from .layouts import create_metal_layout
from .dtype_utils import (
    to_data_type,
    from_data_type,
    TORCH_TO_RUNTIME_DTYPE_INT,
    create_borrowed_tensors,
    torch_dtype_to_ttnn_datatype,
    tile_bytes_from_dtype,
    _is_ttnn_tensor,
)
from .constants import SUPPORTED_MEMORY_SPACES
from ._src.codegen import create_generic_func, copy_symbol_table_globals


class CompilerConfig:
    """
    Configuration for the compiler pipeline and runtime execution.

    Two mutually exclusive compilation paths:
    - Metal: Compiles to flatbuffer, executes on Metal runtime
    - TTNN: Compiles to C++ for ttnn.generic_op
    """

    def __init__(self, ttnn_interop: bool = False, compile_only: bool = False):
        self._compile_only = (
            compile_only or os.environ.get("TTLANG_COMPILE_ONLY", "0") == "1"
        )
        self._runtime_available = binary is not None and runtime is not None
        self._is_macos = platform.system() == "Darwin"
        self._ttnn_interop = ttnn_interop

    @property
    def is_ttnn(self) -> bool:
        return self._ttnn_interop

    @property
    def ttnn_mode(self) -> int:
        """Pass parameter for convert-d2m-to-ttkernel."""
        return 1 if self._ttnn_interop else 0

    def should_execute(self) -> bool:
        """Check if runtime execution should proceed for the selected path."""
        if self._compile_only:
            return False
        if self._ttnn_interop:
            return True
        return not self._is_macos and self._runtime_available


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


def _resolve_grid_params(grid, block_factors, args, kwargs):
    """Resolve grid and block_factors, evaluating callables if needed."""
    resolved_grid = grid(*args, **kwargs) if callable(grid) else grid
    resolved_block_factors = (
        block_factors(*args, **kwargs) if callable(block_factors) else block_factors
    )
    if resolved_block_factors is None:
        resolved_block_factors = [1] * len(resolved_grid)
    return resolved_grid, resolved_block_factors


class CompiledTTNNKernel:
    """
    A compiled tt-lang kernel ready for execution via ttnn.generic_op.

    Caches compilation artifacts (kernel paths, CB descriptors) so the kernel
    can be executed multiple times with different tensors without recompiling.
    """

    def __init__(
        self, kernel_paths, kernel_configs, kernel_arg_specs, num_tensors, core_ranges
    ):
        """
        Initialize with pre-compiled kernel artifacts.

        Args:
            kernel_paths: List of (path, thread_type) tuples for each kernel
            kernel_configs: List of config descriptors matching kernel_paths
            kernel_arg_specs: List of arg specs (rt_args list) for each kernel
            num_tensors: Number of input/output tensors
            core_ranges: CoreRangeSet for kernel execution
        """
        self.kernel_paths = kernel_paths
        self.kernel_configs = kernel_configs
        self.kernel_arg_specs = kernel_arg_specs
        self.num_tensors = num_tensors
        self.core_ranges = core_ranges

    def __call__(self, *args):
        """Execute the kernel with the given tensors."""
        if len(args) != self.num_tensors:
            raise ValueError(f"Expected {self.num_tensors} tensors, got {len(args)}")

        # Build kernel descriptors with current tensor addresses
        kernel_descriptors = []

        for (kernel_path, thread_type), config, rt_args in zip(
            self.kernel_paths, self.kernel_configs, self.kernel_arg_specs
        ):
            # TODO: Derive compile-time args from kernel IR (CB indices, tile counts, etc.)
            compile_time_args = list(range(self.num_tensors))
            # runtime_args structure: [cores][core_ranges][args_per_core]
            # For single-core execution, this is one empty list per core range.
            # TODO: Support per-core runtime args for multi-core grids
            runtime_args = [[[]]]

            # Build common_runtime_args from arg_spec: BufferAddress args -> tensor addresses
            # ArgType.BufferAddress = 1
            common_runtime_args = []
            for arg in rt_args:
                arg = ttkernel.ir.ArgAttr.maybe_downcast(arg)
                if arg.arg_type_as_value == 1:  # BufferAddress
                    tensor_idx = arg.operand_index
                    common_runtime_args.append(args[tensor_idx].buffer_address())

            kernel_desc = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                core_ranges=self.core_ranges,
                compile_time_args=compile_time_args,
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


def _execute_on_metal_runtime(flatbuffer_binary, args):
    """
    Execute compiled kernel on Metal runtime.

    Args:
        flatbuffer_binary: Compiled flatbuffer binary capsule
        args: List of torch tensors as input arguments

    Raises:
        Exception: If runtime execution fails
    """
    binary_obj = binary.load_binary_from_capsule(flatbuffer_binary)
    program_index = 0

    runtime.set_compatible_device_runtime(binary_obj)

    device_options = runtime.MeshDeviceOptions()
    device_options.mesh_shape = binary_obj.get_program_mesh_shape(program_index)
    device = runtime.open_mesh_device(device_options)

    # Borrowed tensors share memory with torch tensors (zero-copy)
    inputs = create_borrowed_tensors(args)
    runtime_outputs = runtime.submit(device, binary_obj, program_index, inputs)
    runtime.wait(runtime_outputs)

    # Results are written directly to torch tensor memory via enqueue_read_buffer
    for runtime_output_tensor in runtime_outputs:
        runtime.deallocate_tensor(runtime_output_tensor, force=True)

    runtime.close_mesh_device(device)


def _write_kernel_to_tmp(name: str, source: str) -> str:
    """Write kernel source to /tmp and return the file path."""
    path = f"/tmp/ttlang_kernel_{name}.cpp"
    with open(path, "w") as f:
        f.write(source)
    return path


def _compile_ttnn_kernel(module, args, grid, num_outs, verbose=True):
    """
    Compile kernel to CompiledTTNNKernel for execution via ttnn.generic_op.

    Builds kernel paths, configs, and CB descriptors from compiled MLIR module.

    Args:
        module: MLIR module after D2M pipeline (with EmitC kernels)
        args: Input/output tensors (used for shape/dtype info)
        grid: Grid dimensions tuple
        num_outs: Number of output tensors
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
    ttnn_count = sum(1 for arg in args if _is_ttnn_tensor(arg))
    if ttnn_count > 0 and ttnn_count < len(args):
        raise ValueError(
            f"TTNN interop requires all tensors to be the same type. "
            f"Got {ttnn_count} TTNN tensors and {len(args) - ttnn_count} host tensors. "
            f"Mixed tensor types would generate extra bounce kernels."
        )

    # Validate TTNN tensors - currently support L1 (sharded) and DRAM (interleaved), must be tilized
    for i, arg in enumerate(args):
        if _is_ttnn_tensor(arg):
            mem_space = _detect_memory_space_from_tensor(arg, "unknown")
            if mem_space not in ("L1", "DRAM"):
                raise ValueError(
                    f"TTNN interop requires L1 or DRAM memory space, but tensor {i} is in {mem_space}."
                )
            if hasattr(arg, "layout") and "TILE" not in str(arg.layout()):
                raise ValueError(
                    f"TTNN interop requires tilized tensors, but tensor {i} has layout {arg.layout()}. "
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

    # Write all kernels to /tmp for debugging
    for name, thread_type in kernel_info:
        cpp_source = ttkernel_to_cpp_by_name(module, name)
        _write_kernel_to_tmp(name, cpp_source)

    kernel_paths = []
    kernel_configs = []
    kernel_arg_specs = []
    noc_kernel_idx = 0

    for name, thread_type in kernel_info:
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
            b = D2MGenericCompiler(
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


_g_current_system_desc = None


def _compile_and_run_kernel(
    f: Callable,
    args: tuple,
    kwargs: dict,
    grid: Union[tuple, List[int]],
    block_factors: Union[List, List[int]],
    indexing_maps: List[Callable],
    iterator_types: List[str],
    num_outs: int,
    memory_space: str,
    tiled: bool,
    ttnn_interop: bool = False,
    compile_only: bool = False,
) -> Optional[CompiledTTNNKernel]:
    """
    Compile kernel function to MLIR and execute compilation pipeline.

    Args:
        f: User kernel function
        args: Positional arguments for the kernel
        kwargs: Keyword arguments for the kernel
        grid: Grid dimensions
        block_factors: Block factors for each argument
        indexing_maps: List of lambda functions for indexing
        iterator_types: List of iterator type strings
        num_outs: Number of output arguments
        memory_space: "L1" or "DRAM"
        tiled: Whether to use tiled layout
        ttnn_interop: If True, compile to C++ for ttnn.generic_op instead of flatbuffer
        compile_only: If True, return compiled kernel without executing (ttnn_interop only)

    Returns:
        CompiledTTNNKernel if ttnn_interop=True, None otherwise
    """
    f_params = inspect.signature(f).parameters

    has_ttnn_tensors = any(_is_ttnn_tensor(arg) for arg in args)

    # For TTNN tensors, detect memory space from tensor's buffer type.
    # L1 tensors use simple NOC addressing, DRAM uses bank-aware addressing.
    # TODO: Check all tensors and handle mixed memory spaces.
    if has_ttnn_tensors:
        first_ttnn_tensor = next((arg for arg in args if _is_ttnn_tensor(arg)), None)
        if first_ttnn_tensor is not None:
            memory_space = _detect_memory_space_from_tensor(
                first_ttnn_tensor, memory_space
            )
            print(f"[TTNN interop] Detected {memory_space} memory space")

    for param_name, arg in zip(f_params, args):
        register_tensor_name(arg, param_name)

    inject_kwargs = [
        ("block_factors", block_factors),
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

    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        compiled_threads = []
        for compile_thread in program.threads:
            compiled_threads.append(compile_thread(*program.args, **program.kwargs))

        module = Module.create(loc)

        module_symbol_table = SymbolTable(module.operation)
        with InsertionPoint.at_block_begin(module.body):
            copy_symbol_table_globals(module_symbol_table, compiled_threads, f_params)

        streams = set().union(*[ct.streams for ct in compiled_threads])
        positional_arg_names = list(f_params.keys())[: len(args)]
        stream_func_arg_attrs = [
            DictAttr.get({"d2m.stream": BoolAttr.get(p in streams)})
            for p in positional_arg_names
        ]
        if positional_arg_names[-num_outs] in streams and not ttnn_interop:
            raise ValueError("Output streaming is not supported")

        with InsertionPoint(module.body):
            create_generic_func(
                ctx,
                f.__name__,
                stream_func_arg_attrs,
                grid,
                block_factors,
                indexing_maps,
                iterator_types,
                compiled_threads,
                num_outs,
                args,
                tiled,
                memory_space,
            )

        initial_mlir_path = os.environ.get("TTLANG_INITIAL_MLIR")
        if initial_mlir_path:
            with open(initial_mlir_path, "w") as fd:
                print(module, file=fd)
            print(f"SAVED INITIAL TO {initial_mlir_path}")

        device_register_options = f"system-desc-path={_g_current_system_desc}"
        verify = True
        config = CompilerConfig(ttnn_interop, compile_only)

        # fmt: off
        # Common pipeline passes up through EmitC conversion
        common_pipeline_passes = [
            "d2m-generic-replace-globals",
            "d2m-lower-to-layout",                         # Lower to_layout to data movement
            "d2m-elementwise-fusion",                      # Fuse d2m.generic operations
            "canonicalize",                                # Cleanup and simplify
            "ttcore-one-shot-bufferize",
            "func.func(d2m-simple-allocate)",              # Our simplified allocator
            "d2m-linalg-to-affine{use-tile-matmul=1}",     # Convert all linalg including matmul
            "d2m-insert-dst-register-access",
            "lower-affine",
            "d2m-generic-linearize-memref",
            "d2m-generic-generate-datamovement",           # Generate DMA regions for streams
            "d2m-generic-lower-dmas",                      # Lower DMAs to hardware
            "canonicalize",                                # Simplify before region extraction
            "loop-invariant-code-motion",                  # Hoist invariants
            "sccp",                                        # Sparse conditional constant propagation
            "cse",                                         # Eliminate common subexpressions
            "d2m-generic-regions-to-funcs",                # Extract regions to functions
            f"convert-d2m-to-ttkernel{{ttnn-mode={config.ttnn_mode}}}",
            "ttkernel-control-dst-section",                # Insert tile_regs_commit/wait/release
            "convert-ttkernel-to-emitc",                   # Convert TTKernel ops to EmitC
        ]

        # Cleanup passes applied to both paths
        cleanup_passes = [
            "canonicalize",                                # Cleanup after conversion
            "loop-invariant-code-motion",                  # Hoist again after backend lowering
            "sccp",                                        # Propagate constants
            "cse",                                         # Final deduplication
            "symbol-dce"                                   # Remove unused functions
        ]

        if config.is_ttnn:
            # TTNN interop path: stop at EmitC, apply cleanup, translate to C++
            pipeline_passes = common_pipeline_passes + cleanup_passes
        else:
            # Metal path: continue to TTMetal, apply cleanup, then flatbuffer
            metal_pipeline_passes = [
                "convert-d2m-to-ttmetal",                  # Convert to_layout to ttmetal enqueue ops
            ]
            pipeline_passes = common_pipeline_passes + metal_pipeline_passes + cleanup_passes

        pipeline = ",".join(pipeline_passes)

        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
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

        pm.run(module.operation)

        final_mlir_path = os.environ.get("TTLANG_FINAL_MLIR")
        if final_mlir_path:
            with open(final_mlir_path, "w") as fd:
                print(module, file=fd)
            print(f"SAVED FINAL TO {final_mlir_path}")

        if config.is_ttnn:
            # TTNN interop path: compile to CompiledTTNNKernel
            compiled_kernel = _compile_ttnn_kernel(module, args, grid, num_outs)
            if compiled_kernel is not None and config.should_execute():
                compiled_kernel(*args)
            return compiled_kernel

        # Metal path: convert to flatbuffer and optionally execute
        flatbuffer_binary = ttmetal_to_flatbuffer_bin(module)

        # Save flatbuffer to file for ttrt execution
        flatbuffer_path = os.environ.get("TTLANG_FLATBUFFER_PATH")
        if flatbuffer_path and binary is not None:
            binary_obj = binary.load_binary_from_capsule(flatbuffer_binary)
            binary_obj.store(flatbuffer_path)
            print(f"SAVED FLATBUFFER TO {flatbuffer_path}")

        if config.should_execute():
            try:
                _execute_on_metal_runtime(flatbuffer_binary, args)
            except Exception as e:
                print(f"Warning: Metal runtime execution failed: {e}")
                print("(This is expected on macOS or if hardware is not available)")
                import traceback

                traceback.print_exc()


def pykernel_gen(
    grid: Optional[Union[tuple, Callable]] = None,
    block_factors: Optional[Union[List, Callable]] = None,
    indexing_maps: Optional[List[Callable]] = None,
    iterator_types: Optional[List[str]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
    ttnn_interop: bool = False,
) -> Callable:
    """
    Decorator for generating D2M kernels from Python functions.

    This decorator compiles Python functions into D2M dialect operations,
    handling thread compilation, stream creation, and pipeline execution.

    Args:
        grid: Grid dimensions as tuple (e.g., (2, 2)) or callable
        block_factors: Block factors for each argument or callable
        indexing_maps: List of lambda functions for indexing (optional)
        iterator_types: List of iterator types ("parallel", "reduction")
        num_outs: Number of output arguments
        memory_space: "L1" or "DRAM"
        tiled: Whether to use tiled layout
        ttnn_interop: If True, compile to C++ for ttnn.generic_op instead of flatbuffer

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

    global _g_current_system_desc
    if _g_current_system_desc is None:
        _g_current_system_desc = os.environ.get("SYSTEM_DESC_PATH", None)
    if _g_current_system_desc is None:
        system_desc = runtime.get_current_system_desc()
        _g_current_system_desc = "current.ttsys"
        system_desc.store(_g_current_system_desc)

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
            if block_factors is None:
                block_factors = [1] * len(num_dims)
            if len(block_factors) != num_dims:
                raise ValueError(
                    f"block_factors length ({len(block_factors)}) must match number of dimensions ({num_dims})"
                )

    if iterator_types is None:
        iterator_types = []

    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            resolved_grid, resolved_block_factors = _resolve_grid_params(
                grid, block_factors, args, kwargs
            )

            _compile_and_run_kernel(
                f,
                args,
                kwargs,
                resolved_grid,
                resolved_block_factors,
                indexing_maps,
                iterator_types,
                num_outs,
                memory_space,
                tiled,
                ttnn_interop,
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
            resolved_grid, resolved_block_factors = _resolve_grid_params(
                grid, block_factors, args, kwargs
            )

            return _compile_and_run_kernel(
                f,
                args,
                kwargs,
                resolved_grid,
                resolved_block_factors,
                indexing_maps,
                iterator_types,
                num_outs,
                memory_space,
                tiled,
                ttnn_interop,
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
    "MemTx",
    "Semaphore",
    "dma",
    "TensorAccessor",
    "CompiledTTNNKernel",
    "create_metal_layout",
    "to_data_type",
    "from_data_type",
]
