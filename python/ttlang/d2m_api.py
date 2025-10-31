# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Main API for the D2M dialect Python DSL."""

from __future__ import annotations

import ast
import inspect
import functools
import os
from typing import List, Optional, Callable, Dict, Any, Union

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from _ttmlir_runtime import runtime, binary
except ModuleNotFoundError:
    runtime = None
    binary = None

from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.dialects import ttcore, d2m, func
from ttmlir.passes import ttmetal_to_flatbuffer_bin

from pykernel._src.utils import _cleanup_source_code
from ._src.stream import Stream

from ._src.d2m_ast import D2MGenericCompiler

from .operators import TensorBlock, MemTx, dma
from .circular_buffer import CircularBuffer
from .semaphore import Semaphore
from .layouts import create_metal_layout
from .dtype_utils import to_data_type, from_data_type
from .constants import SUPPORTED_MEMORY_SPACES
from ._src.codegen import create_generic_func, copy_symbol_table_globals


def _collect_captures(f: Callable) -> Dict[str, Union[int, Stream]]:
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
        elif isinstance(val, Stream):
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
) -> None:
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
    """
    f_params = inspect.signature(f).parameters

    for param_name, arg in zip(f_params, args):
        arg._global_name = param_name

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
        if positional_arg_names[-num_outs] in streams:
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
        use_tile_matmul = True
        pipeline = f"d2m-generic-replace-globals,ttir-to-ttmetal-pipeline{{use-tile-matmul={1 if use_tile_matmul else 0}}}"

        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
        pm = PassManager.parse(pipeline_str)
        pm.enable_verifier(verify)

        try:
            from ttmlir._mlir_libs._ttmlir import enable_pretty_stack_traces

            enable_pretty_stack_traces(pm._CAPIPtr)
        except Exception as e:
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

        bin = ttmetal_to_flatbuffer_bin(module)


def pykernel_gen(
    grid: Optional[Union[tuple, Callable]] = None,
    block_factors: Optional[Union[List, Callable]] = None,
    indexing_maps: Optional[List[Callable]] = None,
    iterator_types: Optional[List[str]] = None,
    num_outs: int = 1,
    memory_space: str = "L1",
    tiled: bool = True,
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
                    raise ValueError(f"Number of dimensions ({num_dims}) must match iterator_types length ({len(iterator_types)})")
            if block_factors is None:
                block_factors = [1] * len(num_dims)
            if len(block_factors) != num_dims:
                raise ValueError(f"block_factors length ({len(block_factors)}) must match number of dimensions ({num_dims})")

    if iterator_types is None:
        iterator_types = []

    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            nonlocal grid
            nonlocal block_factors

            if callable(grid):
                grid = grid(*args, **kwargs)

            if callable(block_factors):
                block_factors = block_factors(*args, **kwargs)

            if block_factors is None:
                block_factors = [1] * len(grid)

            _compile_and_run_kernel(
                f,
                args,
                kwargs,
                grid,
                block_factors,
                indexing_maps,
                iterator_types,
                num_outs,
                memory_space,
                tiled,
            )

        return _wrapper

    return _decorator


__all__ = [
    "pykernel_gen",
    "Program",
    "compute",
    "datamovement",
    "TensorBlock",
    "CircularBuffer",
    "MemTx",
    "Semaphore",
    "dma",
    "Stream",
    "create_metal_layout",
    "to_data_type",
    "from_data_type",
]
