# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Program execution framework for multi-core simulation.

This module provides the core execution framework for running compute and data movement
functions across multiple cores with proper context binding and error handling.
"""

import copy
import ast
import inspect
import textwrap
import traceback
import types
from types import CellType, FunctionType
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol, Tuple

from .block import ThreadType, _set_current_thread_type, _clear_current_thread_type
from .cb import CircularBuffer
from .cbapi import CBAPI
from .ttnnsim import Tensor
from .typedefs import Shape
from .xformyield import transform_wait_reserve_to_yield_ast


# TODO: Pretty printing should be moved to utils maybe?
# Import at runtime to avoid circular dependency
def _get_ttlang_compile_error():
    """Lazy import of TTLangCompileError to avoid circular dependency."""
    import importlib.util
    import sys
    from pathlib import Path

    # Direct import of diagnostics module without going through ttl package
    # This avoids importing the full compiler infrastructure
    diagnostics_path = Path(__file__).parent.parent / "ttl" / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("ttl.diagnostics", diagnostics_path)
    if spec and spec.loader:
        diagnostics = importlib.util.module_from_spec(spec)
        sys.modules["ttl.diagnostics"] = diagnostics
        spec.loader.exec_module(diagnostics)
        return diagnostics.TTLangCompileError
    raise ImportError("Could not load ttl.diagnostics")


# Protocol for templates that have a bind method
class BindableTemplate(Protocol):
    """Protocol for templates that can be bound to a specific execution context."""

    __name__: str

    def bind(self, ctx: Dict[str, Any]) -> Callable[[], Any]:
        """Bind the template to a specific execution context."""
        ...


def _make_cell(value: Any) -> CellType:
    """Create a real closure cell holding `value`."""

    def inner() -> Any:
        return value

    assert inner.__closure__ is not None
    return inner.__closure__[0]


def rebind_func_with_ctx(func: FunctionType, ctx: Dict[str, Any]) -> FunctionType:
    """
    Create a new function from `func` but with:
      - globals = func.__globals__ + ctx
      - closure cells rebuilt from ctx when possible
    so that names like `out_cb` that were captured will now point to the per-core objects.
    """
    freevars = func.__code__.co_freevars
    orig_closure = func.__closure__ or ()
    orig_cell_map: Dict[str, CellType] = {
        name: cell for name, cell in zip(freevars, orig_closure)
    }

    new_cells: List[CellType] = []
    for name in freevars:
        if name in ctx:
            new_cells.append(_make_cell(ctx[name]))
        else:
            # fall back to original cell if we don't have an override
            new_cells.append(orig_cell_map[name])

    # merge globals with ctx so globals-based lookups also see per-core state
    new_globals: Dict[str, Any] = dict(func.__globals__)
    new_globals.update(ctx)

    new_func = types.FunctionType(
        func.__code__, new_globals, func.__name__, func.__defaults__, tuple(new_cells)
    )
    return new_func


def Program(*funcs: BindableTemplate, grid: Shape) -> Any:
    """Program class that combines compute and data movement functions.

    Args:
        *funcs: Compute and data movement function templates
        grid: Grid size tuple
    """

    class ProgramImpl:
        def __init__(
            self,
            *functions: BindableTemplate,
        ):
            self.functions = functions
            self.context: Dict[str, Any] = {"grid": grid}

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                # Capture caller's locals for any remaining context variables
                # Don't reset context - grid was already set in __init__
                self.context.update(frame.f_back.f_locals)

            # Extract closure variables from thread functions and add to context
            # This ensures variables like CBs that were defined in the kernel function
            # are available for per-core copying
            for tmpl in self.functions:
                if hasattr(tmpl, "__wrapped__"):
                    func = tmpl.__wrapped__
                    if func.__code__.co_freevars and func.__closure__:
                        for var_name, cell in zip(
                            func.__code__.co_freevars, func.__closure__
                        ):
                            try:
                                # Only add if not already in context
                                if var_name not in self.context:
                                    self.context[var_name] = cell.cell_contents
                            except ValueError:
                                # Cell is empty (variable not yet bound)
                                pass

            grid = self.context.get("grid", (1, 1))
            # Calculate total cores for any dimension grid
            total_cores = 1
            for dim_size in grid:
                total_cores *= dim_size

            compute_func_tmpl, dm0_tmpl, dm1_tmpl = self.functions

            # Run in cooperative mode
            self._run_cooperative(total_cores, compute_func_tmpl, dm0_tmpl, dm1_tmpl)

        def _build_core_context(self, core: int) -> Dict[str, Any]:
            """Build per-core context with copied circular buffers and other state.

            Args:
                core: Core number to build context for

            Returns:
                Dictionary containing per-core context with fresh CircularBuffers
            """
            memo: Dict[int, Any] = {}
            core_context: Dict[str, Any] = {}
            api = CBAPI()  # new CBAPI per core

            for key, value in self.context.items():
                # Skip module objects (e.g., local imports like `from python.sim import ttnn`)
                if isinstance(value, types.ModuleType):
                    core_context[key] = value
                    continue

                match value:
                    case Tensor():
                        core_context[key] = value
                        memo[id(value)] = value
                    case CircularBuffer():
                        # create a fresh CB for this core
                        new_cb = CircularBuffer(
                            element=value.element,
                            shape=value.shape,
                            buffer_factor=value.buffer_factor,
                            api=api,
                        )
                        # Store the variable name for debugging
                        new_cb._name = key
                        core_context[key] = new_cb
                    case _:
                        core_context[key] = copy.deepcopy(value, memo)

            # also make the core number visible
            core_context["_core"] = core
            # Also inject grid into core context for grid_size() function
            core_context["grid"] = self.context.get("grid", (1, 1))

            return core_context

        def _run_cooperative(
            self,
            total_cores: int,
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> None:
            """Cooperative scheduling execution mode - all cores run in round-robin."""

            # Create transformed sources for all cores
            all_sources: List[
                Tuple[str, str, ast.Module, Dict[str, Any], str, int, ThreadType]
            ] = []

            # Track all per-core contexts for validation
            all_core_contexts: List[Dict[str, Any]] = []

            for core in range(total_cores):
                # build per-core context
                core_context = self._build_core_context(core)
                all_core_contexts.append(core_context)

                # Transform functions to sources with yields
                sources = self._create_cooperative_generators(
                    core, core_context, compute_func_tmpl, dm0_tmpl, dm1_tmpl
                )

                all_sources.extend(sources)

            # Run round-robin scheduler across all cores
            self._run_round_robin_scheduler(all_sources)

            # Validate all CircularBuffers have no pending blocks
            self._validate_circular_buffers(all_core_contexts)

        def _validate_circular_buffers(
            self, all_core_contexts: List[Dict[str, Any]]
        ) -> None:
            """Validate that all CircularBuffers have no pending blocks at end of execution.

            Args:
                all_core_contexts: List of per-core contexts containing CircularBuffers

            Raises:
                RuntimeError: If any CircularBuffer has pending blocks
            """
            errors = []
            for core_idx, core_context in enumerate(all_core_contexts):
                for key, value in core_context.items():
                    if isinstance(value, CircularBuffer):
                        try:
                            value.validate_no_pending_blocks()
                        except RuntimeError as e:
                            errors.append(f"core{core_idx}.{key}: {e}")

            if errors:
                raise RuntimeError(
                    "Kernel execution completed with incomplete CircularBuffer operations:\n"
                    + "\n".join(errors)
                )

        def _create_cooperative_generators(
            self,
            core: int,
            core_context: Dict[str, Any],
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> List[Tuple[str, str, ast.Module, Dict[str, Any], str, int, ThreadType]]:
            """Transform function sources for cooperative execution.

            Returns list of (name, func_name, transformed_ast, namespace, orig_file, orig_lineno, thread_type) tuples
            that the scheduler will compile and execute.

            The transformation inserts `yield (cb, operation)` before each wait()/reserve(),
            allowing the scheduler to:
            1. Receive operation info when generator yields
            2. Check if operation can proceed via can_wait()/can_reserve()
            3. Continue generator if unblocked, or switch to another if blocked
            4. Detect deadlock when all generators are blocked
            """
            sources: List[
                Tuple[str, str, ast.Module, Dict[str, Any], str, int, ThreadType]
            ] = []

            for name, tmpl in [
                ("compute", compute_func_tmpl),
                ("dm0", dm0_tmpl),
                ("dm1", dm1_tmpl),
            ]:
                # Get ThreadType directly from template's thread_type attribute
                thread_type = getattr(tmpl, "thread_type", None)
                if not isinstance(thread_type, ThreadType):
                    raise RuntimeError(
                        f"Template {tmpl} has invalid thread_type '{thread_type}'. "
                        f"Expected ThreadType enum (COMPUTE or DM)."
                    )

                # Bind template to core context
                bound_func = tmpl.bind(core_context)

                # Get source code - unwrap to get original function
                func = inspect.unwrap(bound_func)
                source = textwrap.dedent(inspect.getsource(func))

                # Capture original file and line number for error reporting
                orig_file = inspect.getsourcefile(func) or "<unknown>"
                orig_lineno = inspect.getsourcelines(func)[1]

                # Strip decorators from source (lines starting with @)
                source_lines = source.split("\n")
                func_start_idx = 0
                for i, line in enumerate(source_lines):
                    if line.strip().startswith("def "):
                        func_start_idx = i
                        break

                # Adjust orig_lineno to point to the def line, not the decorator
                orig_lineno += func_start_idx

                source = "\n".join(source_lines[func_start_idx:])

                # Transform: add yields before wait()/reserve() calls, returning AST
                transformed_ast = transform_wait_reserve_to_yield_ast(source)

                # Prepare namespace with core context and function's globals
                # Use original function's globals to get imports like 'ttl', 'copy', etc.
                # Closure variables are already in core_context (extracted in __call__)
                func_globals = func.__globals__
                namespace = {**func_globals, **core_context}

                # Return transformed AST and context for scheduler to execute
                # Use the original function's name (compute, dm0, dm1), not the wrapper's name (runner)
                func_name = func.__name__
                sources.append(
                    (
                        f"core{core}-{name}",
                        func_name,
                        transformed_ast,
                        namespace,
                        orig_file,
                        orig_lineno,
                        thread_type,
                    )
                )

            return sources

        def _run_round_robin_scheduler(
            self,
            sources: List[
                Tuple[str, str, ast.Module, Dict[str, Any], str, int, ThreadType]
            ],
        ) -> None:
            """Compile sources into generators and run them in round-robin.

            All generators across all cores are scheduled together in round-robin fashion,
            allowing true parallel execution simulation.

            Deadlock detection is handled by the scheduler:
            - When a generator yields (cb, operation), check if operation can proceed
            - If yes, continue that generator
            - If no, try other generators
            - If all generators are blocked, raise deadlock error
            """

            # Build mapping of generator names to original source locations
            orig_source_map: Dict[str, Tuple[str, int]] = {
                name: (orig_file, orig_lineno)
                for name, _, _, _, orig_file, orig_lineno, _ in sources
            }

            def _advance_generator(
                name: str,
                gen: Generator[None, None, None],
                thread_type: ThreadType,
                allow_completion: bool = False,
            ) -> Tuple[Any, bool]:
                """Advance a generator one step.

                Args:
                    name: Generator name for error messages
                    gen: Generator to advance
                    thread_type: Thread type for state machine context
                    allow_completion: If False, StopIteration is an error

                Returns:
                    (result, completed) where completed=True if generator finished
                """
                # Set thread context for state machine
                _set_current_thread_type(thread_type)

                try:
                    result: Any = next(gen)
                    return (result, False)
                except StopIteration:
                    if allow_completion:
                        return (None, True)
                    else:
                        raise RuntimeError(
                            f"{name}: Generator completed without yielding any operation. "
                            f"Transformed code should always yield before completing."
                        )
                except Exception as e:
                    # Generator raised an error - add source context
                    # Add original source location if available
                    if name in orig_source_map:
                        orig_file, func_def_line = orig_source_map[name]

                        # Extract line number and column from traceback
                        # Since we preserve original line numbers in AST, the traceback line
                        # is relative to the function definition
                        tb = traceback.extract_tb(e.__traceback__)
                        for frame in tb:
                            # Find the frame in the transformed code
                            if f"<{name}_transformed>" in frame.filename:
                                # frame.lineno is the line number within the transformed function
                                # which has line numbers preserved from the original source
                                # Adjust to absolute line in file: (func_def_line - 1) + frame.lineno
                                if frame.lineno is not None:
                                    actual_line = func_def_line - 1 + frame.lineno
                                    actual_col = getattr(frame, "colno", None) or 1

                                    # Use TTLangCompileError for pretty formatting
                                    TTLangCompileError = _get_ttlang_compile_error()
                                    compile_error = TTLangCompileError(
                                        f"{type(e).__name__}: {e}",
                                        source_file=orig_file,
                                        line=actual_line,
                                        col=actual_col,
                                    )
                                    print(f"\n❌ Error in {name}:")
                                    print(compile_error.format())
                                    print("-" * 50)
                                    raise RuntimeError(str(compile_error))

                                break
                        else:
                            # Fallback if we can't find the transformed frame
                            error_msg = f"{name}: {type(e).__name__}: {e}"
                            error_msg += f"\n  In function defined at: {orig_file}:{func_def_line}"
                            print(f"\n❌ {error_msg}")
                            tb_str = traceback.format_exc()
                            print("   traceback:")
                            print(tb_str)
                            print("-" * 50)
                            raise RuntimeError(error_msg)
                    else:
                        # No source mapping - print full traceback
                        error_msg = f"{name}: {type(e).__name__}: {e}"
                        print(f"\n❌ {error_msg}")
                        tb_str = traceback.format_exc()
                        print("   traceback:")
                        print(tb_str)
                        print("-" * 50)
                        raise RuntimeError(error_msg)

            # First, compile and execute all sources to create generators
            # active[name] = (generator, blocking_object, operation, thread_type)
            # blocking_object can be CircularBuffer or CopyTransaction - both support can_wait()/can_reserve()
            active: Dict[
                str, Tuple[Generator[None, None, None], Any, str, ThreadType]
            ] = {}

            # Track original source file and base line number for each generator
            # Used for mapping transformed line numbers back to original source
            orig_source_info: Dict[str, Tuple[str, int]] = {}

            for (
                name,
                func_name,
                transformed_ast,
                namespace,
                orig_file,
                orig_lineno,
                thread_type,
            ) in sources:
                # Store original source info for error reporting
                orig_source_info[name] = (orig_file, orig_lineno)
                # Compile transformed AST to bytecode (preserves line numbers)
                code_obj = compile(transformed_ast, f"<{name}_transformed>", "exec")

                # Execute to define the function in the namespace
                try:
                    exec(code_obj, namespace)
                except Exception as e:
                    # Error during function definition
                    error_msg = f"{name}: {type(e).__name__}: {e}"
                    raise RuntimeError(error_msg)

                # Call the function to create the generator
                try:
                    gen = namespace[func_name]()
                except Exception as e:
                    # Error during generator creation (e.g., exception in function body before first yield)
                    error_msg = f"{name}: {type(e).__name__}: {e}"
                    raise RuntimeError(error_msg)

                # Skip if function doesn't yield (just has 'pass' or no body)
                if gen is None:
                    continue

                # Run generator once until it yields (must yield at least once)
                result, _ = _advance_generator(
                    name, gen, thread_type, allow_completion=True
                )

                # If generator completed immediately without yielding, skip it
                if result is None:
                    continue

                # Generator yielded - check what it yielded
                match result:
                    case (blocking_obj, operation):
                        # Store (gen, blocking_obj, operation, thread_type) in active
                        # blocking_obj can be CircularBuffer or CopyTransaction
                        active[name] = (gen, blocking_obj, operation, thread_type)
                    case _:
                        # Unexpected yield value
                        raise RuntimeError(
                            f"{name}: Generator yielded unexpected value: {result!r}. "
                            f"Expected (blocking_object, operation) tuple."
                        )

            while active:
                # Track if any generator made progress in this iteration
                any_progress: bool = False
                # Track which generators have completed
                to_remove: List[str] = []

                # Try to advance each active generator
                for name in list(active.keys()):
                    gen, blocking_obj, blocked_op, thread_type = active[name]

                    # Check if operation can proceed
                    # blocking_obj supports can_wait() or can_reserve() depending on blocked_op
                    can_method = getattr(blocking_obj, f"can_{blocked_op}")
                    if not can_method():
                        # Still blocked, skip for now
                        continue

                    # Unblocked! Mark progress and continue running
                    any_progress = True

                    # Run this generator until it blocks or completes
                    while True:
                        result, completed = _advance_generator(
                            name, gen, thread_type, allow_completion=True
                        )

                        if completed:
                            # Generator completed successfully
                            to_remove.append(name)
                            break  # Exit inner while loop, try next generator

                        any_progress = True

                        # Check if generator yielded blocking operation info (blocking_obj, operation)
                        match result:
                            case (blocking_obj, operation):
                                # Check if operation can proceed
                                # blocking_obj can be CircularBuffer or CopyTransaction
                                can_method = getattr(
                                    blocking_obj, f"can_{operation}", None
                                )
                                if can_method and not can_method():
                                    # Operation would block - update state in active
                                    active[name] = (
                                        gen,
                                        blocking_obj,
                                        operation,
                                        thread_type,
                                    )
                                    break  # Exit inner while loop, try next generator
                                # else: operation can proceed, continue this generator
                            case _:
                                # Unexpected yield value
                                raise RuntimeError(
                                    f"{name}: Generator yielded unexpected value: {result!r}. "
                                    f"Expected (blocking_object, operation) tuple."
                                )

                        # If we get here, operation can proceed, continue this generator

                # Remove completed generators
                for name in to_remove:
                    del active[name]

                # Deadlock detection: no progress made and generators still active
                if not any_progress and active:
                    blocked_info: List[str] = []
                    for k, (gen, blocking_obj, op, _) in active.items():
                        # Extract file and line information from generator frame
                        frame = gen.gi_frame

                        # Get a descriptive name for the blocking object
                        obj_desc = ""
                        if hasattr(blocking_obj, "__class__"):
                            obj_type = blocking_obj.__class__.__name__
                            # Try to get a name or identifier for the object
                            if hasattr(blocking_obj, "_name"):
                                # CircularBuffer with name
                                obj_desc = f" on {obj_type}({blocking_obj._name})"
                            elif (
                                obj_type == "CopyTransaction"
                                and hasattr(blocking_obj, "_src")
                                and hasattr(blocking_obj, "_dst")
                            ):
                                # CopyTransaction - show src->dst
                                src_desc = self._get_obj_description(blocking_obj._src)
                                dst_desc = self._get_obj_description(blocking_obj._dst)
                                obj_desc = f" on {obj_type}({src_desc} -> {dst_desc})"
                            else:
                                obj_desc = f" on {obj_type}"

                        if frame and k in orig_source_info:
                            orig_file, base_lineno = orig_source_info[k]
                            # The frame's lineno is relative to the transformed code
                            # which starts at the 'def' line. Add base_lineno to map
                            # back to original source.
                            transformed_lineno = frame.f_lineno
                            # Subtract 1 because the first line of the function def is at base_lineno
                            orig_lineno = base_lineno + transformed_lineno - 1

                            blocked_info.append(
                                f"  {k}: blocked on {op}(){obj_desc} at {orig_file}:{orig_lineno}"
                            )
                        else:
                            blocked_info.append(f"  {k}: blocked on {op}(){obj_desc}")

                    raise RuntimeError(
                        f"Deadlock detected: all generators blocked\n"
                        + "\n".join(blocked_info)
                    )

        def _get_obj_description(self, obj: Any) -> str:
            """Get a brief description of an object for debugging output.

            Args:
                obj: Object to describe (Block, Pipe, Tensor, etc.)

            Returns:
                String description of the object
            """
            from .block import Block
            from .cb import CircularBuffer
            from .pipe import Pipe
            from .ttnnsim import Tensor

            match obj:
                case Block():
                    # For blocks, we don't have direct names, just show Block
                    return "Block"
                case CircularBuffer() if hasattr(obj, "_name"):
                    return obj._name
                case CircularBuffer():
                    return "CB"
                case Pipe():
                    return f"Pipe({obj.src}->{obj.dst})"
                case Tensor():
                    return "Tensor"
                case _:
                    return (
                        obj.__class__.__name__
                        if hasattr(obj, "__class__")
                        else str(obj)
                    )

    return ProgramImpl(*funcs)
