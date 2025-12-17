# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Program execution framework for multi-core simulation.

This module provides the core execution framework for running compute and data movement
functions across multiple cores with proper context binding and error handling.
"""

import types
import copy
import threading
import traceback
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Tuple, Protocol, Generator
from types import CellType, FunctionType
from enum import Enum

import torch

from .cb import CircularBuffer
from .cbapi import CBAPI
from .tensoraccessor import TensorAccessor
from .xformyield import transform_wait_reserve_to_yield


class ExecutionMode(Enum):
    """Execution mode for Program."""

    THREADED = "threaded"  # Original concurrent execution with threads
    COOPERATIVE = "cooperative"  # Round-robin cooperative scheduling with yields


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


def Program(
    *funcs: BindableTemplate, execution_mode: ExecutionMode = ExecutionMode.THREADED
) -> Any:
    """Program class that combines compute and data movement functions.

    Args:
        *funcs: Compute and data movement function templates
        execution_mode: Execution mode (THREADED or COOPERATIVE)
    """

    class ProgramImpl:
        def __init__(
            self,
            *functions: BindableTemplate,
            execution_mode: ExecutionMode = ExecutionMode.THREADED,
        ):
            self.functions = functions
            self.context: Dict[str, Any] = {}
            self.mode = execution_mode

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                # capture locals and globals from the caller (eltwise_add)
                # Check globals first for decorator-injected variables, then locals
                self.context = {}
                # Add relevant items from globals (like grid, granularity from decorators)
                caller_globals = frame.f_back.f_globals
                for key in ["grid", "granularity"]:
                    if key in caller_globals:
                        self.context[key] = caller_globals[key]
                # Add locals (which take precedence)
                self.context.update(frame.f_back.f_locals)

            grid = self.context.get("grid", (1, 1))
            # Calculate total cores for any dimension grid
            total_cores = 1
            for dim_size in grid:
                total_cores *= dim_size

            compute_func_tmpl, dm0_tmpl, dm1_tmpl = self.functions

            # Choose execution strategy based on mode
            if self.mode == ExecutionMode.COOPERATIVE:
                self._run_cooperative(
                    total_cores, compute_func_tmpl, dm0_tmpl, dm1_tmpl
                )
            else:
                self._run_threaded(total_cores, compute_func_tmpl, dm0_tmpl, dm1_tmpl)

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
                match value:
                    case torch.Tensor() | TensorAccessor():
                        core_context[key] = value
                        memo[id(value)] = value
                    case CircularBuffer():
                        # create a fresh CB for this core
                        core_context[key] = CircularBuffer(
                            shape=value.shape,
                            buffer_factor=value.buffer_factor,
                            api=api,
                        )
                    case _:
                        core_context[key] = copy.deepcopy(value, memo)

            # also make the core number visible
            core_context["_core"] = core

            return core_context

        def _run_threaded(
            self,
            total_cores: int,
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> None:
            """Original threaded execution mode - all cores run in parallel."""
            # collect user-facing errors here
            thread_results: List[Tuple[str, Exception, str]] = []
            all_threads: List[threading.Thread] = []

            def run_func_in_thread(name: str, func_factory: Callable[[], Any]) -> None:
                try:
                    func_factory()  # Execute the function factory directly
                except Exception as e:
                    tb_str = traceback.format_exc()
                    thread_results.append((name, e, tb_str))

            # Create and start threads for all cores
            for core in range(total_cores):
                # build per-core context
                core_context = self._build_core_context(core)

                # bind per-core
                core_dm0 = dm0_tmpl.bind(core_context)
                core_compute = compute_func_tmpl.bind(core_context)
                core_dm1 = dm1_tmpl.bind(core_context)

                # Create threads for this core's functions
                t_dm0 = threading.Thread(
                    target=run_func_in_thread, args=(f"core{core}-dm0", core_dm0)
                )
                t_comp = threading.Thread(
                    target=run_func_in_thread,
                    args=(f"core{core}-compute", core_compute),
                )
                t_dm1 = threading.Thread(
                    target=run_func_in_thread, args=(f"core{core}-dm1", core_dm1)
                )

                all_threads.extend([t_dm0, t_comp, t_dm1])

            # Start all threads across all cores
            for t in all_threads:
                t.start()

            # Wait for all threads to finish
            for t in all_threads:
                t.join()

            # Check if any failed
            if thread_results:
                errors: List[str] = []
                for name, e, tb_str in thread_results:
                    # print a user-readable header
                    print(f"\n❌ {name} failed")
                    print(f"   error type   : {type(e).__name__}")
                    print(f"   error message: {e}")
                    print("   traceback:")
                    print(tb_str)
                    print("-" * 50)

                    # add to final aggregation (short)
                    errors.append(f"{name}: {type(e).__name__}: {e}")

                raise RuntimeError("One or more threads failed:\n" + "\n".join(errors))

            return None

        def _run_cooperative(
            self,
            total_cores: int,
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> None:
            """Cooperative scheduling execution mode - all cores run in round-robin."""

            # Create transformed sources for all cores
            all_sources: List[Tuple[str, str, str, Dict[str, Any]]] = []

            for core in range(total_cores):
                # build per-core context
                core_context = self._build_core_context(core)

                # Transform functions to sources with yields
                sources = self._create_cooperative_generators(
                    core, core_context, compute_func_tmpl, dm0_tmpl, dm1_tmpl
                )

                all_sources.extend(sources)

            # Run round-robin scheduler across all cores
            self._run_round_robin_scheduler(all_sources)

        def _create_cooperative_generators(
            self,
            core: int,
            core_context: Dict[str, Any],
            compute_func_tmpl: BindableTemplate,
            dm0_tmpl: BindableTemplate,
            dm1_tmpl: BindableTemplate,
        ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
            """Transform function sources for cooperative execution.

            Returns list of (name, func_name, transformed_source, namespace) tuples
            that the scheduler will compile and execute.

            The transformation inserts `yield (cb, operation)` before each wait()/reserve(),
            allowing the scheduler to:
            1. Receive operation info when generator yields
            2. Check if operation can proceed via can_wait()/can_reserve()
            3. Continue generator if unblocked, or switch to another if blocked
            4. Detect deadlock when all generators are blocked
            """
            sources: List[Tuple[str, str, str, Dict[str, Any]]] = []

            for name, tmpl in [
                ("compute", compute_func_tmpl),
                ("dm0", dm0_tmpl),
                ("dm1", dm1_tmpl),
            ]:
                # Bind template to core context
                bound_func = tmpl.bind(core_context)

                # Get source code - use original function if available to avoid decorator wrapper
                func = getattr(bound_func, "_func", bound_func)
                source = textwrap.dedent(inspect.getsource(func))

                # Strip decorators from source (lines starting with @)
                source_lines = source.split("\n")
                func_start_idx = 0
                for i, line in enumerate(source_lines):
                    if line.strip().startswith("def "):
                        func_start_idx = i
                        break
                source = "\n".join(source_lines[func_start_idx:])

                # Transform: add yields before wait()/reserve() calls
                transformed_source = transform_wait_reserve_to_yield(source)

                # Prepare namespace with core context and function's globals
                # Use original function's globals to get imports like 'ttl', 'copy', etc.
                func_globals = func.__globals__
                namespace = {**func_globals, **core_context}

                # Return transformed source and context for scheduler to execute
                # Use the original function's name (compute, dm0, dm1), not the wrapper's name (runner)
                func_name = func.__name__
                sources.append(
                    (f"core{core}-{name}", func_name, transformed_source, namespace)
                )

            return sources

        def _run_round_robin_scheduler(
            self, sources: List[Tuple[str, str, str, Dict[str, Any]]]
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

            # First, compile and execute all sources to create generators
            active: Dict[str, Generator[None, None, None]] = {}

            for name, func_name, transformed_source, namespace in sources:
                # Compile transformed source to bytecode
                code_obj = compile(transformed_source, f"<{name}_transformed>", "exec")

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

                active[name] = gen

            # Track blocked generators: {name: (generator, cb_object, operation)}
            blocked: Dict[str, Tuple[Generator[Any, None, None], Any, str]] = {}

            while active or blocked:
                # First, check if any blocked generators can now proceed
                unblocked: List[str] = []
                for name, (gen, cb_obj, operation) in list(blocked.items()):
                    can_method = getattr(cb_obj, f"can_{operation}")
                    if can_method():
                        unblocked.append(name)
                        active[name] = gen

                # Remove unblocked from blocked dict
                for name in unblocked:
                    del blocked[name]

                # Check for deadlock: all generators blocked and none can proceed
                if not active and blocked:
                    blocked_info = [
                        f"{name}: {op}()" for name, (_, _, op) in blocked.items()
                    ]
                    raise RuntimeError(
                        f"Deadlock detected: all generators blocked\n"
                        + "\n".join(blocked_info)
                    )

                # Try to advance each active generator
                # Run each generator until it blocks or completes
                for name in list(active.keys()):
                    gen = active[name]

                    # Keep advancing this generator until it blocks or completes
                    while True:
                        try:
                            # Try to advance the generator one step
                            result: Any = next(gen)

                            # Check if generator yielded blocking operation info (cb, operation)
                            match result:
                                case (cb_obj, operation):
                                    # Check if operation can proceed
                                    can_method = getattr(
                                        cb_obj, f"can_{operation}", None
                                    )
                                    if can_method and not can_method():
                                        # Operation would block - mark as blocked and move to next generator
                                        blocked[name] = (gen, cb_obj, operation)
                                        del active[name]
                                        break  # Exit inner while loop, try next generator
                                    # else: operation can proceed, continue this generator
                                case _:
                                    # Not a 2-tuple or None, treat as normal yield
                                    pass

                            # If we get here, either it didn't yield blocking info or it can proceed
                            # Continue running this generator

                        except StopIteration:
                            # Generator completed successfully
                            del active[name]
                            break  # Exit inner while loop, try next generator

                        except Exception as e:
                            # Generator raised an error
                            tb_str = traceback.format_exc()
                            error_msg = f"{name}: {type(e).__name__}: {e}"
                            print(f"\n❌ {error_msg}")
                            print("   traceback:")
                            print(tb_str)
                            print("-" * 50)
                            raise RuntimeError(error_msg)

    return ProgramImpl(*funcs, execution_mode=execution_mode)
