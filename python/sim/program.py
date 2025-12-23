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
import traceback
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Tuple, Protocol, Generator
from types import CellType, FunctionType

import torch

from .cb import CircularBuffer
from .cbapi import CBAPI
from .tensoraccessor import TensorAccessor
from .xformyield import transform_wait_reserve_to_yield


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


def Program(*funcs: BindableTemplate) -> Any:
    """Program class that combines compute and data movement functions.

    Args:
        *funcs: Compute and data movement function templates
    """

    class ProgramImpl:
        def __init__(
            self,
            *functions: BindableTemplate,
        ):
            self.functions = functions
            self.context: Dict[str, Any] = {}

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

                # Get source code - unwrap to get original function
                func = inspect.unwrap(bound_func)
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

            def _advance_generator(
                name: str,
                gen: Generator[None, None, None],
                allow_completion: bool = False,
            ) -> Tuple[Any, bool]:
                """Advance a generator one step.

                Args:
                    name: Generator name for error messages
                    gen: Generator to advance
                    allow_completion: If False, StopIteration is an error

                Returns:
                    (result, completed) where completed=True if generator finished
                """
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
                    # Generator raised an error
                    tb_str = traceback.format_exc()
                    error_msg = f"{name}: {type(e).__name__}: {e}"
                    print(f"\n❌ {error_msg}")
                    print("   traceback:")
                    print(tb_str)
                    print("-" * 50)
                    raise RuntimeError(error_msg)

            # First, compile and execute all sources to create generators
            # active[name] = (generator, cb_object, operation)
            active: Dict[str, Tuple[Generator[None, None, None], Any, str]] = {}

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

                # Run generator once until it yields (must yield at least once)
                result, _ = _advance_generator(name, gen, allow_completion=True)

                # If generator completed immediately without yielding, skip it
                if result is None:
                    continue

                # Generator yielded - check what it yielded
                match result:
                    case (cb_obj, operation):
                        # Store (gen, cb_obj, operation) in active
                        active[name] = (gen, cb_obj, operation)
                    case _:
                        # Unexpected yield value
                        raise RuntimeError(
                            f"{name}: Generator yielded unexpected value: {result!r}. "
                            f"Expected (cb_object, operation) tuple."
                        )

            while active:
                # Track if any generator made progress in this iteration
                any_progress: bool = False
                # Track which generators have completed
                to_remove: List[str] = []

                # Try to advance each active generator
                for name in list(active.keys()):
                    gen, blocked_cb, blocked_op = active[name]

                    # Check if operation can proceed
                    can_method = getattr(blocked_cb, f"can_{blocked_op}")
                    if not can_method():
                        # Still blocked, skip for now
                        continue

                    # Unblocked! Mark progress and continue running
                    any_progress = True

                    # Run this generator until it blocks or completes
                    while True:
                        result, completed = _advance_generator(
                            name, gen, allow_completion=True
                        )

                        if completed:
                            # Generator completed successfully
                            to_remove.append(name)
                            break  # Exit inner while loop, try next generator

                        any_progress = True

                        # Check if generator yielded blocking operation info (cb, operation)
                        match result:
                            case (cb_obj, operation):
                                # Check if operation can proceed
                                can_method = getattr(cb_obj, f"can_{operation}", None)
                                if can_method and not can_method():
                                    # Operation would block - update state in active
                                    active[name] = (gen, cb_obj, operation)
                                    break  # Exit inner while loop, try next generator
                                # else: operation can proceed, continue this generator
                            case _:
                                # Unexpected yield value
                                raise RuntimeError(
                                    f"{name}: Generator yielded unexpected value: {result!r}. "
                                    f"Expected (cb_object, operation) tuple."
                                )

                        # If we get here, operation can proceed, continue this generator

                # Remove completed generators
                for name in to_remove:
                    del active[name]

                # Deadlock detection: no progress made and generators still active
                if not any_progress and active:
                    blocked_info: List[str] = [
                        f"{k}: {op}()" for k, (_, _, op) in active.items()
                    ]
                    raise RuntimeError(
                        f"Deadlock detected: all generators blocked\n"
                        + "\n".join(blocked_info)
                    )

    return ProgramImpl(*funcs)
