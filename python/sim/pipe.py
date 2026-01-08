# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pipe utilities for conditional execution based on pipe roles.
"""

from typing import Any, Callable, Generator, Iterator, List, Optional, Union

from .kernel import core, flatten_core_index
from .typedefs import CoreIndex, Pipe


# TODO: Core Ranges should probably be their own type
# TODO: Bring Pipe type here from typedefs.py
def _core_in_dst_range(
    dst_core_range: Union[CoreIndex, tuple[CoreIndex, CoreIndex]],
) -> bool:
    """Check if the current core is within the destination range.

    Args:
        dst_core_range: Either a single CoreIndex or a tuple of two CoreIndex defining a rectangle

    Returns:
        True if current core is in the range, False otherwise
    """
    match dst_core_range:
        case int():
            # Single 1D core - compare with 1D core index
            current_core_linear = core(dims=1)
            return current_core_linear == dst_core_range
        case (tuple() as first, tuple() as second):
            # Rectangular range - get coordinates matching the dimensionality
            dims = len(first)
            current_core_coords = core(dims=dims)

            # Check each dimension: min(first[i], second[i]) <= current_core_coords[i] <= max(first[i], second[i])
            match current_core_coords:
                case tuple():
                    for i in range(dims):
                        if not (
                            min(first[i], second[i])
                            <= current_core_coords[i]
                            <= max(first[i], second[i])
                        ):
                            return False
                    return True
                case _:
                    return False
        case tuple():
            # Single multi-dimensional core - get coordinates matching the dimensionality
            dims = len(dst_core_range)
            current_core_coords = core(dims=dims)
            return current_core_coords == dst_core_range
        case _:
            return False


def if_pipe_src(
    pipes: Union[Pipe, List[Pipe]],
    func: Callable[[Pipe], Optional[Iterator[Any]]],
) -> Generator[Any, None, None]:
    """Execute a function for each pipe if the current core is the source.

    Args:
        pipes: A single Pipe or list of Pipes to check
        func: Function to call with each pipe where current core is the source.
              The function receives the pipe as its argument.
              If func is a generator, yields from it to propagate cooperative scheduling.
    """
    match pipes:
        case Pipe():
            pipe_list = [pipes]
        case _:
            pipe_list = pipes
    current_core_linear = core(dims=1)  # Already returns linear index

    for pipe in pipe_list:
        pipe_src_linear = flatten_core_index(pipe.src_core)
        if current_core_linear == pipe_src_linear:
            result = func(pipe)
            # If callback returned an iterator/generator, yield from it
            if result is not None:
                yield from result


def if_pipe_dst(
    pipes: Union[Pipe, List[Pipe]],
    func: Callable[[Pipe], Optional[Iterator[Any]]],
) -> Generator[Any, None, None]:
    """Execute a function for each pipe if the current core is a destination.

    Args:
        pipes: A single Pipe or list of Pipes to check
        func: Function to call with each pipe where current core is a destination.
              The function receives the pipe as its argument.
              If func is a generator, yields from it to propagate cooperative scheduling.
    """
    match pipes:
        case Pipe():
            pipe_list = [pipes]
        case _:
            pipe_list = pipes

    for pipe in pipe_list:
        if _core_in_dst_range(pipe.dst_core_range):
            result = func(pipe)
            # If callback returned an iterator/generator, yield from it
            if result is not None:
                yield from result


def core_in_pipe(pipe: Pipe) -> bool:
    """Check if the current core is participating in the pipe (either source or destination).

    Args:
        pipe: The Pipe to check

    Returns:
        True if the current core is either the source or in the destination range, False otherwise.
    """
    current_core_linear = core(dims=1)  # Already returns linear index
    pipe_src_linear = flatten_core_index(pipe.src_core)

    # Check if current core is the source or is the destination range
    return current_core_linear == pipe_src_linear or _core_in_dst_range(
        pipe.dst_core_range
    )
