# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pipe utilities for conditional execution based on pipe roles.
"""

from typing import Callable, Union, List
from .typedefs import Pipe
from .kernel import core, flatten_core_index


def if_pipe_src(pipes: Union[Pipe, List[Pipe]], func: Callable[[Pipe], None]) -> None:
    """Execute a function for each pipe if the current core is the source.

    Args:
        pipes: A single Pipe or list of Pipes to check
        func: Function to call with each pipe where current core is the source.
              The function receives the pipe as its argument.
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
            func(pipe)


def if_pipe_dst(pipes: Union[Pipe, List[Pipe]], func: Callable[[Pipe], None]) -> None:
    """Execute a function for each pipe if the current core is a destination.

    Args:
        pipes: A single Pipe or list of Pipes to check
        func: Function to call with each pipe where current core is a destination.
              The function receives the pipe as its argument.
    """
    match pipes:
        case Pipe():
            pipe_list = [pipes]
        case _:
            pipe_list = pipes
    current_core_linear = core(dims=1)  # Already returns linear index

    for pipe in pipe_list:
        pipe_dst_linear = [flatten_core_index(dst) for dst in pipe.dst_core_range]
        if current_core_linear in pipe_dst_linear:
            func(pipe)


def core_in_pipe(pipe: Pipe) -> bool:
    """Check if the current core is participating in the pipe (either source or destination).

    Args:
        pipe: The Pipe to check

    Returns:
        True if the current core is either the source or in the destination range, False otherwise.
    """
    current_core_linear = core(dims=1)  # Already returns linear index
    pipe_src_linear = flatten_core_index(pipe.src_core)

    if current_core_linear == pipe_src_linear:
        return True

    pipe_dst_linear = [flatten_core_index(dst) for dst in pipe.dst_core_range]
    return current_core_linear in pipe_dst_linear
