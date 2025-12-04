# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pipe utilities for conditional execution based on pipe roles.
"""

from typing import Callable, Union, List
from .typedefs import Pipe
from .kernel import core


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
    current_core = core(dims=1)

    for pipe in pipe_list:
        if current_core == pipe.src_core:
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
    current_core = core(dims=1)

    for pipe in pipe_list:
        if current_core in pipe.dst_core_range:
            func(pipe)
