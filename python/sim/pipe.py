# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pipe and PipeNet implementation for core-to-core communication.

This module provides:
- Pipe: Description of a data transfer from source to destination core(s)
- PipeNet: Network of pipes with conditional execution based on core role
- PipeIdentity classes: Wrappers exposing pipe source/destination information
"""

from typing import Callable, Generator, Generic, List, Union

from .kernel import core, flatten_core_index, grid_size
from .typedefs import CoreCoord, CoreRange, Pipe, DstT
from .xformyield import YieldedValue


class SrcPipeIdentity(Generic[DstT]):
    """
    Pipe identity for source cores.

    Provides access to destination information for pipes where the current core is the source.
    When inside an `if_src()` condition body, you are already on the source core,
    so this identity only exposes the destination.
    """

    def __init__(self, pipe: "Pipe[DstT]"):
        """Initialize with a pipe.

        Args:
            pipe: The underlying pipe object
        """
        self.pipe = pipe

    @property
    def dst(self) -> DstT:
        """Get the destination core coordinate(s) or core range.

        Returns:
            The destination specification from the pipe
        """
        return self.pipe.dst_core_range


class DstPipeIdentity:
    """
    Pipe identity for destination cores.

    Provides access to source information for pipes where the current core is a destination.
    When inside an `if_dst()` condition body, you are already on a destination core,
    so this identity only exposes the source.
    """

    def __init__(self, pipe: "Pipe"):  # type: ignore[type-arg]
        """Initialize with a pipe.

        Args:
            pipe: The underlying pipe object
        """
        self.pipe = pipe

    @property
    def src(self) -> CoreCoord:
        """Get the source core coordinate.

        Returns:
            The source core coordinate from the pipe
        """
        return self.pipe.src_core


def expand_core_range(core_range: CoreRange) -> List[tuple]:
    """Expand a CoreRange with slices into a list of concrete core coordinates.

    Args:
        core_range: A tuple containing indices and/or slices

    Returns:
        List of concrete core coordinate tuples

    Example:
        expand_core_range((0, slice(1, 4))) -> [(0, 1), (0, 2), (0, 3)]
        expand_core_range((slice(0, 2), slice(0, 2))) -> [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    # Get grid dimensions to determine slice bounds
    dims = len(core_range)
    grid_shape = grid_size(dims=dims)

    # Convert to tuple if grid_size returned a single value
    if not isinstance(grid_shape, tuple):
        grid_shape = (grid_shape,)

    # Convert each dimension to a list of indices
    dim_ranges = []
    for i, item in enumerate(core_range):
        if isinstance(item, slice):
            # Convert slice to range using grid bounds
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else grid_shape[i]
            step = item.step if item.step is not None else 1
            dim_ranges.append(list(range(start, stop, step)))
        else:
            # Single index
            dim_ranges.append([item])

    # Generate all combinations (Cartesian product)
    result = []

    def _cartesian_product(ranges, current=[]):
        if not ranges:
            result.append(tuple(current))
            return
        for value in ranges[0]:
            _cartesian_product(ranges[1:], current + [value])

    _cartesian_product(dim_ranges)
    return result


def _core_in_dst_range(
    dst_core_range: Union[CoreCoord, CoreRange, tuple[CoreCoord, CoreCoord]],
) -> bool:
    """Check if the current core is within the destination range.

    Args:
        dst_core_range: Destination specification - can be:
                       - Single CoreCoord (unicast)
                       - CoreRange with slices (multicast)
                       - Tuple of two CoreCoords (legacy rectangular range)

    Returns:
        True if current core is in the range, False otherwise
    """
    match dst_core_range:
        case int():
            # Single 1D core - compare with 1D core index
            current_core_linear = core(dims=1)
            return current_core_linear == dst_core_range

        case tuple() if any(isinstance(item, slice) for item in dst_core_range):
            # CoreRange with slices - expand and check membership
            dims = len(dst_core_range)
            current_core_coords = core(dims=dims)

            # Convert single value to tuple for comparison
            if not isinstance(current_core_coords, tuple):
                current_core_coords = (current_core_coords,)

            # Check each dimension
            for i, item in enumerate(dst_core_range):
                if isinstance(item, slice):
                    # Get grid dimension to determine bounds
                    grid_shape = grid_size(dims=dims)
                    if not isinstance(grid_shape, tuple):
                        grid_shape = (grid_shape,)

                    start = item.start if item.start is not None else 0
                    stop = item.stop if item.stop is not None else grid_shape[i]
                    step = item.step if item.step is not None else 1

                    if not (
                        start <= current_core_coords[i] < stop
                        and (current_core_coords[i] - start) % step == 0
                    ):
                        return False
                else:
                    # Fixed index
                    if current_core_coords[i] != item:
                        return False
            return True

        case (tuple() as first, tuple() as second):
            # Legacy rectangular range - get coordinates matching the dimensionality
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


class PipeNet(Generic[DstT]):
    """
    A network of pipes for organizing core-to-core communication patterns.

    PipeNet groups multiple pipes and provides conditional execution based on
    whether the current core is a source or destination in the network.
    """

    def __init__(self, pipes: "List[Pipe[DstT]]"):
        """Initialize pipe network with a list of pipes.

        Args:
            pipes: List of Pipe objects defining the communication pattern
        """
        self._pipes = pipes

    def if_src(
        self, cond_fun: Callable[[SrcPipeIdentity[DstT]], None]
    ) -> Generator[YieldedValue, None, None]:
        """Execute condition function for each pipe where current core is source.

        The condition function is called once for each pipe in the network where
        the current core matches the pipe's source core.

        Args:
            cond_fun: Function to execute with pipe identity as argument.
                     The function receives a SrcPipeIdentity that exposes the
                     destination via its .dst property.

        Yields:
            Tuples of (CircularBuffer, 'wait'|'reserve') or (CopyTransaction, 'wait')
            representing synchronization points from operations in the callback
        """
        current_core_linear = core(dims=1)

        for pipe in self._pipes:
            pipe_src_linear = flatten_core_index(pipe.src_core)
            if current_core_linear == pipe_src_linear:
                identity = SrcPipeIdentity[DstT](pipe)
                # Use yield from to propagate yields from the callback
                result = cond_fun(identity)
                if result is not None:
                    yield from result  # type: ignore[misc]

    def if_dst(
        self, cond_fun: Callable[[DstPipeIdentity], None]
    ) -> Generator[YieldedValue, None, None]:
        """Execute condition function for each pipe where current core is destination.

        The condition function is called once for each pipe in the network where
        the current core is in the pipe's destination range.

        Args:
            cond_fun: Function to execute with pipe identity as argument.
                     The function receives a DstPipeIdentity that exposes the
                     source via its .src property.

        Yields:
            Tuples of (CircularBuffer, 'wait'|'reserve') or (CopyTransaction, 'wait')
            representing synchronization points from operations in the callback
        """
        for pipe in self._pipes:
            if _core_in_dst_range(pipe.dst_core_range):
                identity = DstPipeIdentity(pipe)
                # Use yield from to propagate yields from the callback
                result = cond_fun(identity)
                if result is not None:
                    yield from result  # type: ignore[misc]
