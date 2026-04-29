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

from dataclasses import dataclass
from typing import Any, Callable, Generic, List, TypeVar, Union

from .corecontext import node, flatten_core_index, grid_size
from .typedefs import CoreCoord, CoreRange

# Type variable for Pipe destination type
DstT = TypeVar("DstT", CoreCoord, CoreRange)

# Union of valid destination types for Pipe
AnyDst = Union[CoreCoord, CoreRange]


@dataclass(frozen=True)
class Pipe(Generic[DstT]):
    """
    Represents a pipe for NoC communication.

    A pipe describes a data transfer from a source core to destination core(s).
    Can be used for both unicast (single destination) and multicast (multiple destinations).

    Type Parameters:
        DstT: The type of the destination - CoreCoord or CoreRange

    Attributes:
        src: Core coordinates of the source/sender. Can be:
             - Index: Single 1D core (e.g., 0, 1, 2)
             - Tuple[Index, ...]: Multi-dimensional core (e.g., (0, 1), (1, 2, 3))

        dst: Destination specification. Can be:
             - CoreCoord: Single destination core (unicast)
               Example: 5 or (1, 2)
             - CoreRange: Range of destination cores using slices (multicast)
               Example: (0, slice(1, 4)) means cores (0,1), (0,2), (0,3)
    """

    src: CoreCoord
    dst: DstT

    def has_current_node(self) -> bool:
        """Check if the current core participates in this pipe (either as source or destination).

        This is useful for early-exit patterns where non-participating cores should skip work.
        Must be called within a kernel context.

        Returns:
            True if the current core is either the source or in the destination range.
        """
        # Check if current core is the source
        current_core_linear = node(dims=1)
        pipe_src_linear = flatten_core_index(self.src)
        if current_core_linear == pipe_src_linear:
            return True

        return core_in_dst_range(self.dst)

    def __hash__(self) -> int:
        """Custom hash implementation to handle slices and nested tuples."""

        def make_hashable(obj: Any) -> Any:
            """Convert potentially unhashable objects to hashable equivalents."""
            match obj:
                case slice():
                    return (obj.start, obj.stop, obj.step)  # type: ignore[return-value]
                case list():
                    return tuple(make_hashable(item) for item in obj)  # type: ignore[misc]
                case tuple():
                    return tuple(make_hashable(item) for item in obj)  # type: ignore[misc]
                case _:
                    return obj

        return hash((make_hashable(self.src), make_hashable(self.dst)))


# Union of Pipe instances with different destination types
AnyPipe = Union[Pipe[CoreCoord], Pipe[CoreRange]]


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
        return self.pipe.dst


# Union of SrcPipeIdentity instances with different destination types
AnySrcPipeIdentity = Union[SrcPipeIdentity[CoreCoord], SrcPipeIdentity[CoreRange]]


class DstPipeIdentity:
    """
    Pipe identity for destination cores.

    Provides access to source information for pipes where the current core is a destination.
    When inside an `if_dst()` condition body, you are already on a destination core,
    so this identity only exposes the source.
    """

    def __init__(self, pipe: "Pipe[Any]"):
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
        return self.pipe.src


def expand_core_range(core_range: CoreRange) -> List[CoreCoord]:
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
    match grid_shape:
        case tuple():
            pass
        case _:
            grid_shape = (grid_shape,)

    # Convert each dimension to a list of indices
    dim_ranges: List[List[int]] = []
    for i, item in enumerate(core_range):
        match item:
            case slice():
                # Convert slice to range using grid bounds
                start = item.start if item.start is not None else 0
                stop = item.stop if item.stop is not None else grid_shape[i]
                step = item.step if item.step is not None else 1
                dim_ranges.append(list(range(start, stop, step)))
            case _:
                # Single index
                dim_ranges.append([item])

    # Generate all combinations (Cartesian product)
    result: List[CoreCoord] = []

    def _cartesian_product(ranges: List[List[int]], current: List[int] = []) -> None:
        if not ranges:
            # For 1D, append single value; for multi-D, append tuple
            if dims == 1:
                result.append(current[0])
            else:
                result.append(tuple(current))
            return
        for value in ranges[0]:
            _cartesian_product(ranges[1:], current + [value])

    _cartesian_product(dim_ranges)
    return result


def core_in_dst_range(
    dst_core_range: AnyDst,
) -> bool:
    """Check if the current core is within the destination range.

    Args:
        dst_core_range: Destination specification - can be:
                       - Single CoreCoord (unicast)
                       - CoreRange with slices (multicast)

    Returns:
        True if current core is in the range, False otherwise
    """
    match dst_core_range:
        case int():
            # Single 1D core - compare with 1D core index
            current_core_linear = node(dims=1)
            return current_core_linear == dst_core_range

        case tuple() if any(type(item) is slice for item in dst_core_range):
            # CoreRange with slices - expand and check membership
            dims = len(dst_core_range)
            current_core_coords = node(dims=dims)

            # Convert single value to tuple for comparison
            match current_core_coords:
                case tuple():
                    pass
                case _:
                    current_core_coords = (current_core_coords,)

            # Check each dimension
            for i, item in enumerate(dst_core_range):
                match item:
                    case slice():
                        # Get grid dimension to determine bounds
                        grid_shape = grid_size(dims=dims)
                        match grid_shape:
                            case tuple():
                                pass
                            case _:
                                grid_shape = (grid_shape,)

                        start = item.start if item.start is not None else 0
                        stop = item.stop if item.stop is not None else grid_shape[i]
                        step = item.step if item.step is not None else 1

                        if not (
                            start <= current_core_coords[i] < stop
                            and (current_core_coords[i] - start) % step == 0
                        ):
                            return False
                    case _:
                        # Fixed index
                        if current_core_coords[i] != item:
                            return False
            return True

        case tuple():
            # Single multi-dimensional core - get coordinates matching the dimensionality
            dims = len(dst_core_range)
            current_core_coords = node(dims=dims)
            return current_core_coords == dst_core_range


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

    def if_src(self, cond_fun: Callable[[SrcPipeIdentity[DstT]], None]) -> None:
        """Execute condition function for each pipe where current core is source.

        The condition function is called once for each pipe in the network where
        the current core matches the pipe's source core.

        Args:
            cond_fun: Function to execute with pipe identity as argument.
                     The function receives a SrcPipeIdentity that exposes the
                     destination via its .dst property.
        """
        current_core_linear = node(dims=1)

        for pipe in self._pipes:
            pipe_src_linear = flatten_core_index(pipe.src)
            if current_core_linear == pipe_src_linear:
                identity = SrcPipeIdentity[DstT](pipe)
                cond_fun(identity)

    def if_dst(self, cond_fun: Callable[[DstPipeIdentity], None]) -> None:
        """Execute condition function for each pipe where current core is destination.

        The condition function is called once for each pipe in the network where
        the current core is in the pipe's destination range.

        Args:
            cond_fun: Function to execute with pipe identity as argument.
                     The function receives a DstPipeIdentity that exposes the
                     source via its .dst property.
        """
        for pipe in self._pipes:
            if core_in_dst_range(pipe.dst):
                identity = DstPipeIdentity(pipe)
                cond_fun(identity)
