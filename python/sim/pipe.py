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
from typing import Any, Callable, Generic, List, Optional, Set, Tuple, TypeVar, Union

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


# PipeNets constructed during the current operation's kernel body live in the
# per-greenlet SimulatorContext (see context_types.SimulatorContext.kernel_pipe_nets).
# These helpers wrap that storage so callers don't need to import context internals.


def clear_pipe_net_registry() -> None:
    """Reset the PipeNet registry. Called before each operation runs."""
    from .context import get_context

    get_context().kernel_pipe_nets.clear()


def _register_pipe_net(net: "PipeNet") -> None:
    """Append a constructed PipeNet to the active-set registry."""
    from .context import get_context

    get_context().kernel_pipe_nets.append(net)


def _coord_to_tuple(coord: CoreCoord) -> Tuple[int, ...]:
    """Normalize a CoreCoord (int or tuple) to a tuple of ints."""
    if isinstance(coord, int):
        return (coord,)
    return tuple(coord)


def _linearize(coord_tuple: Tuple[int, ...], grid: Tuple[int, ...]) -> int:
    """Linearize an n-d coord using the same row-major rule as flatten_core_index."""
    linear = coord_tuple[0]
    for i in range(1, len(coord_tuple)):
        linear = linear * grid[i] + coord_tuple[i]
    return linear


def _expand_range_with_grid(
    core_range: Tuple[Any, ...], grid: Tuple[int, ...]
) -> List[Tuple[int, ...]]:
    """Expand a CoreRange (tuple of ints and slices) to a list of coord tuples,
    using the explicit grid for slice bounds rather than relying on the
    grid_size() frame lookup. This avoids depending on the caller having a
    `grid` local variable."""
    dims = len(core_range)
    dim_ranges: List[List[int]] = []
    for i, item in enumerate(core_range):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else grid[i]
            step = item.step if item.step is not None else 1
            dim_ranges.append(list(range(start, stop, step)))
        else:
            dim_ranges.append([item])

    result: List[Tuple[int, ...]] = []

    def _cartesian(remaining: List[List[int]], current: List[int]) -> None:
        if not remaining:
            result.append(tuple(current) if dims > 1 else (current[0],))
            return
        for value in remaining[0]:
            _cartesian(remaining[1:], current + [value])

    _cartesian(dim_ranges, [])
    return result


def _expand_dst(dst: Any, grid: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Expand a pipe destination (CoreCoord or CoreRange) to a list of coord tuples."""
    if isinstance(dst, int):
        return [(dst,)]
    if not isinstance(dst, tuple):
        raise TypeError(f"Pipe.dst must be int or tuple, got {type(dst).__name__}")
    if any(isinstance(item, slice) for item in dst):
        return _expand_range_with_grid(dst, grid)
    return [tuple(dst)]


def compute_active_linear_nodes(
    grid: Tuple[int, ...],
) -> Optional[Set[int]]:
    """Return the set of linear core indices active in any registered PipeNet.

    Returns None when no PipeNets were registered, signaling that no active-set
    filtering should be applied (every core is active).
    """
    from .context import get_context

    registry = get_context().kernel_pipe_nets
    if not registry:
        return None

    active: Set[int] = set()
    for net in registry:
        for pipe in net._pipes:
            active.add(_linearize(_coord_to_tuple(pipe.src), grid))
            for dst_tuple in _expand_dst(pipe.dst, grid):
                active.add(_linearize(dst_tuple, grid))
    return active


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
        _register_pipe_net(self)

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
