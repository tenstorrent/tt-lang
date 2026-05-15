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
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ttl._pipenets import NodeCoord, NodeRange, OperationPipeNets, PipeUse

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

    def __post_init__(self) -> None:
        """Validate slice bounds in `dst` at construction time."""
        if isinstance(self.dst, tuple):
            for item, name in zip(self.dst, ("x", "y", "z")):
                _validate_dst_slice(item, name)

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


def _coord_to_tuple(coord: CoreCoord) -> Tuple[int, ...]:
    """Normalize a CoreCoord (int or tuple) to a tuple of ints."""
    if isinstance(coord, int):
        return (coord,)
    return tuple(coord)


def _axis_bounds(item: Any) -> Tuple[int, int]:
    """Half-open `(lo, hi)` bounds for one axis of a destination tuple.

    Slice bounds are assumed valid here — validated up front by
    `Pipe.__post_init__` via `_validate_dst_slice`.
    """
    if isinstance(item, slice):
        return (item.start, item.stop)
    return (item, item + 1)


def _validate_dst_slice(item: Any, name: str) -> None:
    """Raise ValueError if `item` is a malformed slice; no-op for ints."""
    if not isinstance(item, slice):
        return
    if item.start is None or item.stop is None:
        raise ValueError(
            f"dst {name} slice must have explicit start and stop, "
            f"got slice({item.start}, {item.stop})"
        )
    if not isinstance(item.start, int) or not isinstance(item.stop, int):
        raise ValueError(
            f"dst {name} slice bounds must be integers, "
            f"got slice({item.start}, {item.stop})"
        )
    if item.start >= item.stop:
        raise ValueError(
            f"dst {name} slice start must be < stop, "
            f"got slice({item.start}, {item.stop})"
        )
    if item.step is not None and item.step != 1:
        raise ValueError(
            f"dst {name} slice step must be 1 or None "
            f"(strided multicast is not supported), got step={item.step}"
        )


def _normalize_dst_rect(dst: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Half-open per-axis bounds for a multicast destination, or None if
    `dst` is unicast (no slices)."""
    if not isinstance(dst, tuple) or not any(isinstance(i, slice) for i in dst):
        return None
    return tuple(_axis_bounds(item) for item in dst)


def _pipe_to_pipe_use(pipe: "Pipe") -> PipeUse:
    """Convert a sim `Pipe` to a `PipeUse`.

    Slice bounds were already validated by `Pipe.__post_init__`; multicast
    rectangles are read directly from the `dst` slices without needing the
    operation grid.
    """
    src = NodeCoord(coords=_coord_to_tuple(pipe.src))
    rect = _normalize_dst_rect(pipe.dst)
    if rect is None:
        return PipeUse(src=src, dst=NodeCoord(coords=_coord_to_tuple(pipe.dst)))
    return PipeUse(
        src=src,
        dst=NodeRange(
            lo=tuple(lo for lo, _ in rect),
            hi=tuple(hi for _, hi in rect),
        ),
    )


def build_pipenets(pipe_nets: List["PipeNet"]) -> OperationPipeNets:
    """Build an OperationPipeNets from a list of unique PipeNet objects.

    Order is preserved: the first PipeNet in `pipe_nets` becomes id 0.
    """
    graph = OperationPipeNets()
    for net in pipe_nets:
        graph.add_pipe_net(_pipe_to_pipe_use(p) for p in net._pipes)
    return graph


def discover_pipe_nets_from_closures(*funcs: Any) -> List["PipeNet"]:
    """Walk function closures and return unique PipeNet objects in encounter order.

    PipeNets are deduplicated by `id()` so the same captured net referenced
    from multiple threads contributes one entry.
    """
    seen: dict = {}
    for func in funcs:
        if func is None:
            continue
        for net in _iter_pipe_nets_in_func(func):
            if id(net) not in seen:
                seen[id(net)] = net
    return list(seen.values())


def _iter_pipe_nets_in_func(func: Any) -> Iterable["PipeNet"]:
    # The Python module is an enclosing scope of an @ttl.operation
    # function, so module-scope PipeNets satisfy the spec's "enclosing
    # scope" rule and must be discovered. Walks closure cells and the
    # function's globals; the compiler's _build_operation_pipenets does
    # the same so validation and grid="auto" work extent agree.
    closure = getattr(func, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, PipeNet):
            yield value
    fn_globals = getattr(func, "__globals__", None) or {}
    for value in fn_globals.values():
        if isinstance(value, PipeNet):
            yield value


class PipeNet(Generic[DstT]):
    """
    A network of pipes for organizing core-to-core communication patterns.

    PipeNet groups multiple pipes and provides conditional execution based on
    whether the current core is a source or destination in the network.
    """

    def __init__(self, pipes: "List[Pipe[DstT]]"):
        # Validate at construction time by building a one-net graph and
        # delegating to OperationPipeNets.validate(). Single source of
        # truth for empty/overlap/mixed-kind rules; the same graph is
        # rebuilt and re-validated at operation build time.
        if not pipes:
            raise ValueError("PipeNet requires at least one pipe")
        graph = OperationPipeNets()
        graph.add_pipe_net(_pipe_to_pipe_use(p) for p in pipes)
        graph.validate()
        self._pipes = pipes

    def is_active(self) -> bool:
        """Return True if the current node participates in any pipe (source or destination).

        Useful for early-exit when only PipeNet participants should run thread body code.
        Must be called within a kernel context.

        Returns:
            True if the current core is a source or destination for at least one pipe.
        """
        return self.is_src() or self.is_dst()

    def is_src(self) -> bool:
        """Return True if the current node is the source of at least one pipe in this net."""
        current_core_linear = node(dims=1)
        for pipe in self._pipes:
            if flatten_core_index(pipe.src) == current_core_linear:
                return True
        return False

    def is_dst(self) -> bool:
        """Return True if the current node lies in the destination of at least one pipe."""
        for pipe in self._pipes:
            if core_in_dst_range(pipe.dst):
                return True
        return False

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
