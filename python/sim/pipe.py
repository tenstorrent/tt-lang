# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pipe and PipeNet implementation for node-to-node communication.

This module provides:
- Pipe: Description of a data transfer from source to destination node(s)
- PipeNet: Network of pipes with conditional execution based on node role
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

from ttl._pipenets import NodeCoord as PipeNodeCoord
from ttl._pipenets import NodeRange as PipeNodeRange
from ttl._pipenets import OperationPipeNets, PipeUse

from .nodecontext import node, flatten_node_index, grid_size
from .typedefs import NodeCoord, NodeRange

# Type variable for Pipe destination type
DstT = TypeVar("DstT", NodeCoord, NodeRange)

# Union of valid destination types for Pipe
AnyDst = Union[NodeCoord, NodeRange]


@dataclass(frozen=True)
class Pipe(Generic[DstT]):
    """
    Represents a pipe for NoC communication.

    A pipe describes a data transfer from a source node to destination node(s).
    Can be used for both unicast (single destination) and multicast (multiple destinations).

    Type Parameters:
        DstT: The type of the destination - NodeCoord or NodeRange

    Attributes:
        src: Node coordinates of the source/sender. Can be:
             - Index: Single 1D node (e.g., 0, 1, 2)
             - Tuple[Index, ...]: Multi-dimensional node (e.g., (0, 1), (1, 2, 3))

        dst: Destination specification. Can be:
             - NodeCoord: Single destination node (unicast)
               Example: 5 or (1, 2)
             - NodeRange: Range of destination nodes using slices (multicast)
               Example: (0, slice(1, 4)) means nodes (0,1), (0,2), (0,3)
    """

    src: NodeCoord
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
AnyPipe = Union[Pipe[NodeCoord], Pipe[NodeRange]]


class SrcPipeIdentity(Generic[DstT]):
    """
    Pipe identity for source nodes.

    Provides access to destination information for pipes where the current node is the source.
    When inside an `if_src()` condition body, you are already on the source node,
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
        """Get the destination node coordinate(s) or node range.

        Returns:
            The destination specification from the pipe
        """
        return self.pipe.dst


# Union of SrcPipeIdentity instances with different destination types
AnySrcPipeIdentity = Union[SrcPipeIdentity[NodeCoord], SrcPipeIdentity[NodeRange]]


class DstPipeIdentity:
    """
    Pipe identity for destination nodes.

    Provides access to source information for pipes where the current node is a destination.
    When inside an `if_dst()` condition body, you are already on a destination node,
    so this identity only exposes the source.
    """

    def __init__(self, pipe: "Pipe[Any]"):
        """Initialize with a pipe.

        Args:
            pipe: The underlying pipe object
        """
        self.pipe = pipe

    @property
    def src(self) -> NodeCoord:
        """Get the source node coordinate.

        Returns:
            The source node coordinate from the pipe
        """
        return self.pipe.src


def expand_node_range(node_range: NodeRange) -> List[NodeCoord]:
    """Expand a NodeRange with slices into a list of concrete node coordinates.

    Args:
        node_range: A tuple containing indices and/or slices

    Returns:
        List of concrete node coordinate tuples

    Example:
        expand_node_range((0, slice(1, 4))) -> [(0, 1), (0, 2), (0, 3)]
        expand_node_range((slice(0, 2), slice(0, 2))) -> [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    # Get grid dimensions to determine slice bounds
    dims = len(node_range)
    grid_shape = grid_size(dims=dims)

    # Convert to tuple if grid_size returned a single value
    match grid_shape:
        case tuple():
            pass
        case _:
            grid_shape = (grid_shape,)

    # Convert each dimension to a list of indices
    dim_ranges: List[List[int]] = []
    for i, item in enumerate(node_range):
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
    result: List[NodeCoord] = []

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


def node_in_dst_range(
    dst_node_range: AnyDst,
) -> bool:
    """Check if the current node is within the destination range.

    Args:
        dst_node_range: Destination specification - can be:
                       - Single NodeCoord (unicast)
                       - NodeRange with slices (multicast)

    Returns:
        True if current node is in the range, False otherwise
    """
    match dst_node_range:
        case int():
            # Single 1D node - compare with 1D node index
            current_node_linear = node(dims=1)
            return current_node_linear == dst_node_range

        case tuple() if any(type(item) is slice for item in dst_node_range):
            # NodeRange with slices - expand and check membership
            dims = len(dst_node_range)
            current_node_coords = node(dims=dims)

            # Convert single value to tuple for comparison
            match current_node_coords:
                case tuple():
                    pass
                case _:
                    current_node_coords = (current_node_coords,)

            # Check each dimension
            for i, item in enumerate(dst_node_range):
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
                            start <= current_node_coords[i] < stop
                            and (current_node_coords[i] - start) % step == 0
                        ):
                            return False
                    case _:
                        # Fixed index
                        if current_node_coords[i] != item:
                            return False
            return True

        case tuple():
            # Single multi-dimensional node - get coordinates matching the dimensionality
            dims = len(dst_node_range)
            current_node_coords = node(dims=dims)
            return current_node_coords == dst_node_range


def _coord_to_tuple(coord: NodeCoord) -> Tuple[int, ...]:
    """Normalize a NodeCoord (int or tuple) to a tuple of ints."""
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
    src = PipeNodeCoord(coords=_coord_to_tuple(pipe.src))
    rect = _normalize_dst_rect(pipe.dst)
    if rect is None:
        return PipeUse(src=src, dst=PipeNodeCoord(coords=_coord_to_tuple(pipe.dst)))
    return PipeUse(
        src=src,
        dst=PipeNodeRange(
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
    from multiple kernels contributes one entry.
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
    A network of pipes for organizing node-to-node communication patterns.

    PipeNet groups multiple pipes and provides conditional execution based on
    whether the current node is a source or destination in the network.
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

        Useful for early-exit when only PipeNet participants should run kernel body code.
        Must be called within a kernel context.

        Returns:
            True if the current node is a source or destination for at least one pipe.
        """
        return self.is_src() or self.is_dst()

    def is_src(self) -> bool:
        """Return True if the current node is the source of at least one pipe in this net."""
        current_node_linear = node(dims=1)
        for pipe in self._pipes:
            if flatten_node_index(pipe.src) == current_node_linear:
                return True
        return False

    def is_dst(self) -> bool:
        """Return True if the current node lies in the destination of at least one pipe."""
        for pipe in self._pipes:
            if node_in_dst_range(pipe.dst):
                return True
        return False

    def if_src(self, cond_fun: Callable[[SrcPipeIdentity[DstT]], None]) -> None:
        """Execute condition function for each pipe where current node is source.

        The condition function is called once for each pipe in the network where
        the current node matches the pipe's source.

        Args:
            cond_fun: Function to execute with pipe identity as argument.
                     The function receives a SrcPipeIdentity that exposes the
                     destination via its .dst property.
        """
        current_node_linear = node(dims=1)

        for pipe in self._pipes:
            pipe_src_linear = flatten_node_index(pipe.src)
            if current_node_linear == pipe_src_linear:
                identity = SrcPipeIdentity[DstT](pipe)
                cond_fun(identity)

    def if_dst(self, cond_fun: Callable[[DstPipeIdentity], None]) -> None:
        """Execute condition function for each pipe where current node is destination.

        The condition function is called once for each pipe in the network where
        the current node is in the pipe's destination range.

        Args:
            cond_fun: Function to execute with pipe identity as argument.
                     The function receives a DstPipeIdentity that exposes the
                     source via its .dst property.
        """
        for pipe in self._pipes:
            if node_in_dst_range(pipe.dst):
                identity = DstPipeIdentity(pipe)
                cond_fun(identity)
