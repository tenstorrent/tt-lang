# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Node coordinate and grid utilities for kernel execution contexts.

Provides functions to query the current node index and grid size from within
a running kernel, and to convert multi-dimensional node coordinates to a
linear index.
"""

import inspect
from typing import Any, List, Union

from .typedefs import NodeCoord, Index, Shape, Size


def _get_from_frame(var_name: str, error_msg: str) -> Any:
    """Helper to walk up the call stack and find a variable.

    Searches through the call stack (locals first, then globals) to find
    a variable by name. This is used by functions like grid_size(), node(),
    and flatten_node_index() to access context variables like 'grid' and '_node'.

    Args:
        var_name: Name of the variable to search for
        error_msg: Error message to raise if not found

    Returns:
        The value of the variable if found

    Raises:
        RuntimeError: If the variable is not found in any frame
    """
    frame = inspect.currentframe()
    # Start from the caller's caller frame (skip _get_from_frame and the immediate caller)
    current_frame = frame.f_back.f_back if frame and frame.f_back else None

    while current_frame:
        # Check locals first (takes precedence)
        if var_name in current_frame.f_locals:
            return current_frame.f_locals[var_name]
        # Then check globals
        if var_name in current_frame.f_globals:
            return current_frame.f_globals[var_name]
        current_frame = current_frame.f_back

    raise RuntimeError(error_msg)


def node_coord_from_linear(linear_node: Index, grid: Shape) -> tuple[int, ...]:
    """Decompose a linear node index into per-axis grid coordinates.

    Uses the same row-major, outermost-first convention as :func:`node`: for a
    grid ``(g0, g1, ..., g_{k-1})`` the returned coordinate ``(c0, ..., c_{k-1})``
    satisfies ``linear_node = ((c0 * g1 + c1) * g2 + ...)``.
    """
    nid = int(linear_node)
    coords: List[int] = []
    for s in reversed(grid):
        coords.append(nid % s)
        nid //= s
    coords.reverse()
    return tuple(coords)


def mesh_axes_of_grid(grid: Shape) -> tuple[int, ...]:
    """Return the leading mesh-axis sizes of a grid.

    A grid's trailing two dimensions are the Tensix core grid; any leading
    dimensions are device-mesh axes (one virtual device per leading-coordinate
    combination).  Grids of rank <= 2 describe a single device and therefore
    have no mesh axes.
    """
    return tuple(grid[:-2]) if len(grid) > 2 else ()


def node_mesh_coord(linear_node: Index, grid: Shape) -> tuple[int, ...]:
    """Return the device-mesh coordinate a node belongs to.

    The mesh coordinate is the leading portion of the node's full grid
    coordinate, covering only the mesh axes (see :func:`mesh_axes_of_grid`).
    For single-device grids (rank <= 2) this is the empty tuple.
    """
    n_mesh = len(mesh_axes_of_grid(grid))
    if n_mesh == 0:
        return ()
    return node_coord_from_linear(linear_node, grid)[:n_mesh]


def pipe_crosses_mesh(src: NodeCoord, dst: Any, grid: Shape) -> bool:
    """Return True if a pipe from ``src`` to ``dst`` spans more than one device.

    A pipe is "fabric" (cross-device) when its source and any destination differ
    on a mesh axis (the leading ``len(grid) - 2`` grid dims; see
    :func:`mesh_axes_of_grid`).  Otherwise it stays within a single device's core
    grid and lowers to an on-chip NoC transfer.

    ``dst`` may be a single coordinate or a :class:`~sim.typedefs.NodeRange`
    (a tuple that may contain ``slice`` objects for multicast); a slice on a mesh
    axis that covers any value other than the source's coordinate makes the pipe
    fabric.  Grids of rank <= 2 have no mesh axes, so the result is always False.

    Endpoints are accepted in two unambiguous forms and normalized to full grid
    coordinates before comparison:

    * A bare linear index (an ``int`` or a 1-tuple): unflattened via
      :func:`node_coord_from_linear`.  Both node-flattening conventions agree on
      a single linear index, so this is unambiguous.
    * A full grid-rank coordinate: used directly (this is the only form that may
      carry ``slice`` multicast selectors).

    A multi-element coordinate shorter than the grid rank is ambiguous: the
    simulator's :func:`node` flattens the *leading* axes while
    :func:`flatten_node_index` flattens differently, so its entries cannot be
    mapped one-to-one to mesh axes.  Such a coordinate raises rather than guess;
    the trace-only caller (:func:`sim.copyhandlers._pipe_is_fabric`) treats that
    as non-fabric so tracing never crashes an otherwise-valid run.

    Raises:
        ValueError: If ``src`` or ``dst`` is a multi-element coordinate whose
            rank is neither 1 nor ``len(grid)`` (and the grid has mesh axes).
    """
    n_mesh = len(mesh_axes_of_grid(grid))
    if n_mesh == 0:
        return False

    rank = len(grid)

    def _to_full(coord: Any, what: str) -> tuple[Any, ...]:
        if isinstance(coord, int):
            return node_coord_from_linear(coord, grid)
        coord_t = tuple(coord)
        if len(coord_t) == 1 and not isinstance(coord_t[0], slice):
            return node_coord_from_linear(coord_t[0], grid)
        if len(coord_t) == rank:
            return coord_t
        raise ValueError(
            f"pipe {what} {coord_t} is rank {len(coord_t)} but grid {tuple(grid)} "
            f"is rank {rank}; use a full-rank coordinate or a bare linear index "
            f"to classify mesh crossing"
        )

    src_t = _to_full(src, "src")
    dst_t = _to_full(dst, "dst")

    for axis in range(n_mesh):
        src_coord = src_t[axis]
        dst_sel = dst_t[axis]
        match dst_sel:
            case slice():
                start = dst_sel.start if dst_sel.start is not None else 0
                stop = dst_sel.stop if dst_sel.stop is not None else grid[axis]
                step = dst_sel.step if dst_sel.step is not None else 1
                if any(v != src_coord for v in range(start, stop, step)):
                    return True
            case _:
                if dst_sel != src_coord:
                    return True
    return False


def flatten_node_index(node_coord: NodeCoord) -> Index:
    """Flatten a NodeCoord to a linear node index.

    Args:
        node_coord: A NodeCoord which can be a single Index or a tuple of Indices

    Returns:
        A linear Index (single integer)

    Example:
        >>> flatten_node_index(5)  # Already linear
        5
        >>> # With grid (8, 8), node (2, 3) -> 2 * 8 + 3 = 19
        >>> flatten_node_index((2, 3))
        19
    """
    match node_coord:
        case int():
            return node_coord
        case _:
            # Convert to linear index using grid dimensions
            grid = _get_from_frame(
                "grid",
                "grid not available - function must be called within a kernel context",
            )

            coords = list(node_coord)

            # Calculate linear index: for (y, x) with grid (h, w), linear = y * w + x
            # For 3D: (z, y, x) with grid (d, h, w), linear = z * h * w + y * w + x
            linear = coords[0]
            for i in range(1, len(coords)):
                linear = linear * grid[i] + coords[i]

            return int(linear)


def grid_size(dims: Size = 2) -> Union[Size, Shape]:
    """Get the grid size from the execution context.

    Returns the size of the grid in the specified dimensionality.
    - If requested dims < actual grid dims: highest rank dimensions are flattened
    - If requested dims > actual grid dims: lowest rank dimensions are padded with 1s

    Args:
        dims: Number of dimensions to return (must be positive). Defaults to 2.

    Returns:
        Size if dims == 1, otherwise Tuple[Size, ...] of length dims

    Raises:
        ValueError: If dims is not positive
        RuntimeError: If called outside of a kernel function context

    Example:
        # For grid=(8, 8):
        grid_size(dims=1) -> 64 (flattened)
        grid_size(dims=2) -> (8, 8)
        grid_size(dims=3) -> (8, 8, 1) (padded)
    """
    if dims <= 0:
        raise ValueError(f"dims must be positive, got {dims}")

    grid = _get_from_frame(
        "grid", "grid not available - function must be called within a kernel context"
    )

    grid_dims = len(grid)

    if dims == grid_dims:
        result = tuple(grid)
    elif dims < grid_dims:
        if dims == 1:
            flattened = 1
            for d in grid:
                flattened *= d
            result = (flattened,)
        else:
            kept = tuple(grid[: dims - 1])
            flattened = 1
            for i in range(dims - 1, grid_dims):
                flattened *= grid[i]
            result = kept + (flattened,)
    else:  # dims > grid_dims
        padding = (1,) * (dims - grid_dims)
        result = tuple(grid) + padding

    if dims == 1:
        return result[0]
    else:
        return result


def node(dims: Size = 2) -> NodeCoord:
    """Get the current node coordinates from injected context.

    Args:
        dims: Number of dimensions for the node coordinates. Default is 2

    Returns:
        NodeCoord: The node coordinates (int for 1D, tuple for > 1D)

    Raises:
        RuntimeError: If called outside of a Program context
    """
    nid = _get_from_frame(
        "_node", "node not available - function must be called within Program context"
    )

    grid = _get_from_frame(
        "grid", "grid not available - function must be called within a kernel context"
    )

    coords: List[Index] = []

    for s in reversed(grid):
        coords.append(nid % s)
        nid = nid // s
    coords.reverse()

    # If dims < len(grid), flatten the first dimension(s)
    if dims < len(coords):
        flattened = coords[0]
        for i in range(1, len(coords) - dims + 1):
            flattened = flattened * grid[i] + coords[i]
        coords = [flattened] + coords[len(coords) - dims + 1 :]

    # Pad with zeros if dims > len(grid)
    while len(coords) < dims:
        coords.append(0)

    if dims == 1:
        return coords[0]
    else:
        return tuple(coords)
