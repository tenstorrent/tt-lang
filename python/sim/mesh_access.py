# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-device (mesh) memory-access validation.

On a multidevice mesh, a tensor created with a sharding mesh mapper is split
across virtual devices: each device physically holds only its own shard.  The
functional simulator stores the full tensor in one buffer, so a kernel that
reads bytes physically owned by a *different* device "just works" in the
simulator but would fail on hardware (separate per-device memories).

This module detects those illegal cross-device accesses.  For a tensor carrying
:class:`~sim.ttnnsim.MeshShardInfo`, when the launch grid has mesh axes (the
leading grid dimensions; see :func:`~sim.nodecontext.mesh_axes_of_grid`), each
node is mapped to a device-mesh coordinate and may only touch the element slice
that coordinate owns.  Touching any element outside the owned slice raises
:class:`MeshAccessError`.

Validation is *skipped* (a mesh-sharded tensor is treated as fully owned) only in
the two cases where every device legitimately owns the whole tensor:

* The tensor has no mesh sharding metadata (replicated / unsharded tensors are
  owned in full by every device).
* The launch grid has no mesh axes (single device or SPMD multi-chip, where
  every chip runs the identical program over the full tensor).

Every other ill-defined ownership situation is a hard :class:`MeshAccessError`
rather than a silent pass, because each one indicates a program that cannot run
correctly on a real mesh:

* The access happens outside a scheduled kernel: a mesh-sharded tensor has no
  device to attribute the access to, so it may only be touched from within a
  kernel.
* The grid's mesh axes do not match the tensor's mesh shape (different axis
  count or axis sizes): a tensor must be launched on a grid whose leading (mesh)
  dimensions equal its mesh shape so each node maps to exactly one shard.
* A sharded dimension is not evenly divisible by its mesh-axis device count:
  ownership boundaries are undefined unless every device owns an equal slice.
"""

from __future__ import annotations

from typing import Optional

from greenlet import getcurrent

from .context import get_context
from .nodecontext import mesh_axes_of_grid, node_mesh_coord
from .ttnnsim import Tensor


class MeshAccessError(RuntimeError):
    """Raised when a node accesses tensor data owned by a different mesh device."""


def validate_mesh_access(tensor: Tensor, direction: str) -> None:
    """Verify ``tensor`` access stays within the current node's owned mesh shard.

    Args:
        tensor: The tensor (or sliced view) being read from or written to.
        direction: ``"read"`` or ``"write"`` -- used only for the error message.

    Raises:
        MeshAccessError: If the tensor is mesh-sharded but is accessed outside a
            kernel, if the launch grid's mesh axes do not match the tensor's mesh
            shape, if a sharded dimension is not evenly divisible by its
            mesh-axis device count, or if the accessed element range exceeds, on
            any mesh-sharded dimension, the slice owned by the current node's
            device-mesh coordinate.
    """
    msi = getattr(tensor, "mesh_shard_info", None)
    if msi is None:
        return

    name = getattr(tensor, "_name", None) or "tensor"

    node: Optional[int] = getattr(getcurrent(), "_sim_node", None)
    if node is None:
        raise MeshAccessError(
            f"Cross-device access: mesh-sharded tensor '{name}' was {direction} "
            f"outside any scheduled kernel, so the access cannot be attributed to "
            f"a device. Mesh-sharded tensors may only be accessed from within a "
            f"kernel, where the node maps to a device-mesh coordinate."
        )

    scheduler = get_context().scheduler
    grid = tuple(getattr(scheduler, "grid", ()) or ()) if scheduler is not None else ()
    mesh_axes = mesh_axes_of_grid(grid)
    if not mesh_axes:
        return

    if mesh_axes != tuple(msi.mesh_shape):
        raise MeshAccessError(
            f"Mesh mismatch: the launch grid's mesh axes {list(mesh_axes)} do not "
            f"match tensor '{name}' mesh shape {list(msi.mesh_shape)}. A "
            f"mesh-sharded tensor must be launched on a grid whose leading (mesh) "
            f"dimensions equal its mesh shape so each node maps to exactly one "
            f"device shard."
        )

    mesh_coord = node_mesh_coord(node, grid)

    root_shape = getattr(tensor, "_root_shape", None) or tensor.shape
    origin = getattr(tensor, "_element_origin", None) or (0,) * len(root_shape)
    shape = tensor.shape

    for axis, dim in enumerate(msi.dims):
        if dim is None:
            continue
        n = msi.mesh_shape[axis]
        if n <= 1:
            continue
        full = root_shape[dim]
        if full % n != 0:
            raise MeshAccessError(
                f"Uneven mesh shard: tensor '{name}' dim {dim} has size {full}, "
                f"which is not evenly divisible by its {n}-way shard on mesh axis "
                f"{axis}. Each device must own an equal contiguous slice; pad the "
                f"dimension to a multiple of {n} so ownership boundaries are "
                f"well-defined."
            )
        shard = full // n
        coord = mesh_coord[axis]
        owned_start = coord * shard
        owned_stop = (coord + 1) * shard

        access_start = origin[dim]
        access_stop = origin[dim] + shape[dim]
        if access_start < owned_start or access_stop > owned_stop:
            raise MeshAccessError(
                f"Cross-device access: node{node} (device {list(mesh_coord)}) "
                f"would {direction} '{name}' elements "
                f"[{access_start}:{access_stop}) along dim {dim}, but this device "
                f"owns only [{owned_start}:{owned_stop}) on mesh axis {axis} "
                f"({n}-way shard of dim size {full}). On hardware that data lives "
                f"in another device's memory; each device may access only its own "
                f"shard."
            )
