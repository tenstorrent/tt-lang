# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-device (mesh) memory-access validation.

A tensor created with a sharding mesh mapper is split across virtual devices.
When the launch grid has mesh axes (its leading dimensions), each node maps to a
device-mesh coordinate and may touch only the element slice that coordinate
owns.  Accessing another device's slice raises ``MeshAccessError``.  Validation
is skipped (the tensor is treated as fully owned) only for unsharded tensors and
for single-device / SPMD grids (no mesh axes).  Every other ill-defined ownership
situation -- accessing a mesh-sharded tensor outside a kernel, a grid/tensor mesh
mismatch, or an unevenly divisible shard dim -- is a hard ``MeshAccessError``.
"""

import contextlib
from typing import Iterator

import pytest
import torch
from greenlet import getcurrent

from sim import ttl, ttnn
from sim.context import get_context
from sim.greenlet_scheduler import GreenletScheduler
from sim.mesh_access import MeshAccessError, validate_mesh_access


@contextlib.contextmanager
def _node_on_grid(node: int, grid: tuple[int, ...]) -> Iterator[None]:
    """Simulate running as ``node`` on a program launched over ``grid``."""
    ctx = get_context()
    scheduler = GreenletScheduler()
    scheduler.grid = grid
    ctx.scheduler = scheduler
    cur = getcurrent()
    had_attr = hasattr(cur, "_sim_node")
    prev = getattr(cur, "_sim_node", None)
    cur._sim_node = node  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if had_attr:
            cur._sim_node = prev  # type: ignore[attr-defined]
        else:
            delattr(cur, "_sim_node")
        ctx.scheduler = None


def _shard_dim0(rows: int, cols: int, mesh_n: int) -> ttnn.Tensor:
    """A tile-layout tensor of shape (rows, cols) sharded along dim 0 over mesh_n devices."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(mesh_n))
    return ttnn.from_torch(
        torch.zeros(rows, cols),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _shard_2d(rows: int, cols: int) -> ttnn.Tensor:
    """A tile-layout (rows, cols) tensor sharded over a 2x2 device mesh on dims (0, 1).

    Device ``(i, j)`` owns the block ``rows[i*rows/2 : ...]`` x ``cols[j*cols/2 : ...]``.
    """
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    return ttnn.from_torch(
        torch.zeros(rows, cols),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 2), dims=(0, 1)),
    )


class TestValidateMeshAccessUnit:
    """Direct unit tests of validate_mesh_access against sliced tensor views."""

    def test_owned_tile_read_allowed(self) -> None:
        """Each device reading its own tile is permitted."""
        a = _shard_dim0(64, 32, mesh_n=2)  # device 0 owns rows[0:32), device 1 [32:64)
        with _node_on_grid(0, (2, 1, 1)):
            validate_mesh_access(a[0, 0], "read")  # rows [0:32) -> owned by dev 0
        with _node_on_grid(1, (2, 1, 1)):
            validate_mesh_access(a[1, 0], "read")  # rows [32:64) -> owned by dev 1

    def test_foreign_tile_read_raises(self) -> None:
        """Reading a tile owned by another device is a hard error."""
        a = _shard_dim0(64, 32, mesh_n=2)
        with _node_on_grid(1, (2, 1, 1)):
            with pytest.raises(MeshAccessError, match="Cross-device access"):
                validate_mesh_access(a[0, 0], "read")  # rows [0:32) owned by dev 0

    def test_foreign_tile_write_raises(self) -> None:
        """Writing a tile owned by another device is a hard error."""
        a = _shard_dim0(64, 32, mesh_n=2)
        with _node_on_grid(0, (2, 1, 1)):
            with pytest.raises(MeshAccessError, match="along dim 0"):
                validate_mesh_access(a[1, 0], "write")  # rows [32:64) owned by dev 1

    def test_full_tensor_access_raises(self) -> None:
        """Accessing the whole sharded tensor touches other devices' shards."""
        a = _shard_dim0(64, 32, mesh_n=2)
        with _node_on_grid(0, (2, 1, 1)):
            with pytest.raises(MeshAccessError):
                validate_mesh_access(a, "read")  # rows [0:64) but dev 0 owns [0:32)

    def test_unsharded_tensor_skipped(self) -> None:
        """A tensor with no mesh sharding metadata is never restricted."""
        a = ttnn.from_torch(torch.zeros(64, 32))
        with _node_on_grid(1, (2, 1, 1)):
            validate_mesh_access(a[0, 0], "read")  # no MeshShardInfo -> allowed

    def test_outside_kernel_raises(self) -> None:
        """Accessing a mesh-sharded tensor outside a kernel is a hard error."""
        a = _shard_dim0(64, 32, mesh_n=2)
        # No _node_on_grid context: getcurrent() has no _sim_node tag.
        ctx = get_context()
        ctx.scheduler = None
        with pytest.raises(MeshAccessError, match="outside any scheduled kernel"):
            validate_mesh_access(a[0, 0], "read")

    def test_spmd_grid_skipped(self) -> None:
        """A rank-2 grid has no mesh axes (SPMD / single device): no enforcement."""
        a = _shard_dim0(64, 32, mesh_n=2)
        with _node_on_grid(1, (1, 1)):  # no leading mesh dims
            validate_mesh_access(a, "read")

    def test_axis_count_mismatch_raises(self) -> None:
        """A 2-axis mesh tensor on a 1-mesh-axis grid cannot be mapped: hard error."""
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
        a = ttnn.from_torch(
            torch.zeros(64, 64),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 2), dims=(0, 1)),
        )
        # Grid (2, 1, 1) has a single mesh axis; tensor mesh has two -> mismatch.
        with _node_on_grid(1, (2, 1, 1)):
            with pytest.raises(MeshAccessError, match="Mesh mismatch"):
                validate_mesh_access(a, "read")

    def test_axis_size_mismatch_raises(self) -> None:
        """A grid whose mesh-axis size differs from the tensor's mesh shape: hard error."""
        a = _shard_dim0(64, 32, mesh_n=4)  # tensor mesh shape (4,)
        # Grid (2, 1, 1) has mesh axes (2,) != (4,) -> cannot map cleanly.
        with _node_on_grid(1, (2, 1, 1)):
            with pytest.raises(MeshAccessError, match="Mesh mismatch"):
                validate_mesh_access(a, "read")

    def test_indivisible_shard_dim_raises(self) -> None:
        """An unevenly divisible shard dim has undefined ownership: hard error."""
        # 64 rows sharded 3 ways -> 64 % 3 != 0, ownership boundaries undefined.
        a = _shard_dim0(64, 32, mesh_n=3)
        with _node_on_grid(1, (3, 1, 1)):
            with pytest.raises(MeshAccessError, match="Uneven mesh shard"):
                validate_mesh_access(a, "read")


class TestTwoAxisMesh:
    """Validation across a 2-axis device mesh (the 4D launch-grid layout).

    A rank-4 grid ``(m0, m1, r, c)`` has mesh axes ``(m0, m1)`` (its leading dims)
    over a ``r x c`` Tensix core grid.  On grid ``(2, 2, 1, 1)`` the four nodes map
    to mesh coordinates ``0->(0,0) 1->(0,1) 2->(1,0) 3->(1,1)`` and device ``(i, j)``
    owns tile ``(i, j)`` of a ``(64, 64)`` tensor sharded on both dims.
    """

    def test_each_device_reads_owned_tile(self) -> None:
        """Device (i, j) reading its own tile (i, j) is permitted on a 4D grid."""
        a = _shard_2d(64, 64)
        for node, (i, j) in [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))]:
            with _node_on_grid(node, (2, 2, 1, 1)):
                validate_mesh_access(a[i, j], "read")

    def test_foreign_tile_on_first_axis_raises(self) -> None:
        """Crossing the first mesh axis (dim 0) is a hard error."""
        a = _shard_2d(64, 64)
        with _node_on_grid(1, (2, 2, 1, 1)):  # mesh (0, 1) owns rows [0:32)
            with pytest.raises(MeshAccessError, match="along dim 0"):
                validate_mesh_access(a[1, 1], "read")  # rows [32:64)

    def test_foreign_tile_on_second_axis_raises(self) -> None:
        """Crossing the second mesh axis (dim 1) is a hard error."""
        a = _shard_2d(64, 64)
        with _node_on_grid(0, (2, 2, 1, 1)):  # mesh (0, 0) owns cols [0:32)
            with pytest.raises(MeshAccessError, match="along dim 1"):
                validate_mesh_access(a[0, 1], "read")  # cols [32:64)


class TestMeshAccessIntegration:
    """End-to-end validation through a running mesh-sharded kernel."""

    def _build_io(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        a = _shard_dim0(64, 32, mesh_n=2)
        a._name = "a"  # type: ignore[attr-defined]
        o = ttnn.empty((64, 32))
        return a, o

    def test_legal_per_device_access_runs(self) -> None:
        """A kernel where each node reads only its owned tile completes cleanly."""
        a, o = self._build_io()

        @ttl.operation(grid=(2, 1, 1))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                    out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read() -> None:
                r = ttl.node(dims=3)[0]
                with dfb.reserve() as blk:
                    tx = ttl.copy(a[r, 0], blk)
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                r = ttl.node(dims=3)[0]
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[r, 0])
                    tx.wait()

        kernel(a, o)

    def test_illegal_cross_device_access_raises(self) -> None:
        """A kernel where every node reads tile 0 trips validation on node 1."""
        a, o = self._build_io()

        @ttl.operation(grid=(2, 1, 1))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                    out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read() -> None:
                with dfb.reserve() as blk:
                    tx = ttl.copy(a[0, 0], blk)  # every node reads tile 0
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                r = ttl.node(dims=3)[0]
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[r, 0])
                    tx.wait()

        with pytest.raises(RuntimeError, match="Cross-device access"):
            kernel(a, o)


class TestTwoAxisMeshIntegration:
    """End-to-end validation through a kernel launched on a 4D (2-mesh-axis) grid."""

    def _build_io(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        a = _shard_2d(64, 64)
        a._name = "a"  # type: ignore[attr-defined]
        o = ttnn.empty((64, 64))
        return a, o

    def test_legal_per_device_access_runs(self) -> None:
        """Each device reading and writing only its own (i, j) tile completes cleanly."""
        a, o = self._build_io()

        @ttl.operation(grid=(2, 2, 1, 1))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                    out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read() -> None:
                r = ttl.node(dims=4)
                with dfb.reserve() as blk:
                    tx = ttl.copy(a[r[0], r[1]], blk)  # tile owned by this device
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                r = ttl.node(dims=4)
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[r[0], r[1]])
                    tx.wait()

        kernel(a, o)

    def test_illegal_cross_device_access_raises(self) -> None:
        """Every node reading tile (0, 0) trips validation on the off-mesh devices."""
        a, o = self._build_io()

        @ttl.operation(grid=(2, 2, 1, 1))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                    out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read() -> None:
                with dfb.reserve() as blk:
                    tx = ttl.copy(a[0, 0], blk)  # every device reads tile (0, 0)
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                r = ttl.node(dims=4)
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[r[0], r[1]])
                    tx.wait()

        with pytest.raises(RuntimeError, match="Cross-device access"):
            kernel(a, o)
