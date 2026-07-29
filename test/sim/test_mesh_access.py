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


def _shard_dim1(rows: int, cols: int, mesh_n: int) -> ttnn.Tensor:
    """A tile-layout (rows, cols) tensor sharded along dim 1 (the last dim).

    This is the matmul-style partition: each device owns a contiguous band of
    columns rather than of rows.
    """
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(mesh_n))
    return ttnn.from_torch(
        torch.zeros(rows, cols),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )


def _replicated(rows: int, cols: int, mesh_n: int) -> ttnn.Tensor:
    """A tile-layout (rows, cols) tensor replicated across every mesh device."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(mesh_n))
    return ttnn.from_torch(
        torch.zeros(rows, cols),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
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

    def test_two_mesh_axes_on_same_dim_raises(self) -> None:
        """Hierarchical sharding (one tensor dim split by two mesh axes) is rejected.

        With ``dims=(0, 0)`` each device really owns ``rows / (2 * 2)``, which the
        one-axis-per-dim ownership model cannot express: taken per axis it would
        report node 0 as owning rows [0:32) (too many) and node 1 as owning nothing.
        A clear error beats contradictory ownership.
        """
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
        a = ttnn.from_torch(
            torch.zeros(64, 32),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 2), dims=(0, 0)),
        )
        for node in range(4):
            with _node_on_grid(node, (2, 2, 1, 1)):
                with pytest.raises(MeshAccessError, match="Unsupported sharding"):
                    validate_mesh_access(a[0, 0], "read")

    def test_repeated_dim_on_degenerate_axis_allowed(self) -> None:
        """A repeated dim is only hierarchical when both axes actually shard.

        Mesh ``(1, 2)`` with ``dims=(0, 0)`` has a single-device first axis, so dim 0
        is partitioned exactly once and ownership stays well-defined.
        """
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
        a = ttnn.from_torch(
            torch.zeros(64, 32),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(1, 2), dims=(0, 0)),
        )
        with _node_on_grid(0, (1, 2, 1, 1)):
            validate_mesh_access(a[0, 0], "read")  # rows [0:32) -> owned
            with pytest.raises(MeshAccessError, match="Cross-device access"):
                validate_mesh_access(a[1, 0], "read")  # rows [32:64) -> device (0,1)


class TestRowVectorMeshGrid:
    """Ownership on a ``MeshShape(1, n)`` mesh, whose leading axis is degenerate.

    ``ShardTensorToMesh`` keeps that mesh's axis layout, so the tensor's mesh shape
    is ``(1, n)`` and the launch grid must declare both axes: ``(1, n, rows, cols)``.
    The grid's mesh axes are required to equal the mesh shape exactly, so the
    degenerate ``1`` is not optional.
    """

    def _shard(self, rows: int, cols: int, n: int) -> ttnn.Tensor:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, n))
        return ttnn.from_torch(
            torch.zeros(rows, cols),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    def test_each_device_reads_owned_tile(self) -> None:
        """The size-1 leading axis is skipped; axis 1 drives ownership."""
        a = self._shard(128, 32, n=4)  # device (0, j) owns rows [j*32, (j+1)*32)
        for node in range(4):
            with _node_on_grid(node, (1, 4, 1, 1)):
                validate_mesh_access(a[node, 0], "read")

    def test_foreign_tile_raises(self) -> None:
        a = self._shard(128, 32, n=4)
        with _node_on_grid(0, (1, 4, 1, 1)):
            with pytest.raises(MeshAccessError, match="Cross-device access"):
                validate_mesh_access(a[3, 0], "read")

    def test_grid_must_declare_the_degenerate_axis(self) -> None:
        """Dropping the leading 1 from the grid is a hard error, not an equivalence."""
        a = self._shard(128, 32, n=4)
        with _node_on_grid(0, (4, 1, 1)):
            with pytest.raises(MeshAccessError, match="Mesh mismatch"):
                validate_mesh_access(a[0, 0], "read")


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


class TestDeviceScopedOwnership:
    """Ownership is scoped to the *device*, not to the individual node.

    A launch grid usually gives each device a core grid larger than 1x1, so many
    nodes share one device-mesh coordinate and therefore one shard.  On grid
    ``(2, 2, 2)`` the single leading dim is a 2-device mesh axis over a 2x2 core
    grid, so nodes 0-3 all map to device ``(0,)`` and nodes 4-7 to device ``(1,)``.
    Every node of a device may touch that device's whole shard, and no node of it
    may touch another device's shard.
    """

    GRID = (2, 2, 2)
    DEVICE_NODES = {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}

    def test_every_node_of_a_device_reads_that_devices_shard(self) -> None:
        """All four cores of a device may read the tile that device owns."""
        a = _shard_dim0(64, 32, mesh_n=2)  # dev 0 owns rows[0:32), dev 1 [32:64)
        for device, nodes in self.DEVICE_NODES.items():
            for node in nodes:
                with _node_on_grid(node, self.GRID):
                    validate_mesh_access(a[device, 0], "read")

    def test_no_node_of_a_device_reads_a_foreign_shard(self) -> None:
        """No core of a device may read the tile owned by the other device."""
        a = _shard_dim0(64, 32, mesh_n=2)
        for device, nodes in self.DEVICE_NODES.items():
            foreign = 1 - device
            for node in nodes:
                with _node_on_grid(node, self.GRID):
                    with pytest.raises(MeshAccessError, match="Cross-device access"):
                        validate_mesh_access(a[foreign, 0], "read")


class TestMultiTilePerDeviceShard:
    """A device owning several tiles may touch all of them, but not the next one.

    With 128 rows over 2 devices each device owns 64 rows = 2 tiles: device 0 owns
    tiles 0-1 (rows [0:64)) and device 1 tiles 2-3 (rows [64:128)).  This checks the
    owned-slice arithmetic across a tile boundary rather than only at a single tile.
    """

    def test_device_reads_each_of_its_local_tiles(self) -> None:
        """A kernel looping over its device's local tiles is permitted throughout."""
        a = _shard_dim0(128, 32, mesh_n=2)
        for node, local_tiles in [(0, (0, 1)), (1, (2, 3))]:
            with _node_on_grid(node, (2, 1, 1)):
                for tile in local_tiles:
                    validate_mesh_access(a[tile, 0], "read")

    def test_multi_tile_slice_within_shard_allowed(self) -> None:
        """A single access spanning all of a device's tiles is permitted."""
        a = _shard_dim0(128, 32, mesh_n=2)
        with _node_on_grid(0, (2, 1, 1)):
            validate_mesh_access(a[0:2, 0], "read")  # rows [0:64) == dev 0's shard

    def test_first_tile_past_the_shard_raises(self) -> None:
        """The tile just past a device's last owned tile belongs to its neighbour."""
        a = _shard_dim0(128, 32, mesh_n=2)
        with _node_on_grid(0, (2, 1, 1)):
            with pytest.raises(MeshAccessError, match=r"\[64:96\) along dim 0"):
                validate_mesh_access(a[2, 0], "read")  # rows [64:96) -> dev 1

    def test_slice_straddling_the_shard_boundary_raises(self) -> None:
        """An access crossing the ownership boundary is rejected."""
        a = _shard_dim0(128, 32, mesh_n=2)
        with _node_on_grid(0, (2, 1, 1)):
            with pytest.raises(MeshAccessError, match="Cross-device access"):
                validate_mesh_access(a[1:3, 0], "read")  # rows [32:96) straddles 64


class TestLastDimShard:
    """Ownership on a last-dim (column) shard, the matmul-style partition.

    With 128 columns over 2 devices, device 0 owns cols [0:64) (tile columns 0-1)
    and device 1 owns cols [64:128) (tile columns 2-3).  Validation must report the
    sharded dim (1) rather than assuming rows.
    """

    def test_device_reads_its_own_column_band(self) -> None:
        """Each device may read the tile columns within its own band."""
        a = _shard_dim1(32, 128, mesh_n=2)
        for node, local_cols in [(0, (0, 1)), (1, (2, 3))]:
            with _node_on_grid(node, (2, 1, 1)):
                for col in local_cols:
                    validate_mesh_access(a[0, col], "read")

    def test_foreign_column_read_raises_on_dim1(self) -> None:
        """A device reading another device's columns is rejected, naming dim 1."""
        a = _shard_dim1(32, 128, mesh_n=2)
        with _node_on_grid(1, (2, 1, 1)):  # dev 1 owns cols [64:128)
            with pytest.raises(MeshAccessError, match="along dim 1"):
                validate_mesh_access(a[0, 0], "read")  # cols [0:32) -> dev 0

    def test_row_position_is_unrestricted(self) -> None:
        """Only the sharded dim constrains access: any row is fine."""
        a = _shard_dim1(64, 128, mesh_n=2)
        with _node_on_grid(1, (2, 1, 1)):
            validate_mesh_access(a[0, 2], "read")
            validate_mesh_access(a[1, 3], "read")  # different row, still owned cols


class TestReplicatedAlongsideSharded:
    """Data-parallel mixing: replicated weights plus a per-device sharded input.

    ``ReplicateTensorToMesh`` attaches no ``MeshShardInfo``, so every device owns a
    replicated tensor in full and may read all of it, while a sharded tensor in the
    same kernel stays restricted to each device's own shard.
    """

    def test_replicated_tensor_fully_readable_by_every_device(self) -> None:
        """Any device may read the whole replicated tensor, including foreign tiles."""
        w = _replicated(64, 32, mesh_n=2)
        assert w.mesh_shard_info is None
        for node in (0, 1):
            with _node_on_grid(node, (2, 1, 1)):
                validate_mesh_access(w, "read")  # full tensor
                validate_mesh_access(w[0, 0], "read")
                validate_mesh_access(w[1, 0], "read")

    def test_sharded_input_still_restricted_alongside_replicated(self) -> None:
        """The replicated tensor's permissiveness does not relax the sharded one."""
        w = _replicated(64, 32, mesh_n=2)
        a = _shard_dim0(64, 32, mesh_n=2)
        with _node_on_grid(0, (2, 1, 1)):
            validate_mesh_access(w[1, 0], "read")  # replicated: allowed
            with pytest.raises(MeshAccessError, match="Cross-device access"):
                validate_mesh_access(a[1, 0], "read")  # sharded: dev 1's shard


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


class TestDeviceScopedOwnershipIntegration:
    """End-to-end: several cores per device all reading that device's shard.

    Grid ``(2, 2, 2)`` is a 2-device mesh axis over a 2x2 core grid, so four nodes
    share each device's shard.  A kernel where every node reads its own device's
    tile runs cleanly; one where every node reads tile 0 trips validation on the
    four nodes of device 1.
    """

    def _build_io(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        a = _shard_dim0(64, 32, mesh_n=2)
        a._name = "a"  # type: ignore[attr-defined]
        o = ttnn.empty((64, 32))
        return a, o

    def test_all_cores_of_a_device_read_its_shard(self) -> None:
        """Every core reading its own device's tile completes cleanly."""
        a, o = self._build_io()

        @ttl.operation(grid=(2, 2, 2))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                    out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read() -> None:
                device = ttl.node(dims=3)[0]  # leading dim is the mesh axis
                with dfb.reserve() as blk:
                    tx = ttl.copy(a[device, 0], blk)
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                # All four cores of a device write that device's tile; they carry
                # identical data, so the result is order-independent.
                device = ttl.node(dims=3)[0]
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[device, 0])
                    tx.wait()

        kernel(a, o)

    def test_all_cores_reading_tile0_raises_on_foreign_device(self) -> None:
        """Every node reading tile 0 trips validation on device 1's cores."""
        a, o = self._build_io()

        @ttl.operation(grid=(2, 2, 2))
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
                    tx = ttl.copy(a[0, 0], blk)  # every core reads device 0's tile
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                device = ttl.node(dims=3)[0]
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[device, 0])
                    tx.wait()

        with pytest.raises(RuntimeError, match="Cross-device access"):
            kernel(a, o)


class TestMultiTilePerDeviceIntegration:
    """End-to-end: a kernel looping over the several tiles its device owns.

    With 128 rows over 2 devices each device owns 2 tiles, so a realistic kernel
    iterates over its local tiles rather than handling exactly one.  Every
    iteration stays inside the device's shard, so the run completes cleanly.
    """

    TILES_PER_DEVICE = 2

    def test_kernel_loops_over_its_local_tiles(self) -> None:
        tiles = self.TILES_PER_DEVICE
        a = _shard_dim0(128, 32, mesh_n=2)
        a._name = "a"  # type: ignore[attr-defined]
        o = ttnn.empty((128, 32))

        @ttl.operation(grid=(2, 1, 1))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                for _ in range(tiles):
                    with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                        out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read() -> None:
                device = ttl.node(dims=3)[0]
                for local in range(tiles):
                    with dfb.reserve() as blk:
                        tx = ttl.copy(a[device * tiles + local, 0], blk)
                        tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                device = ttl.node(dims=3)[0]
                for local in range(tiles):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o[device * tiles + local, 0])
                        tx.wait()

        kernel(a, o)


class TestReplicatedAlongsideShardedIntegration:
    """End-to-end data-parallel kernel: replicated weights + sharded activations.

    Each device reads the whole replicated tensor (legal, no ``MeshShardInfo``) and
    only its own shard of the sharded input, mirroring a data-parallel layer.
    """

    def test_replicated_and_sharded_reads_run(self) -> None:
        a = _shard_dim0(64, 32, mesh_n=2)
        a._name = "a"  # type: ignore[attr-defined]
        w = _replicated(64, 32, mesh_n=2)
        w._name = "w"  # type: ignore[attr-defined]
        o = ttnn.empty((64, 32))

        @ttl.operation(grid=(2, 1, 1))
        def kernel(a: ttnn.Tensor, w: ttnn.Tensor, o: ttnn.Tensor) -> None:
            a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute() -> None:
                with a_dfb.wait() as a_blk, w_dfb.wait() as w_blk:
                    with out_dfb.reserve() as out_blk:
                        out_blk.store(a_blk + w_blk)

            @ttl.datamovement()
            def dm_read() -> None:
                r = ttl.node(dims=3)[0]
                with a_dfb.reserve() as a_blk:
                    tx = ttl.copy(a[r, 0], a_blk)  # own shard only
                    tx.wait()
                with w_dfb.reserve() as w_blk:
                    # Replicated: every device may read any tile, including the
                    # one another device's shard would cover.
                    tx = ttl.copy(w[1, 0], w_blk)
                    tx.wait()

            @ttl.datamovement()
            def dm_write() -> None:
                r = ttl.node(dims=3)[0]
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[r, 0])
                    tx.wait()

        kernel(a, w, o)


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
