# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`sim.sharding` (shard locality from :class:`~sim.ttnnsim.MemoryConfig`).

``ShardSpec.shard_shape`` / ``NdShardSpec.shard_shape`` use **element** units (tt-metal style).
"""

import pytest
import torch

from sim import ttnn
from sim.constants import TILE_SHAPE
from sim.sharding import (
    _linear_to_nd_pos,
    count_local_remote_l1_dram,
    count_local_remote_l1_dram_for_getitem,
    shard_origin_from_key,
)
from sim.ttnnsim import (
    MemoryConfig,
    NdShardSpec,
    ShardDistributionStrategy,
    ShardSpec,
    ShardingStrategy,
)


class TestCountLocalRemoteL1Dram:
    """Tests for :func:`~sim.sharding.count_local_remote_l1_dram`."""

    # ---- Unsharded / interleaved ----

    def test_no_memory_config_all_dram(self) -> None:
        """Interleaved path counts total elements as DRAM."""
        t = ttnn.Tensor(torch.zeros(64, 64))
        local, remote, dram = count_local_remote_l1_dram(t, 0)
        assert local == 0
        assert remote == 0
        assert dram == 4096

    def test_interleaved_memory_config_all_dram(self) -> None:
        mc = MemoryConfig(strategy=ShardingStrategy.INTERLEAVED)
        t = ttnn.Tensor(torch.zeros(64, 64), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram(t, 0)
        assert dram == 4096

    # ---- HEIGHT_SHARDED (shard_shape in elements) ----

    def test_height_sharded_local_access(self) -> None:
        """Core 0 reading its own shard: all elements are local."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(32, 128))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            full, (slice(0, 1), slice(0, 4)), 0
        )
        assert local == 32 * 128
        assert remote == 0
        assert dram == 0

    def test_height_sharded_remote_access(self) -> None:
        """Core 0 reading core 1's shard: all elements are remote."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(32, 128))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            full, (slice(1, 2), slice(0, 4)), 0
        )
        assert local == 0
        assert remote == 32 * 128
        assert dram == 0

    def test_height_sharded_full_tensor_core0(self) -> None:
        """Core 0 on full tensor: only its row band is local."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(32, 64))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(128, 64), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram(t, 0)
        assert local == 32 * 64
        assert remote == 96 * 64
        assert dram == 0

    def test_height_sharded_each_core_local(self) -> None:
        """Each core reading its own height shard reports all-local."""
        spec = ShardSpec(shard_grid=(4,), shard_shape=(32, 64))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(128, 64), memory_config=mc)
        for core in range(4):
            local, remote, dram = count_local_remote_l1_dram_for_getitem(
                full, (slice(core, core + 1), slice(0, 2)), core
            )
            assert local == 32 * 64, f"core {core}"
            assert remote == 0
            assert dram == 0

    # ---- WIDTH_SHARDED ----

    def test_width_sharded_local_access(self) -> None:
        spec = ShardSpec(shard_grid=(4,), shard_shape=(64, 32))
        mc = MemoryConfig(strategy=ShardingStrategy.WIDTH_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(64, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            full, (slice(0, 2), slice(0, 1)), 0
        )
        assert local == 64 * 32
        assert remote == 0
        assert dram == 0

    def test_width_sharded_remote_access(self) -> None:
        spec = ShardSpec(shard_grid=(4,), shard_shape=(64, 32))
        mc = MemoryConfig(strategy=ShardingStrategy.WIDTH_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(64, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            full, (slice(0, 2), slice(2, 3)), 0
        )
        assert local == 0
        assert remote == 64 * 32
        assert dram == 0

    # ---- BLOCK_SHARDED ----

    def test_block_sharded_local_access(self) -> None:
        spec = ShardSpec(shard_grid=(2, 2), shard_shape=(64, 64))
        mc = MemoryConfig(strategy=ShardingStrategy.BLOCK_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            full, (slice(2, 4), slice(2, 4)), 3
        )
        assert local == 64 * 64
        assert remote == 0
        assert dram == 0

    def test_block_sharded_remote_access(self) -> None:
        spec = ShardSpec(shard_grid=(2, 2), shard_shape=(64, 64))
        mc = MemoryConfig(strategy=ShardingStrategy.BLOCK_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            full, (slice(2, 4), slice(2, 4)), 0
        )
        assert local == 0
        assert remote == 64 * 64
        assert dram == 0

    def test_block_sharded_all_cores_local(self) -> None:
        spec = ShardSpec(shard_grid=(2, 2), shard_shape=(32, 64))
        mc = MemoryConfig(strategy=ShardingStrategy.BLOCK_SHARDED, shard_spec=spec)
        full = ttnn.Tensor(torch.zeros(64, 128), memory_config=mc)
        for r in range(2):
            for c in range(2):
                core = r * 2 + c
                local, remote, dram = count_local_remote_l1_dram_for_getitem(
                    full, (slice(r, r + 1), slice(c * 2, c * 2 + 2)), core
                )
                assert local == 32 * 64, f"core ({r},{c})"
                assert remote == 0
                assert dram == 0


class TestNdSharding:
    """Tests for ND_SHARDED counting (NdShardSpec / ShardDistributionStrategy)."""

    def test_nd_shard_origin_from_key_full_rank(self) -> None:
        spec = NdShardSpec(
            shard_shape=(64, 64),
            shard_grid=(2, 4),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(128, 256), memory_config=mc)
        assert shard_origin_from_key(t, (slice(2, 4), slice(4, 6))) == (64, 128)

    def test_nd_missing_spec_raises(self) -> None:
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED)
        t = ttnn.Tensor(torch.zeros(64, 64), memory_config=mc)
        with pytest.raises(ValueError, match="nd_shard_spec"):
            count_local_remote_l1_dram(t, 0)

    def test_grid2d_equivalent_to_block_sharded(self) -> None:
        """GRID_2D on a 2-D tensor matches BLOCK_SHARDED ownership (element shards)."""
        nd_spec = NdShardSpec(
            shard_shape=(64, 64),
            shard_grid=(2, 4),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        nd_mc = MemoryConfig(
            strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=nd_spec
        )
        block_spec = ShardSpec(shard_grid=(2, 4), shard_shape=(64, 64))
        block_mc = MemoryConfig(
            strategy=ShardingStrategy.BLOCK_SHARDED, shard_spec=block_spec
        )
        t_nd = ttnn.Tensor(torch.zeros(128, 256), memory_config=nd_mc)
        t_block = ttnn.Tensor(torch.zeros(128, 256), memory_config=block_mc)
        for core in range(8):
            core_row = core // 4
            core_col = core % 4
            r0, c0 = core_row * 2, core_col * 2
            key = (slice(r0, r0 + 2), slice(c0, c0 + 2))
            shard_nd = t_nd[key]
            shard_block = t_block[key]
            assert count_local_remote_l1_dram(
                shard_nd,
                core,
                origin_in_parent_elements=shard_origin_from_key(t_nd, key),
            ) == count_local_remote_l1_dram(
                shard_block,
                core,
                origin_in_parent_elements=shard_origin_from_key(t_block, key),
            ), f"core {core} mismatch"

    def test_grid2d_3d_tensor_all_local(self) -> None:
        """GRID_2D 3D: each core's shard box matches one full element/tile key slice."""
        shard_grid = (2, 2, 2)
        shard_shape = (32, 64, 64)
        spec = NdShardSpec(
            shard_shape=shard_shape,
            shard_grid=shard_grid,
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(64, 128, 128), memory_config=mc)
        tr = shard_shape[1] // TILE_SHAPE[0]
        tc = shard_shape[2] // TILE_SHAPE[1]
        vol = shard_shape[0] * shard_shape[1] * shard_shape[2]
        for core in range(8):
            pos = _linear_to_nd_pos(core, shard_grid)
            key = (
                slice(pos[0] * shard_shape[0], (pos[0] + 1) * shard_shape[0]),
                slice(pos[1] * tr, (pos[1] + 1) * tr),
                slice(pos[2] * tc, (pos[2] + 1) * tc),
            )
            local, remote, dram = count_local_remote_l1_dram_for_getitem(t, key, core)
            assert local == vol, f"core {core}"
            assert remote == 0
            assert dram == 0

    def test_grid2d_remote_access(self) -> None:
        spec = NdShardSpec(
            shard_shape=(64, 64),
            shard_grid=(2, 2),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        local, remote, dram = count_local_remote_l1_dram_for_getitem(
            t, (slice(2, 4), slice(0, 2)), 0
        )
        assert local == 0
        assert remote == 64 * 64
        assert dram == 0

    def test_round_robin_single_tile_shards(self) -> None:
        """ROUND_ROBIN_1D: one element-shard per 32x32 tile cell; assignment RR."""
        spec = NdShardSpec(
            shard_grid=(2, 2),
            shard_shape=(32, 32),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(64, 64), memory_config=mc)
        one = 32 * 32
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(0, 1), slice(0, 1)), 0
        ) == (
            one,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(0, 1), slice(0, 1)), 1
        ) == (
            0,
            one,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(0, 1), slice(1, 2)), 1
        ) == (
            one,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(0, 1), slice(1, 2)), 0
        ) == (
            0,
            one,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(1, 2), slice(0, 1)), 2
        ) == (
            one,
            0,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(1, 2), slice(0, 1)), 0
        ) == (
            0,
            one,
            0,
        )
        assert count_local_remote_l1_dram_for_getitem(
            t, (slice(1, 2), slice(1, 2)), 3
        ) == (
            one,
            0,
            0,
        )

    def test_round_robin_multi_shard_per_core(self) -> None:
        spec = NdShardSpec(
            shard_shape=(32, 32),
            shard_grid=(1, 2),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(32, 64), memory_config=mc)
        local0, remote0, _ = count_local_remote_l1_dram_for_getitem(
            t, (slice(0, 1), slice(0, 2)), 0
        )
        local1, remote1, _ = count_local_remote_l1_dram_for_getitem(
            t, (slice(0, 1), slice(0, 2)), 1
        )
        assert local0 == 1024
        assert local1 == 1024

    def test_round_robin_3d_tensor_batch_sharded(self) -> None:
        spec = NdShardSpec(
            shard_grid=(2, 1, 1),
            shard_shape=(1, 32, 32),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(2, 32, 32), memory_config=mc)
        one = 32 * 32
        for core in range(2):
            local, remote, dram = count_local_remote_l1_dram_for_getitem(
                t, (slice(core, core + 1), slice(0, 1), slice(0, 1)), core
            )
            assert local == one, f"core {core}"
            assert remote == 0
            assert dram == 0


class TestNdShardingTechReportExamples:
    """ND scenarios aligned with element ``shard_shape`` (see tt-metal tech report)."""

    def test_example1_simple_3d_sharding_round_robin(self) -> None:
        spec = NdShardSpec(
            shard_shape=(1, 32, 32),
            shard_grid=(2, 2, 2),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(2, 64, 64), memory_config=mc)

        def _linear_shard(i: int, j: int, k: int) -> int:
            return i * 4 + j * 2 + k

        vol = 1 * 32 * 32
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    key = (slice(i, i + 1), slice(j, j + 1), slice(k, k + 1))
                    owner = _linear_shard(i, j, k)
                    for core in range(8):
                        loc, rem, dram = count_local_remote_l1_dram_for_getitem(
                            t, key, core
                        )
                        if core == owner:
                            assert loc == vol and rem == 0 and dram == 0
                        else:
                            assert loc == 0 and rem == vol and dram == 0

    def test_example3_single_dimension_sharded_grid_2d(self) -> None:
        spec = NdShardSpec(
            shard_shape=(3, 32, 32),
            shard_grid=(1, 3, 1),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(3, 96, 32), memory_config=mc)
        vol = 3 * 32 * 32
        for core in range(3):
            key = (slice(0, 3), slice(core, core + 1), slice(0, 1))
            for c in range(3):
                loc, rem, dram = count_local_remote_l1_dram_for_getitem(t, key, c)
                if c == core:
                    assert loc == vol and rem == 0 and dram == 0
                else:
                    assert loc == 0 and rem == vol and dram == 0

    def test_example2_uneven_shard_distribution_three_cores(self) -> None:
        spec = NdShardSpec(
            shard_grid=(2, 2, 1),
            shard_shape=(2, 32, 96),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
            num_cores=3,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(4, 64, 96), memory_config=mc)
        vol = 2 * 32 * 96
        k00 = (slice(0, 2), slice(0, 1), slice(0, 3))
        assert count_local_remote_l1_dram_for_getitem(t, k00, 0) == (vol, 0, 0)
        assert count_local_remote_l1_dram_for_getitem(t, k00, 1)[0] == 0
        assert count_local_remote_l1_dram_for_getitem(t, k00, 2)[0] == 0
        k11 = (slice(2, 4), slice(1, 2), slice(0, 3))
        assert count_local_remote_l1_dram_for_getitem(t, k11, 0) == (vol, 0, 0)
        k01 = (slice(0, 2), slice(1, 2), slice(0, 3))
        assert count_local_remote_l1_dram_for_getitem(t, k01, 1) == (vol, 0, 0)
        assert count_local_remote_l1_dram_for_getitem(t, k01, 0)[0] == 0
        k10 = (slice(2, 4), slice(0, 1), slice(0, 3))
        assert count_local_remote_l1_dram_for_getitem(t, k10, 2) == (vol, 0, 0)

    def test_example4_small_core_grid_many_shards_two_cores(self) -> None:
        spec = NdShardSpec(
            shard_shape=(2, 64, 96),
            shard_grid=(2, 3, 1),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
            num_cores=2,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(4, 192, 96), memory_config=mc)
        vol = 2 * 32 * 96
        k0 = (slice(0, 2), slice(0, 1), slice(0, 3))
        assert count_local_remote_l1_dram_for_getitem(t, k0, 0) == (vol, 0, 0)
        assert count_local_remote_l1_dram_for_getitem(t, k0, 1)[0] == 0
        k1 = (slice(0, 2), slice(2, 3), slice(0, 3))
        assert count_local_remote_l1_dram_for_getitem(t, k1, 1) == (vol, 0, 0)
        assert count_local_remote_l1_dram_for_getitem(t, k1, 0)[0] == 0


class TestRowMajorShardingParity:
    """TILE_LAYOUT tile keys vs ROW_MAJOR element keys: same element counts."""

    def test_height_sharded_slice_matches_tile_layout(self) -> None:
        spec = ShardSpec(shard_grid=(4,), shard_shape=(32, 128))
        mc = MemoryConfig(strategy=ShardingStrategy.HEIGHT_SHARDED, shard_spec=spec)
        full_tile = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        full_rm = ttnn.Tensor(
            torch.zeros(128, 128), ttnn.ROW_MAJOR_LAYOUT, memory_config=mc
        )
        key_tile = (slice(0, 1), slice(0, 4))
        key_rm = (slice(0, 32), slice(0, 128))
        for core in range(4):
            assert count_local_remote_l1_dram_for_getitem(
                full_tile, key_tile, core
            ) == count_local_remote_l1_dram_for_getitem(full_rm, key_rm, core)

    def test_block_sharded_slice_matches_tile_layout(self) -> None:
        spec = ShardSpec(shard_grid=(2, 2), shard_shape=(64, 64))
        mc = MemoryConfig(strategy=ShardingStrategy.BLOCK_SHARDED, shard_spec=spec)
        full_tile = ttnn.Tensor(torch.zeros(128, 128), memory_config=mc)
        full_rm = ttnn.Tensor(
            torch.zeros(128, 128), ttnn.ROW_MAJOR_LAYOUT, memory_config=mc
        )
        key_tile = (slice(2, 4), slice(2, 4))
        key_rm = (slice(64, 128), slice(64, 128))
        for core in range(4):
            assert count_local_remote_l1_dram_for_getitem(
                full_tile, key_tile, core
            ) == count_local_remote_l1_dram_for_getitem(full_rm, key_rm, core)

    def test_nd_round_robin_3d_matches_tile_layout(self) -> None:
        spec = NdShardSpec(
            shard_shape=(1, 32, 32),
            shard_grid=(2, 1, 1),
            distribution=ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t_tile = ttnn.Tensor(torch.zeros(2, 32, 32), memory_config=mc)
        t_rm = ttnn.Tensor(
            torch.zeros(2, 32, 32), ttnn.ROW_MAJOR_LAYOUT, memory_config=mc
        )
        for core in range(2):
            kt = (slice(core, core + 1), slice(0, 1), slice(0, 1))
            kr = (slice(core, core + 1), slice(0, 32), slice(0, 32))
            assert count_local_remote_l1_dram_for_getitem(
                t_tile, kt, core
            ) == count_local_remote_l1_dram_for_getitem(t_rm, kr, core)

    def test_shard_origin_from_key_row_major_2d(self) -> None:
        spec = NdShardSpec(
            shard_shape=(64, 64),
            shard_grid=(2, 4),
            distribution=ShardDistributionStrategy.GRID_2D,
        )
        mc = MemoryConfig(strategy=ShardingStrategy.ND_SHARDED, nd_shard_spec=spec)
        t = ttnn.Tensor(torch.zeros(128, 256), ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        assert shard_origin_from_key(t, (slice(64, 128), slice(128, 192))) == (64, 128)
