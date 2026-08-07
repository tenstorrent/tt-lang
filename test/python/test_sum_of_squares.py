# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Full scalar multiply-reduction in one capacity-fitting DST section."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from utils.correctness import assert_allclose  # noqa: E402

WIDTH = 7168
COMPUTE_TILE = (32, 32)
NUM_TILES = WIDTH // (COMPUTE_TILE[0] * COMPUTE_TILE[1])


@ttl.operation(
    grid=(1, 1),
    fp32_dest_acc_en=False,
    dst_full_sync_en=False,
    options="--no-ttl-reduce-full-fp32",
)
def sum_of_squares(input_tensor, output_tensor):
    input_dfb = ttl.make_tensor_backed_dfb(
        input_tensor, shape=(1, NUM_TILES), tile=COMPUTE_TILE
    )
    output_dfb = ttl.make_tensor_backed_dfb(
        output_tensor, shape=(1, 1), tile=COMPUTE_TILE
    )

    @ttl.compute()
    def compute():
        with input_dfb.wait() as input_block, output_dfb.reserve() as output_block:
            output_block.store(
                ttl.math.reduce_sum(input_block * input_block, dims=[0, 1])
            )

    @ttl.datamovement()
    def read():
        input_dfb.publish()

    @ttl.datamovement()
    def write():
        with output_dfb.wait():
            pass


def make_scaled_sum_of_squares(scale):
    @ttl.operation(
        grid=(1, 1),
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        options="--no-ttl-reduce-full-fp32",
    )
    def sum_of_squares(input_tensor, output_tensor):
        input_dfb = ttl.make_tensor_backed_dfb(
            input_tensor, shape=(1, NUM_TILES), tile=COMPUTE_TILE
        )
        output_dfb = ttl.make_tensor_backed_dfb(
            output_tensor, shape=(1, 1), tile=COMPUTE_TILE
        )

        @ttl.compute()
        def compute():
            with (
                input_dfb.wait() as input_block,
                output_dfb.reserve() as output_block,
            ):
                output_block.store(
                    scale * ttl.math.reduce_sum(input_block * input_block, dims=[0, 1])
                )

        @ttl.datamovement()
        def read():
            input_dfb.publish()

        @ttl.datamovement()
        def write():
            with output_dfb.wait():
                pass

    return sum_of_squares


SUM_OF_SQUARES_KERNELS = {
    1.0: sum_of_squares,
    0.5: make_scaled_sum_of_squares(0.5),
}


def one_core_l1_height_sharded(shape):
    core = ttnn.CoreCoord(0, 0)
    core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_ranges, shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("scale", (1.0, 0.5), ids=("unit", "nonunit"))
def test_sum_of_squares(device, scale):
    """Square and reduce seven BF16 pages with the requested scale."""
    torch.manual_seed(0)
    input_torch = torch.randn((1, WIDTH), dtype=torch.bfloat16)
    output_torch = torch.zeros(COMPUTE_TILE, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, COMPUTE_TILE[1])),
        device=device,
        memory_config=one_core_l1_height_sharded(input_torch.shape),
    )
    output_tensor = ttnn.from_torch(
        output_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile(COMPUTE_TILE),
        device=device,
        memory_config=one_core_l1_height_sharded(output_torch.shape),
    )

    SUM_OF_SQUARES_KERNELS[scale](input_tensor, output_tensor)

    result = ttnn.to_torch(output_tensor).float()[0, 0]
    expected = (scale * input_torch.float().square().sum()).to(torch.bfloat16).float()
    assert_allclose(result, expected, rtol=0.12, atol=1.0)
