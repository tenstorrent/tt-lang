# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm row-fusion and ordinary-lowering device coverage."""

import pytest
import torch
import ttl
from ttl import ttl_api

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose, assert_pcc  # noqa: E402

TILE_WIDTH = 32
EPSILON = 1.0e-5
ROW_NORMALIZATION_TARGET_ARCHS = frozenset({"blackhole"})


def require_row_normalization_schedule(device):
    """Skip schedule-specific cases when the pinned LLK lacks the schedule."""
    target_arch = ttl_api._detect_device_arch(device)
    if target_arch is None:
        pytest.fail("unable to determine the test device architecture")
    if target_arch not in ROW_NORMALIZATION_TARGET_ARCHS:
        pytest.skip(
            "the pinned target dependency does not implement the fixed-block "
            "row-normalization schedule"
        )


def make_rmsnorm_kernel(
    tile_height,
    num_tiles,
    gamma_mode,
    fp32_dest_acc_en=None,
    dst_full_sync_en=None,
):
    scale = 1.0 / (num_tiles * tile_height * TILE_WIDTH)

    if gamma_mode == "none":

        @ttl.operation(
            grid=(1, 1),
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
        )
        def rmsnorm_kernel(input_tensor, gamma_tensor, output_tensor):
            input_dfb = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, num_tiles), block_count=2
            )
            output_dfb = ttl.make_dataflow_buffer_like(
                output_tensor, shape=(1, num_tiles), block_count=2
            )

            @ttl.compute()
            def compute():
                with (
                    input_dfb.wait() as input_block,
                    output_dfb.reserve() as output_block,
                ):
                    squared = input_block * input_block
                    reduced = ttl.math.reduce_sum(squared, dims=[0, 1])
                    mean_square = reduced * scale
                    biased = mean_square + ttl.block.fill(
                        EPSILON,
                        shape=mean_square.shape,
                        tile=(tile_height, TILE_WIDTH),
                    )
                    inverse_rms = ttl.math.rsqrt(biased)
                    scalar = ttl.block.broadcast(
                        inverse_rms, dims=[0, 1], shape=input_block.shape
                    )
                    output_block.store(input_block * scalar)

            @ttl.datamovement()
            def dm_read():
                with input_dfb.reserve() as input_block:
                    ttl.copy(input_tensor[0:1, 0:num_tiles], input_block).wait()

            @ttl.datamovement()
            def dm_write():
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output_tensor[0:1, 0:num_tiles]).wait()

        return rmsnorm_kernel

    if gamma_mode == "full":

        @ttl.operation(
            grid=(1, 1),
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
        )
        def rmsnorm_kernel(input_tensor, gamma_tensor, output_tensor):
            input_dfb = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, num_tiles), block_count=2
            )
            gamma_dfb = ttl.make_dataflow_buffer_like(
                gamma_tensor, shape=(1, num_tiles), block_count=2
            )
            output_dfb = ttl.make_dataflow_buffer_like(
                output_tensor, shape=(1, num_tiles), block_count=2
            )

            @ttl.compute()
            def compute():
                with (
                    input_dfb.wait() as input_block,
                    gamma_dfb.wait() as gamma_block,
                    output_dfb.reserve() as output_block,
                ):
                    squared = input_block * input_block
                    reduced = ttl.math.reduce_sum(squared, dims=[0, 1])
                    mean_square = reduced * scale
                    biased = mean_square + ttl.block.fill(
                        EPSILON,
                        shape=mean_square.shape,
                        tile=(tile_height, TILE_WIDTH),
                    )
                    inverse_rms = ttl.math.rsqrt(biased)
                    scalar = ttl.block.broadcast(
                        inverse_rms, dims=[0, 1], shape=input_block.shape
                    )
                    output_block.store(input_block * scalar * gamma_block)

            @ttl.datamovement()
            def dm_read():
                with input_dfb.reserve() as input_block:
                    ttl.copy(input_tensor[0:1, 0:num_tiles], input_block).wait()
                with gamma_dfb.reserve() as gamma_block:
                    ttl.copy(gamma_tensor[0:1, 0:num_tiles], gamma_block).wait()

            @ttl.datamovement()
            def dm_write():
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output_tensor[0:1, 0:num_tiles]).wait()

        return rmsnorm_kernel

    if gamma_mode == "column_broadcast":

        @ttl.operation(
            grid=(1, 1),
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=dst_full_sync_en,
        )
        def rmsnorm_kernel(input_tensor, gamma_tensor, output_tensor):
            input_dfb = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, num_tiles), block_count=2
            )
            gamma_dfb = ttl.make_dataflow_buffer_like(
                gamma_tensor, shape=(1, 1), block_count=2
            )
            output_dfb = ttl.make_dataflow_buffer_like(
                output_tensor, shape=(1, num_tiles), block_count=2
            )

            @ttl.compute()
            def compute():
                with (
                    input_dfb.wait() as input_block,
                    gamma_dfb.wait() as gamma_block,
                    output_dfb.reserve() as output_block,
                ):
                    squared = input_block * input_block
                    reduced = ttl.math.reduce_sum(squared, dims=[0, 1])
                    mean_square = reduced * scale
                    biased = mean_square + ttl.block.fill(
                        EPSILON,
                        shape=mean_square.shape,
                        tile=(tile_height, TILE_WIDTH),
                    )
                    inverse_rms = ttl.math.rsqrt(biased)
                    scalar = ttl.block.broadcast(
                        inverse_rms, dims=[0, 1], shape=input_block.shape
                    )
                    repeated_gamma = ttl.block.broadcast(
                        gamma_block, dims=[1], shape=input_block.shape
                    )
                    output_block.store(input_block * scalar * repeated_gamma)

            @ttl.datamovement()
            def dm_read():
                with input_dfb.reserve() as input_block:
                    ttl.copy(input_tensor[0:1, 0:num_tiles], input_block).wait()
                with gamma_dfb.reserve() as gamma_block:
                    ttl.copy(gamma_tensor[0:1, 0:1], gamma_block).wait()

            @ttl.datamovement()
            def dm_write():
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output_tensor[0:1, 0:num_tiles]).wait()

        return rmsnorm_kernel

    raise ValueError(f"unsupported gamma mode: {gamma_mode}")


def make_materialized_rmsnorm_kernel(tile_height, num_tiles):
    """Build an equivalent RMSNorm with explicit intermediate DFBs."""
    scale = 1.0 / (num_tiles * tile_height * TILE_WIDTH)

    @ttl.operation(grid=(1, 1))
    def rmsnorm_kernel(input_tensor, gamma_tensor, output_tensor):
        input_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, num_tiles), block_count=2
        )
        squared_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, num_tiles), block_count=2
        )
        reduced_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=2
        )
        inverse_rms_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_tensor, shape=(1, num_tiles), block_count=2
        )

        @ttl.compute()
        def compute():
            with input_dfb.wait() as input_block:
                with squared_dfb.reserve() as squared_block:
                    squared_block.store(input_block * input_block)

                with squared_dfb.wait() as squared_block:
                    with reduced_dfb.reserve() as reduced_block:
                        reduced_block.store(
                            ttl.math.reduce_sum(squared_block, dims=[0, 1])
                        )

                with reduced_dfb.wait() as reduced_block:
                    with inverse_rms_dfb.reserve() as inverse_rms_block:
                        mean_square = reduced_block * scale
                        biased = mean_square + ttl.block.fill(
                            EPSILON,
                            shape=mean_square.shape,
                            tile=(tile_height, TILE_WIDTH),
                        )
                        inverse_rms_block.store(ttl.math.rsqrt(biased))

                with (
                    inverse_rms_dfb.wait() as inverse_rms_block,
                    output_dfb.reserve() as output_block,
                ):
                    scalar = ttl.block.broadcast(
                        inverse_rms_block, dims=[0, 1], shape=input_block.shape
                    )
                    output_block.store(input_block * scalar)

        @ttl.datamovement()
        def dm_read():
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0:1, 0:num_tiles], input_block).wait()

        @ttl.datamovement()
        def dm_write():
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0:1, 0:num_tiles]).wait()

    return rmsnorm_kernel


GAMMA_MODES = ("none", "full", "column_broadcast")
ROW_CASES = ((16, 1, 512), (16, 3, 1536), (32, 7, 7168))
RMSNORM_KERNELS = {
    (tile_height, num_tiles, gamma_mode): make_rmsnorm_kernel(
        tile_height, num_tiles, gamma_mode
    )
    for tile_height, num_tiles, _ in ROW_CASES
    for gamma_mode in GAMMA_MODES
}
KERNEL_CONFIGS = ((False, False), (True, False), (False, True), (True, True))
CONFIG_RMSNORM_KERNELS = {
    (fp32_dest_acc_en, dst_full_sync_en): make_rmsnorm_kernel(
        32,
        4,
        "none",
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
    )
    for fp32_dest_acc_en, dst_full_sync_en in KERNEL_CONFIGS
}
FIVE_TILE_FP32_KERNEL = make_rmsnorm_kernel(
    32,
    5,
    "none",
    fp32_dest_acc_en=True,
)
MATERIALIZED_RMSNORM_KERNEL = make_materialized_rmsnorm_kernel(16, 3)
TENSOR_FACTORIES = (to_dram, to_l1)


def run_rmsnorm(
    device, tile_height, num_tiles, width, gamma_mode, kernel, tensor_factory
):
    """Run one RMSNorm kernel and return its result and reference."""
    shape = (tile_height, num_tiles * TILE_WIDTH)
    assert shape[0] * shape[1] == width
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_shape = (
        (tile_height, TILE_WIDTH) if gamma_mode == "column_broadcast" else shape
    )
    gamma_torch = torch.randn(gamma_shape, dtype=torch.bfloat16)
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)

    tile = (tile_height, TILE_WIDTH)
    input_tensor = tensor_factory(input_torch, device, tile=tile)
    gamma_tensor = tensor_factory(gamma_torch, device, tile=tile)
    output_tensor = tensor_factory(output_torch, device, tile=tile)

    kernel(input_tensor, gamma_tensor, output_tensor)
    result = ttnn.to_torch(output_tensor).float()

    input_float = input_torch.float()
    expected = input_float * torch.rsqrt(input_float.square().mean() + EPSILON)
    if gamma_mode == "full":
        expected = expected * gamma_torch.float()
    elif gamma_mode == "column_broadcast":
        expected = expected * gamma_torch[:, :1].float().repeat(
            1, num_tiles * TILE_WIDTH
        )
    return result, expected


def assert_rmsnorm_close(result, expected):
    """Check RMSNorm correlation and magnitude at bf16 precision."""
    assert_pcc(expected, result, threshold=0.999)
    assert_allclose(result, expected, rtol=0.05, atol=0.4)


@pytest.mark.parametrize(
    "tile_height, num_tiles, width",
    ROW_CASES,
    ids=[str(row_case[2]) for row_case in ROW_CASES],
)
@pytest.mark.parametrize("gamma_mode", GAMMA_MODES)
@pytest.mark.parametrize("tensor_factory", TENSOR_FACTORIES, ids=("dram", "l1"))
def test_rmsnorm(device, tile_height, num_tiles, width, gamma_mode, tensor_factory):
    """Validate RMSNorm semantics at benchmark widths and gamma modes."""
    # The specialized hardware operation intentionally accepts bf16 tiles only.
    # Column-broadcast gamma exercises ordinary compute creation because the
    # fixed-block schedule accepts only absent or full-row gamma. Targets whose
    # dependencies lack that schedule use ordinary compute creation for all
    # modes.
    result, expected = run_rmsnorm(
        device,
        tile_height,
        num_tiles,
        width,
        gamma_mode,
        RMSNORM_KERNELS[(tile_height, num_tiles, gamma_mode)],
        tensor_factory,
    )
    assert_rmsnorm_close(result, expected)


@pytest.mark.parametrize(
    "fp32_dest_acc_en, dst_full_sync_en",
    KERNEL_CONFIGS,
    ids=["bf16-half", "fp32-half", "bf16-full", "fp32-full"],
)
@pytest.mark.parametrize("tensor_factory", TENSOR_FACTORIES, ids=("dram", "l1"))
def test_rmsnorm_kernel_config(
    device, fp32_dest_acc_en, dst_full_sync_en, tensor_factory
):
    """Exercise the fused LLK under every DST register configuration."""
    require_row_normalization_schedule(device)
    result, expected = run_rmsnorm(
        device,
        tile_height=32,
        num_tiles=4,
        width=4096,
        gamma_mode="none",
        kernel=CONFIG_RMSNORM_KERNELS[(fp32_dest_acc_en, dst_full_sync_en)],
        tensor_factory=tensor_factory,
    )
    assert_rmsnorm_close(result, expected)


@pytest.mark.parametrize("tensor_factory", TENSOR_FACTORIES, ids=("dram", "l1"))
def test_rmsnorm_five_tile_fp32_dest(
    device,
    monkeypatch,
    tmp_path,
    tensor_factory,
):
    """Select full synchronization when FP32 destination needs five slots."""
    require_row_normalization_schedule(device)
    final_mlir = tmp_path / "rmsnorm.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    result, expected = run_rmsnorm(
        device,
        tile_height=32,
        num_tiles=5,
        width=5120,
        gamma_mode="none",
        kernel=FIVE_TILE_FP32_KERNEL,
        tensor_factory=tensor_factory,
    )
    assert_rmsnorm_close(result, expected)
    assert final_mlir.read_text().count("tile_regs_acquire") == 1


@pytest.mark.parametrize("tensor_factory", TENSOR_FACTORIES, ids=("dram", "l1"))
def test_rmsnorm_matches_materialized(device, tensor_factory):
    """Compare scalar-retaining fusion with explicit DFB materialization."""
    tile_height = 16
    num_tiles = 3
    shape = (tile_height, num_tiles * TILE_WIDTH)
    input_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.zeros(shape, dtype=torch.bfloat16)
    output_torch = torch.zeros(shape, dtype=torch.bfloat16)
    materialized_output_torch = torch.zeros(shape, dtype=torch.bfloat16)

    tile = (tile_height, TILE_WIDTH)
    input_tensor = tensor_factory(input_torch, device, tile=tile)
    gamma_tensor = tensor_factory(gamma_torch, device, tile=tile)
    output_tensor = tensor_factory(output_torch, device, tile=tile)
    materialized_output_tensor = tensor_factory(
        materialized_output_torch, device, tile=tile
    )

    RMSNORM_KERNELS[(tile_height, num_tiles, "none")](
        input_tensor, gamma_tensor, output_tensor
    )
    MATERIALIZED_RMSNORM_KERNEL(input_tensor, gamma_tensor, materialized_output_tensor)

    result = ttnn.to_torch(output_tensor).float()
    materialized_result = ttnn.to_torch(materialized_output_tensor).float()
    assert_pcc(materialized_result, result, threshold=0.999)
    assert_allclose(result, materialized_result, rtol=0.05, atol=0.2)
