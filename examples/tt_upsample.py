# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# TTLANG_HARDWARE_CI: xfail-compiler
import pytest
import torch

import ttnn
import ttl

from utils.correctness import assert_with_ulp
from utils.block_allocation import split_work_to_nodes


@ttl.operation(grid=(8, 8))
def tt_lang_upsample_nearest_rowwise_interleaved(
    input_t: ttnn.Tensor,
    output: ttnn.Tensor,
    scale_factor: tuple[int, int],
):
    # input and output expected to be 4D tensors already in NxHxWxC row-wise interleaved layout
    (N, H, W, C) = input_t.shape

    buffer_factor = 1
    io_dfb = ttl.make_dataflow_buffer_like(
        input_t, shape=(C,), buffer_factor=buffer_factor
    )

    num_rows = N * H * W
    print(f"num_rows: {num_rows}")
    (all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2) = (
        split_work_to_nodes((ttl.grid_size(dims=1),), num_rows)
    )
    print(
        f"all_cores: {all_cores}, core_group_1: {core_group_1}, core_group_2: {core_group_2}, work_per_core1: {work_per_core1}, work_per_core2: {work_per_core2}"
    )

    num_cores_group_1 = (
        core_group_1[1][-1] - core_group_1[0][-1] + 1 if core_group_1 else 0
    )
    num_cores_group_2 = (
        core_group_2[1][-1] - core_group_2[0][-1] + 1 if core_group_2 else 0
    )

    def get_work_per_core(core_id):
        if core_id < num_cores_group_1:
            return work_per_core1
        elif core_id < num_cores_group_1 + num_cores_group_2:
            return work_per_core2
        else:  # no work assigned
            return 0

    def get_start_row(core_id):
        if core_id < num_cores_group_1:
            return core_id * work_per_core1
        elif core_id < num_cores_group_1 + num_cores_group_2:
            return (
                num_cores_group_1 * work_per_core1
                + (core_id - num_cores_group_1) * work_per_core2
            )
        else:  # no work assigned
            return 0

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def mm_reader():
        core_id = ttl.node(dims=1)
        for row in range(get_work_per_core(core_id)):
            row_idx = get_start_row(core_id) + row
            n = row_idx // (H * W)
            rem = row_idx % (H * W)
            h = rem // W
            w = rem % W
            with io_dfb.reserve() as in_blk:
                in_wr = ttl.copy(input_t[n, h, w, :], in_blk)
                in_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        core_id = ttl.node(dims=1)
        for row in range(get_work_per_core(core_id)):
            row_idx = get_start_row(core_id) + row
            n = row_idx // (H * W)
            rem = row_idx % (H * W)
            h = rem // W
            w = rem % W
            with io_dfb.wait() as out_blk:
                for h_1 in range(scale_factor[0]):
                    for w_1 in range(scale_factor[1]):
                        out_wr = ttl.copy(
                            out_blk,
                            output[
                                n,
                                h * scale_factor[0] + h_1,
                                w * scale_factor[1] + w_1,
                                :,
                            ],
                        )
                        out_wr.wait()


@pytest.mark.parametrize("input_shape,", [([1, 64, 64, 64]), ([2, 32, 32, 128])])
@pytest.mark.parametrize("scale_factor", [(2, 2)])
def test_tt_lang_upsample_nearest_rowwise_interleaved(input_shape, scale_factor):
    device = ttnn.open_device(device_id=0)

    input_tensor = ttnn.rand(
        input_shape,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_shape = (
        input_shape[0],
        input_shape[1] * scale_factor[0],
        input_shape[2] * scale_factor[1],
        input_shape[3],
    )
    output_tensor = ttnn.empty(
        output_shape,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_lang_upsample_nearest_rowwise_interleaved(
        input_tensor, output_tensor, scale_factor
    )

    golden_tensor = ttnn.upsample(input_tensor, scale_factor)
    print(f"golden_tensor: {golden_tensor}")
    print(f"output_tensor: {output_tensor}")

    assert_with_ulp(output_tensor.to_torch(), golden_tensor.to_torch(), ulp_threshold=1)
    print("Test passed!")


test_tt_lang_upsample_nearest_rowwise_interleaved([1, 64, 64, 64], [2, 2])
test_tt_lang_upsample_nearest_rowwise_interleaved([2, 32, 32, 128], [2, 2])
