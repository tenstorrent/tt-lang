# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# up to tt-lang spec, not intended to compile or run currently
import pytest
import torch

import ttnn
import ttl

from utils.correctness import assert_with_ulp
from utils.block_allocation import split_work_to_cores


@ttl.kernel(grid=(13, 10))
def tt_lang_multicore_matmul(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    out: ttnn.Tensor,
    num_cores_group_1: int,
    work_per_core1: int,
    num_cores_group_2: int,
    work_per_core2: int,
):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    buffering_factor = 2
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, 1), buffer_factor=buffering_factor
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )

    def get_tiles_per_core(core_id):
        if core_id < num_cores_group_1:
            return work_per_core1
        elif core_id < num_cores_group_1 + num_cores_group_2:
            return work_per_core2
        else:
            return 0

    def get_start_tile_id(core_id):
        if core_id < num_cores_group_1:
            return core_id * work_per_core1
        elif core_id < num_cores_group_1 + num_cores_group_2:
            return (
                num_cores_group_1 * work_per_core1
                + (core_id - num_cores_group_1) * work_per_core2
            )
        else:
            return 0

    @ttl.compute()
    def mm_compute():
        core_id = ttl.core(dims=1)
        for _ in range(get_tiles_per_core(core_id)):
            with out_dfb.reserve() as out_blk:
                for _ in range(Kt):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        core_id = ttl.core(dims=1)
        for tile_id in range(get_tiles_per_core(core_id)):
            current_tile_id = get_start_tile_id(core_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt
            for k in range(Kt):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_wr = ttl.copy(a[out_row, k], a_blk)
                    b_wr = ttl.copy(b[k, out_col], b_blk)
                    a_wr.wait()
                    b_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        core_id = ttl.core(dims=1)
        for tile_id in range(get_tiles_per_core(core_id)):
            current_tile_id = get_start_tile_id(core_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt
            with out_dfb.wait() as out_blk:
                out_wr = ttl.copy(out_blk, out[out_row, out_col])
                out_wr.wait()


def get_number_of_cores(grid_range):
    total_cores = 0
    if len(grid_range) != 0:
        start = grid_range[0]
        end = grid_range[1]
        x_range = end[0] - start[0] + 1
        y_range = end[1] - start[1] + 1
        total_cores += x_range * y_range
    return total_cores


@pytest.mark.parametrize("M,K,N", [(256, 256, 256), (512, 512, 512)])
def test_multicore_matmul_tt_lang(M, K, N):
    """Test multicore matmul kernel."""
    device = ttnn.open_device(device_id=0)
    num_output_tiles_total = (M * N) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)

    # Use 13x10 grid to match kernel decorator
    grid_size = (10, 13)
    (_, core_group_1, core_group_2, work_per_core1, work_per_core2) = (
        split_work_to_cores(grid_size, num_output_tiles_total, row_wise=True)
    )
    num_cores_group_1 = get_number_of_cores(core_group_1)
    num_cores_group_2 = get_number_of_cores(core_group_2)

    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_multicore_matmul(
        a, b, c, num_cores_group_1, work_per_core1, num_cores_group_2, work_per_core2
    )

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)

    ttnn.close_device(device)


if __name__ == "__main__":
    test_multicore_matmul_tt_lang(640, 640, 640)
