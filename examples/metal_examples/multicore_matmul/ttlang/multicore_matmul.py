# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np

import ttnn
import ttl

from utils.correctness import assert_with_ulp


def visualize_diff_tensor(diff_tensor):
    diff = diff_tensor.float().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(diff, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title(f"Absolute Error ({diff.shape[0]}x{diff.shape[1]})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("multicore_matmul_diff_with_tmp.png", dpi=150)
    print(f"Saved to multicore_matmul_diff_with_tmp.png  |  max={diff.max():.4e}  mean={diff.mean():.4e}")
    plt.show()


@ttl.kernel(grid=('auto'))
def tt_lang_multicore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_output_tiles = Mt * Nt
    buffering_factor = 1
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, 1), buffer_factor=buffering_factor
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )
    # tmp_dfb = ttl.make_dataflow_buffer_like(
    #     out, shape=(1, 1), buffer_factor=1
    # )

    (device_grid_x, device_grid_y) = ttl.grid_size(dims=2)
    total_cores = device_grid_x * device_grid_y
    device_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_x - 1, device_grid_y - 1))]
    )
    (_, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2) = ttnn.split_work_to_cores(
        device_core_grid, num_output_tiles, row_wise=True
    )
    
    num_cores_group_1 = core_group_1.num_cores()
    num_cores_group_2 = core_group_2.num_cores()
    print(f"num_cores_group_1: {num_cores_group_1}, num_cores_group_2: {num_cores_group_2}")
    print(f"work_per_core1: {work_per_core1}, work_per_core2: {work_per_core2}")
    print(f"Kt: {Kt}")

    @ttl.compute()
    def mm_compute():
        core_x, core_y = ttl.core(dims=2)
        core_id = core_y * device_grid_x + core_x
        if core_id < num_cores_group_1:
            for _ in range(work_per_core1):
                with out_dfb.reserve() as out_blk:
                    for _ in range(Kt):
                        # with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, tmp_dfb.reserve() as tmp:
                        #     tmp.store(ttl.math.matmul(a_blk, b_blk))
                        # with tmp_dfb.wait() as tmp_blk:
                        #     out_blk.store(out_blk + tmp_blk)
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                            out_blk.store(ttl.math.matmul(a_blk, b_blk))
        elif core_id < num_cores_group_1 + num_cores_group_2:
            for _ in range(work_per_core2):
                with out_dfb.reserve() as out_blk:
                    for _ in range(Kt):
                        # with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, tmp_dfb.reserve() as tmp:
                        #     tmp.store(ttl.math.matmul(a_blk, b_blk))
                        # with tmp_dfb.wait() as tmp_blk:
                        #     out_blk.store(out_blk + tmp_blk)
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                            out_blk.store(ttl.math.matmul(a_blk, b_blk))

    @ttl.datamovement()
    def mm_reader():
        core_x, core_y = ttl.core(dims=2)
        core_id = core_y * device_grid_x + core_x
        if core_id < num_cores_group_1:
            start_tile_id = core_id * work_per_core1
            for tile_id in range(work_per_core1):
                current_tile_id = start_tile_id + tile_id
                out_row = current_tile_id // Nt
                out_col = current_tile_id % Nt
                for k in range(Kt):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        a_wr = ttl.copy(a[out_row, k], a_blk)
                        b_wr = ttl.copy(b[k, out_col], b_blk)
                        a_wr.wait()
                        b_wr.wait()
        elif core_id < num_cores_group_1 + num_cores_group_2:
            start_tile_id = num_cores_group_1 * work_per_core1 + (core_id - num_cores_group_1) * work_per_core2
            for tile_id in range(work_per_core2):
                current_tile_id = start_tile_id + tile_id
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
        core_x, core_y = ttl.core(dims=2)
        core_id = core_y * device_grid_x + core_x
        if core_id < num_cores_group_1:
            start_tile_id = core_id * work_per_core1
            for tile_id in range(work_per_core1):
                current_tile_id = start_tile_id + tile_id
                out_row = current_tile_id // Nt
                out_col = current_tile_id % Nt
                with out_dfb.wait() as out_blk:
                    out_wr = ttl.copy(out_blk, out[out_row, out_col])
                    out_wr.wait()
        elif core_id < num_cores_group_1 + num_cores_group_2:
            start_tile_id = num_cores_group_1 * work_per_core1 + (core_id - num_cores_group_1) * work_per_core2
            for tile_id in range(work_per_core2):
                current_tile_id = start_tile_id + tile_id
                out_row = current_tile_id // Nt
                out_col = current_tile_id % Nt
                with out_dfb.wait() as out_blk:
                    out_wr = ttl.copy(out_blk, out[out_row, out_col])
                    out_wr.wait()



@pytest.mark.parametrize("M,K,N", [
    (320,32,320)
    #(640, 640, 640)
])
def test_multicore_matmul_tt_lang(M, K, N):
    """Test multicore matmul kernel."""
    device = ttnn.open_device(device_id=0)
    dram = ttnn.DRAM_MEMORY_CONFIG

    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

    tt_lang_multicore_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)

    diff_tensor = (result - golden).abs()

    visualize_diff_tensor(diff_tensor)

    assert_with_ulp(golden, result)

    ttnn.close_device(device)


if __name__ == "__main__":
    test_multicore_matmul_tt_lang(640, 640, 640)
