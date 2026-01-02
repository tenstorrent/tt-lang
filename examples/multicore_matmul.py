# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore

from sim import ttl, ttnn
from sim.testing import assert_pcc


def get_number_of_cores(core_range_set):
    """Get total number of cores in a CoreRangeSet.

    Args:
        core_range_set: A CoreRangeSet containing one or more CoreRange objects

    Returns:
        Total number of cores across all ranges
    """
    total_cores = 0
    for core_range in core_range_set.ranges():
        x_range = core_range.end.x - core_range.start.x + 1
        y_range = core_range.end.y - core_range.start.y + 1
        total_cores += x_range * y_range
    return total_cores


@ttl.kernel(grid=(13, 10))
def tt_lang_multicore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."

    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_output_tiles_total = (M * N) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)

    buffering_factor = 2
    a_cb = ttl.make_circular_buffer_like(
        a, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(1, 1), buffer_factor=buffering_factor
    )
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )

    # Get grid size and compute work distribution
    y_size, x_size = ttl.grid_size(dims=2)
    core_grid = ttnn.CoreCoord(x_size, y_size)

    print(f"core_grid: {core_grid}, num_output_tiles_total: {num_output_tiles_total}")
    (
        _,
        all_cores,
        core_group_1,
        core_group_2,
        work_per_core1,
        work_per_core2,
    ) = ttnn.split_work_to_cores(core_grid, num_output_tiles_total, row_wise=True)
    print(
        f"all_cores: {all_cores}, core_group_1: {core_group_1}, core_group_2: {core_group_2}, "
        f"work_per_core1: {work_per_core1}, work_per_core2: {work_per_core2}"
    )

    num_cores_group_1 = get_number_of_cores(core_group_1)
    num_cores_group_2 = get_number_of_cores(core_group_2)

    def get_tiles_per_core(core_id):
        if core_id < num_cores_group_1:
            return work_per_core1
        elif core_id < num_cores_group_1 + num_cores_group_2:
            return work_per_core2
        else:  # no work assigned
            return 0

    def get_start_tile_id(core_id):
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
    def mm_compute():
        core_id = ttl.core(dims=1)
        num_tiles = get_tiles_per_core(core_id)

        for _ in range(num_tiles):
            # Translated from: with out_cb.reserve() as out_blk:
            out_blk = out_cb.reserve()

            # Accumulate over K dimension
            for k_idx in range(Kt):
                # Translated from: with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                a_blk = a_cb.wait()
                b_blk = b_cb.wait()

                # Translated from: out_blk.store(a_blk @ b_blk, acc=True)
                # Manual accumulation: first iteration stores, subsequent add
                if k_idx == 0:
                    result = a_blk @ b_blk
                else:
                    result = out_blk + (a_blk @ b_blk)
                out_blk.store(result)

                a_cb.pop()
                b_cb.pop()

            out_cb.push()

    @ttl.datamovement()
    def mm_reader():
        core_id = ttl.core(dims=1)
        num_tiles = get_tiles_per_core(core_id)

        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_id in range(num_tiles):
            current_tile_id = get_start_tile_id(core_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt

            for k in range(Kt):
                # Translated from: with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                a_blk = a_cb.reserve()
                b_blk = b_cb.reserve()

                # Note: Using slice notation for tile indexing
                a_wr = ttl.copy(a[out_row : out_row + 1, k : k + 1], a_blk)
                b_wr = ttl.copy(b[k : k + 1, out_col : out_col + 1], b_blk)
                a_wr.wait()
                b_wr.wait()

                a_cb.push()
                b_cb.push()

    @ttl.datamovement()
    def mm_writer():
        core_id = ttl.core(dims=1)
        num_tiles = get_tiles_per_core(core_id)

        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_id in range(num_tiles):
            current_tile_id = get_start_tile_id(core_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt

            # Translated from: with out_cb.wait() as out_blk:
            out_blk = out_cb.wait()

            out_wr = ttl.copy(
                out_blk, out[out_row : out_row + 1, out_col : out_col + 1]
            )
            out_wr.wait()

            out_cb.pop()

    # Execute the program
    ttl.Program(mm_compute, mm_reader, mm_writer)(a, b, out)


def main() -> None:
    # Test with matrices that are multiples of tile size
    M, K, N = 128, 256, 64
    a = ttnn.rand((M, K), dtype=ttnn.float32)
    b = ttnn.rand((K, N), dtype=ttnn.float32)
    out = ttnn.empty((M, N), dtype=ttnn.float32)

    print(f"Matrix multiplication: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"Tiles: A={M//32}x{K//32}, B={K//32}x{N//32}, Out={M//32}x{N//32}")
    print(f"Total output tiles: {(M//32) * (N//32)}")
    print(f"Grid: 8x8 = 64 cores")

    tt_lang_multicore_matmul(a, b, out)

    # Compute golden result
    golden = a @ b

    # Verify correctness with relaxed tolerance for matmul
    assert_pcc(golden, out, rtol=1e-4, atol=1e-4)
    print("tt_lang_multicore_matmul: success")


if __name__ == "__main__":
    main()
