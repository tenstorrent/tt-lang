#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test comparing split_work_to_cores with ttnn.split_work_to_cores
get_large_matmul_params is compared with hard coded expected values
"""
import pytest

from ttlang.utils.block_allocation import split_work_to_cores, get_large_matmul_params

ttnn = pytest.importorskip("ttnn")


def extract_coords_from_ttnn_corerangeset(core_range_set):
    """Extract all start and end coordinates from a ttnn CoreRangeSet"""
    if not core_range_set.ranges():
        return []

    coords = []
    for r in core_range_set.ranges():
        coords.append(((r.start.y, r.start.x), (r.end.y, r.end.x)))
    return coords


@pytest.mark.parametrize(
    "grid_size_tuple,units,row_wise",
    [
        # Test cases with more work than cores
        ((8, 8), 100, True),
        ((8, 8), 100, False),
        ((8, 8), 65, True),
        ((8, 8), 129, True),
        # Test even distribution
        ((8, 8), 64, True),
        ((8, 8), 128, True),
        # Test with different grid sizes
        ((4, 8), 50, True),
        ((7, 9), 100, False),
        # Test fewer units than cores
        ((8, 8), 10, True),
        ((8, 8), 20, False),
        ((8, 8), 1, True),
        # Test edge cases
        ((8, 8), 63, True),
        ((8, 8), 127, True),
    ],
)
def test_split_work_to_cores(grid_size_tuple, units, row_wise):
    """Compare results from split_work_to_cores and ttnn.split_work_to_cores"""
    # Call new function
    new_result = split_work_to_cores(grid_size_tuple, units, row_wise)
    new_total, new_g1, new_g2, new_w1, new_w2 = new_result

    # Call ttnn function
    # Create CoreRangeSet from grid_size_tuple
    num_cores_x = grid_size_tuple[-1]
    num_cores_y = grid_size_tuple[-2]
    ttnn_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
            )
        ]
    )

    ttnn_result = ttnn.split_work_to_cores(ttnn_grid, units, row_wise)
    ttnn_total, ttnn_all, ttnn_g1, ttnn_g2, ttnn_w1, ttnn_w2 = ttnn_result

    # Extract coordinates from ttnn function
    ttnn_g1_coords = extract_coords_from_ttnn_corerangeset(ttnn_g1)
    ttnn_g2_coords = extract_coords_from_ttnn_corerangeset(ttnn_g2)

    # Verify work distribution matches
    assert new_w1 == ttnn_w1, f"Work per core G1 mismatch: {new_w1} vs {ttnn_w1}"
    assert new_w2 == ttnn_w2, f"Work per core G2 mismatch: {new_w2} vs {ttnn_w2}"

    # Calculate total cores in each group from ttnn
    ttnn_g1_num_cores = sum(
        (end[1] - start[1] + 1) * (end[0] - start[0] + 1)
        for start, end in ttnn_g1_coords
    )
    ttnn_g2_num_cores = sum(
        (end[1] - start[1] + 1) * (end[0] - start[0] + 1)
        for start, end in ttnn_g2_coords
    )

    # Verify total work matches
    new_total_work = ttnn_g1_num_cores * new_w1 + ttnn_g2_num_cores * new_w2
    ttnn_total_work = ttnn_g1_num_cores * ttnn_w1 + ttnn_g2_num_cores * ttnn_w2
    assert (
        new_total_work == ttnn_total_work == units
    ), f"Total work mismatch: {new_total_work} vs {ttnn_total_work} vs {units}"

    # Verify group 1 coordinates
    if new_g1 and ttnn_g1_coords:
        new_g1_start, new_g1_end = new_g1
        ttnn_g1_first_start = ttnn_g1_coords[0][0]
        ttnn_g1_last_end = ttnn_g1_coords[-1][1]
        assert (
            new_g1_start == ttnn_g1_first_start and new_g1_end == ttnn_g1_last_end
        ), f"Group 1 coordinates mismatch: new {new_g1_start} -> {new_g1_end}, ttnn {ttnn_g1_first_start} -> {ttnn_g1_last_end}"

    # Verify group 2 coordinates
    if new_g2 and ttnn_g2_coords:
        new_g2_start, new_g2_end = new_g2
        ttnn_g2_first_start = ttnn_g2_coords[0][0]
        ttnn_g2_last_end = ttnn_g2_coords[-1][1]
        assert (
            new_g2_start == ttnn_g2_first_start and new_g2_end == ttnn_g2_last_end
        ), f"Group 2 coordinates mismatch: new {new_g2_start} -> {new_g2_end}, ttnn {ttnn_g2_first_start} -> {ttnn_g2_last_end}"

    # Check empty groups match
    if not new_g1:
        assert not ttnn_g1_coords, "Group 1 empty mismatch"
    if not new_g2:
        assert not ttnn_g2_coords, "Group 2 empty mismatch"


@pytest.mark.parametrize(
    "num_cores_y,num_cores_x,Mt,Nt,K_block_t,expected",
    [
        # Blackhole 13x10 grid tests
        # NOTE: K_block_t values are reduced to fit within the 400-tile L1 budget
        # The constraint is: 2*K*MT + 2*K*N + M*N ≤ 400
        # Wormhole 8x8 grid tests
        (8, 8, 1, 1, 8, (1, 1, 1, 1)),
        (8, 8, 7, 7, 8, (7, 1, 7, 1)),
        (8, 8, 13, 13, 4, (13, 13, 1, 1)),
        (8, 8, 16, 16, 8, (16, 2, 4, 2)),
        (8, 8, 21, 42, 8, (7, 6, 7, 1)),
        (8, 8, 32, 32, 8, (16, 4, 4, 2)),
        (8, 8, 35, 70, 8, (7, 10, 7, 1)),
        (8, 8, 64, 64, 8, (8, 8, 4, 2)),
        (8, 8, 77, 77, 4, (11, 11, 1, 1)),
        (8, 8, 128, 128, 2, (16, 16, 4, 2)),
        (8, 8, 256, 256, 1, (0, 0, 0, 0)),  # Too large even with K=1
        # Blackhole 13x10 grid tests
        (13, 10, 21, 21, 8, (7, 3, 7, 1)),
        (13, 10, 32, 32, 8, (16, 4, 4, 2)),
        (13, 10, 35, 35, 8, (7, 5, 7, 1)),
        (13, 10, 40, 65, 8, (5, 13, 5, 1)),
        (13, 10, 64, 128, 4, (8, 16, 4, 2)),
        (13, 10, 64, 256, 1, (8, 32, 4, 2)),
        (13, 10, 77, 77, 4, (11, 11, 1, 1)),
        (13, 10, 128, 128, 2, (16, 16, 4, 2)),
        (13, 10, 128, 256, 1, (0, 0, 0, 0)),  # Too large even with K=1
    ],
)
def test_get_large_matmul_params(num_cores_y, num_cores_x, Mt, Nt, K_block_t, expected):
    """
    Test get_large_matmul_params with various grid sizes and matrix dimensions.

    This test captures the current behavior as a baseline for regression testing.
    Expected values were generated by running the current implementation and
    represent the optimal block and subblock configurations for each test case.

    Args:
        Mt: Total number of tiles in M dimension
        Nt: Total number of tiles in N dimension
        num_cores_y: Number of cores in Y dimension
        num_cores_x: Number of cores in X dimension
        K_block_t: K dimension block width
        expected: Tuple of (block_h, block_w, subblock_h, subblock_w)
    """
    result = get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, K_block_t)

    assert result.block_h == expected[0], (
        f"block_h mismatch for Mt={Mt}, Nt={Nt}, cores=({num_cores_y},{num_cores_x}), "
        f"K={K_block_t}: expected {expected[0]}, got {result.block_h}"
    )
    assert result.block_w == expected[1], (
        f"block_w mismatch for Mt={Mt}, Nt={Nt}, cores=({num_cores_y},{num_cores_x}), "
        f"K={K_block_t}: expected {expected[1]}, got {result.block_w}"
    )
    assert result.subblock_h == expected[2], (
        f"subblock_h mismatch for Mt={Mt}, Nt={Nt}, cores=({num_cores_y},{num_cores_x}), "
        f"K={K_block_t}: expected {expected[2]}, got {result.subblock_h}"
    )
    assert result.subblock_w == expected[3], (
        f"subblock_w mismatch for Mt={Mt}, Nt={Nt}, cores=({num_cores_y},{num_cores_x}), "
        f"K={K_block_t}: expected {expected[3]}, got {result.subblock_w}"
    )

    # Verify that the configuration can tile the output matrix
    if result.block_h > 0 and result.block_w > 0:
        assert (
            Mt % result.block_h == 0
        ), f"block_h={result.block_h} does not evenly divide Mt={Mt}"
        assert (
            Nt % result.block_w == 0
        ), f"block_w={result.block_w} does not evenly divide Nt={Nt}"

        # Verify subblock constraints
        assert (
            result.block_h % result.subblock_h == 0
        ), f"subblock_h={result.subblock_h} does not evenly divide block_h={result.block_h}"
        assert (
            result.block_w % result.subblock_w == 0
        ), f"subblock_w={result.subblock_w} does not evenly divide block_w={result.block_w}"

        # Verify core grid constraints
        cores_needed_y = Mt // result.block_h
        cores_needed_x = Nt // result.block_w
        assert (
            cores_needed_y <= num_cores_y
        ), f"Need {cores_needed_y} cores in Y but only {num_cores_y} available"
        assert (
            cores_needed_x <= num_cores_x
        ), f"Need {cores_needed_x} cores in X but only {num_cores_x} available"
