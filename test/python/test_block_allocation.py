#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test comparing split_work_to_nodes with ttnn.split_work_to_cores
get_large_matmul_params is compared with hard coded expected values
"""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.utils.block_allocation import (
    get_large_matmul_params,
    get_number_of_nodes_from_ranges,
    split_work_to_nodes,
)


def extract_coords_from_ttnn_corerangeset(core_range_set):
    """Extract all start and end coordinates from a ttnn CoreRangeSet"""
    if not core_range_set.ranges():
        return []

    coords = []
    for r in core_range_set.ranges():
        coords.append(((r.start.y, r.start.x), (r.end.y, r.end.x)))
    return coords


@pytest.mark.parametrize(
    "ranges,expected",
    [
        # Empty range list
        ([], 0),
        # Single 1D range
        ([((0,), (4,))], 5),
        # Single 2D range: full rectangle
        ([((0, 0), (3, 7))], 32),
        # Single point
        ([((2, 3), (2, 3))], 1),
        # Multiple 2D ranges: L-shape (4 full rows + partial row)
        ([((0, 0), (3, 7)), ((4, 0), (4, 3))], 36),
        # Multiple 1D ranges (disjoint)
        ([((0,), (2,)), ((4,), (6,))], 6),
        # Multiple 2D ranges: partial row + full rows + partial row
        ([((0, 4), (0, 7)), ((1, 0), (2, 7)), ((3, 0), (3, 2))], 4 + 16 + 3),
        # 3D range with leading dimension
        ([((0, 0, 0), (0, 2, 4))], 15),
    ],
)
def test_get_number_of_nodes_from_ranges(ranges, expected):
    """Test get_number_of_nodes_from_ranges with known inputs and expected counts."""
    assert get_number_of_nodes_from_ranges(ranges) == expected


@pytest.mark.parametrize(
    "grid_size_tuple,units,row_wise",
    [
        # Test cases with more work than cores
        ((8, 8), 100, True),
        ((8, 8), 100, False),
        ((8, 8), 65, True),
        ((8, 8), 65, False),
        ((8, 8), 129, True),
        # Test even distribution
        ((8, 8), 64, True),
        ((8, 8), 128, True),
        # Test with different grid sizes
        ((4, 8), 50, True),
        ((7, 9), 100, False),
        ((7, 9), 100, True),
        # Test fewer units than cores
        ((8, 8), 10, True),
        ((8, 8), 20, False),
        ((8, 8), 1, True),
        # Test edge cases
        ((8, 8), 63, True),
        ((8, 8), 127, True),
        # 2D grids that force multiple CoreRanges per group (L-shapes)
        ((13, 10), 200, True),
        ((13, 10), 200, False),
        ((5, 7), 50, True),
        ((3, 12), 40, False),
        # Small grids with multi-range groups
        ((2, 3), 10, True),
        ((3, 2), 8, False),
    ],
)
def test_split_work_to_nodes(grid_size_tuple, units, row_wise):
    """Compare results from split_work_to_nodes and ttnn.split_work_to_cores"""
    new_result = split_work_to_nodes(grid_size_tuple, units, row_wise)
    new_total, new_g1, new_g2, new_w1, new_w2 = new_result

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

    ttnn_g1_coords = extract_coords_from_ttnn_corerangeset(ttnn_g1)
    ttnn_g2_coords = extract_coords_from_ttnn_corerangeset(ttnn_g2)

    assert new_w1 == ttnn_w1, f"Work per core G1 mismatch: {new_w1} vs {ttnn_w1}"
    assert new_w2 == ttnn_w2, f"Work per core G2 mismatch: {new_w2} vs {ttnn_w2}"

    new_g1_num_cores = get_number_of_nodes_from_ranges(new_g1)
    new_g2_num_cores = get_number_of_nodes_from_ranges(new_g2)
    ttnn_g1_num_cores = sum(
        (end[1] - start[1] + 1) * (end[0] - start[0] + 1)
        for start, end in ttnn_g1_coords
    )
    ttnn_g2_num_cores = sum(
        (end[1] - start[1] + 1) * (end[0] - start[0] + 1)
        for start, end in ttnn_g2_coords
    )

    assert (
        new_g1_num_cores == ttnn_g1_num_cores
    ), f"Group 1 core count mismatch: {new_g1_num_cores} vs {ttnn_g1_num_cores}"
    assert (
        new_g2_num_cores == ttnn_g2_num_cores
    ), f"Group 2 core count mismatch: {new_g2_num_cores} vs {ttnn_g2_num_cores}"

    new_total_work = new_g1_num_cores * new_w1 + new_g2_num_cores * new_w2
    assert new_total_work == units, f"Total work mismatch: {new_total_work} vs {units}"

    assert len(new_g1) == len(
        ttnn_g1_coords
    ), f"Group 1 range count mismatch: {len(new_g1)} vs {len(ttnn_g1_coords)}"
    for i, (new_range, ttnn_range) in enumerate(zip(new_g1, ttnn_g1_coords)):
        assert (
            new_range[0] == ttnn_range[0]
        ), f"G1 range {i} start mismatch: {new_range[0]} vs {ttnn_range[0]}"
        assert (
            new_range[1] == ttnn_range[1]
        ), f"G1 range {i} end mismatch: {new_range[1]} vs {ttnn_range[1]}"

    assert len(new_g2) == len(
        ttnn_g2_coords
    ), f"Group 2 range count mismatch: {len(new_g2)} vs {len(ttnn_g2_coords)}"
    for i, (new_range, ttnn_range) in enumerate(zip(new_g2, ttnn_g2_coords)):
        assert (
            new_range[0] == ttnn_range[0]
        ), f"G2 range {i} start mismatch: {new_range[0]} vs {ttnn_range[0]}"
        assert (
            new_range[1] == ttnn_range[1]
        ), f"G2 range {i} end mismatch: {new_range[1]} vs {ttnn_range[1]}"


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
