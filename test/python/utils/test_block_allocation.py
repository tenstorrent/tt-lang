#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test comparing split_work_to_cores with ttnn.split_work_to_cores
"""
import pytest

from ttlang.utils.block_allocation import split_work_to_cores

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
