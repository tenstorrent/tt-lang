# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import itertools
import math
import sympy
from collections import namedtuple
from typing import List, Tuple


def remove_leading_ones(grid: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(itertools.dropwhile(lambda x: x == 1, grid))


def get_number_of_cores(grid: Tuple[int, ...]) -> int:
    core_count = 1
    for dim in grid:
        assert dim > 0, "grid dimensions must be positive"
        core_count *= dim
    return core_count


def filter_factor_pairs_by_2d_grid(
    factor_pairs: list[Tuple[int, int]], grid: Tuple[int, int]
) -> list[Tuple[int, int]]:
    valid_pairs = []
    for pair in factor_pairs:
        if pair[0] <= grid[0] and pair[1] <= grid[1]:
            valid_pairs.append(pair)
        elif pair[1] <= grid[0] and pair[0] <= grid[1]:
            valid_pairs.append((pair[1], pair[0]))
    return valid_pairs


def num_cores_to_grid_ranges(
    start_coord: Tuple[int, ...],
    target_num_cores: int,
    grid_size: Tuple[int, ...],
    row_wise: bool = True,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Generate a list of grid ranges covering target_num_cores cores starting from start_coord.
    Similar to num_cores_to_corerangeset but returns simple tuples.

    Args:
        start_coord: The starting coordinate
        target_num_cores: Number of cores to include
        grid_size: The dimensions of the grid
        row_wise: If True, fill row-wise, else column-wise

    Returns:
        List of (start_coord, end_coord) tuples representing rectangular ranges
    """
    assert len(start_coord) == len(
        grid_size
    ), "start_coord and grid_size must have same dimensions"

    # Only support 2D grids for now
    simplified_grid = remove_leading_ones(grid_size)
    assert len(simplified_grid) <= 2, "Only supports 2D grids"

    # Get the actual 2D dimensions (last 2 dimensions)
    num_cores_x = grid_size[-1] if len(grid_size) >= 1 else 1
    num_cores_y = grid_size[-2] if len(grid_size) >= 2 else 1

    start_x = start_coord[-1] if len(start_coord) >= 1 else 0
    start_y = start_coord[-2] if len(start_coord) >= 2 else 0

    assert (
        start_x < num_cores_x and start_y < num_cores_y
    ), "Start coord must be within grid"

    if row_wise:
        # Calculate available cores
        total_available_cores = (num_cores_y - 1 - start_y) * num_cores_x
        total_available_cores += num_cores_x - start_x
    else:
        # Column-wise
        total_available_cores = (num_cores_x - 1 - start_x) * num_cores_y
        total_available_cores += num_cores_y - start_y

    assert (
        target_num_cores <= total_available_cores
    ), f"Target {target_num_cores} exceeds available {total_available_cores}"

    # Build list of ranges
    all_ranges = []
    leftover_size = target_num_cores
    s_x, s_y = start_x, start_y

    prefix = tuple(0 for _ in range(len(grid_size) - 2))

    if row_wise:
        # Partial row at start
        if s_x != 0 and leftover_size > num_cores_x - start_x:
            start_c = prefix + (s_y, s_x)
            end_c = prefix + (s_y, num_cores_x - 1)
            all_ranges.append((start_c, end_c))
            cores_taken = num_cores_x - s_x
            leftover_size -= cores_taken
            s_x = 0
            s_y += 1

        # Full rows
        if leftover_size >= num_cores_x:
            num_full_rows = leftover_size // num_cores_x
            start_c = prefix + (s_y, s_x)
            end_c = prefix + (s_y + num_full_rows - 1, num_cores_x - 1)
            all_ranges.append((start_c, end_c))
            leftover_size -= num_full_rows * num_cores_x
            s_y += num_full_rows
            s_x = 0

        # Partial row at end
        if leftover_size > 0:
            start_c = prefix + (s_y, s_x)
            end_c = prefix + (s_y, s_x + leftover_size - 1)
            all_ranges.append((start_c, end_c))
    else:
        # Column-wise
        # Partial col at start
        if s_y != 0 and leftover_size > num_cores_y - start_y:
            start_c = prefix + (s_y, s_x)
            end_c = prefix + (num_cores_y - 1, s_x)
            all_ranges.append((start_c, end_c))
            cores_taken = num_cores_y - s_y
            leftover_size -= cores_taken
            s_y = 0
            s_x += 1

        # Full cols
        if leftover_size >= num_cores_y:
            num_full_cols = leftover_size // num_cores_y
            start_c = prefix + (s_y, s_x)
            end_c = prefix + (num_cores_y - 1, s_x + num_full_cols - 1)
            all_ranges.append((start_c, end_c))
            leftover_size -= num_full_cols * num_cores_y
            s_x += num_full_cols
            s_y = 0

        # Partial col at end
        if leftover_size > 0:
            start_c = prefix + (s_y, s_x)
            end_c = prefix + (s_y + leftover_size - 1, s_x)
            all_ranges.append((start_c, end_c))

    return all_ranges


def split_work_to_cores(
    grid_size: Tuple[int, ...], units_to_divide: int, row_wise: bool = True
) -> Tuple[
    int,
    Tuple[Tuple[int, ...], Tuple[int, ...]],
    Tuple[Tuple[int, ...], Tuple[int, ...]],
    int,
    int,
]:
    """Splits work units among cores in a from a single device grid.
    currently can produce work splits that cannot map to CoreRanges directly, particlarily in 1-d grids

    Args:
        grid_size: A tuple representing the dimensions of the core grid.
        units_to_divide: The total number of work units to be divided among the cores.
        row_wise: If True, split work in a row-wise manner; otherwise, column-wise.

    Returns: A tuple containing:
            - total number of cores
            - core group 1 as a tuple of tuples, start coord to end coord rectangle [inclusive, inclusive]
            - core group 2 as a tuple of tuples, start coord to end coord rectangle [inclusive, inclusive]
            - work units per core in group 1
            - work units per core in group 2
    """
    if units_to_divide == 0:
        return (0, (), (), 0, 0)
    simplified_grid_size = remove_leading_ones(grid_size)
    assert len(simplified_grid_size) <= 2, "only supports grids with a single device"
    total_cores = get_number_of_cores(grid_size)
    assert total_cores > 0, "grid must have at least one core"
    start_coord = (0,) * len(grid_size)
    if (
        total_cores >= units_to_divide
    ):  # more cores than work units, assign 1 unit to first N cores
        if len(simplified_grid_size) == 1:
            end_coord = ((0,) * (len(grid_size) - 1)) + (units_to_divide - 1,)
        elif len(simplified_grid_size) == 2:
            ranges = num_cores_to_grid_ranges(
                start_coord, units_to_divide, grid_size, row_wise
            )
            end_coord = ((0,) * (len(grid_size) - 2)) + ranges[-1][
                1
            ]  # Last range's end coordinate
        return (units_to_divide, (start_coord, end_coord), (), 1, 0)
    else:
        # more work units than cores, divide work as evenly as possible
        if len(simplified_grid_size) == 1:
            work_per_core = units_to_divide // total_cores
            remaining_work = units_to_divide % total_cores
            end_coord_1 = ((0,) * (len(grid_size) - 1)) + (remaining_work,)
            start_coord_2 = ((0,) * (len(grid_size) - 1)) + (remaining_work + 1,)
            end_coord_2 = ((0,) * (len(grid_size) - 1)) + (total_cores - 1,)
            return (
                total_cores,
                ((0,) * len(grid_size), end_coord_1),
                (start_coord_2, end_coord_2),
                work_per_core + 1,
                work_per_core,
            )

        elif len(simplified_grid_size) == 2:
            """
            For 2D grids with more work than cores:
            - Use all available cores
            - Distribute work as evenly as possible
            - Group 1: cores that get (work_per_core + 1) units
            - Group 2: cores that get work_per_core units
            """
            work_per_core = units_to_divide // total_cores
            num_cores_with_more_work = units_to_divide % total_cores

            # Evenly divided - all cores get same amount
            if num_cores_with_more_work == 0:
                num_cores_x = grid_size[-1]
                num_cores_y = grid_size[-2]
                prefix = (0,) * (len(grid_size) - 2)
                end_coord = prefix + (num_cores_y - 1, num_cores_x - 1)
                return (total_cores, (start_coord, end_coord), (), work_per_core, 0)

            # Uneven division - need two groups
            else:
                # Group 1: first num_cores_with_more_work cores get (work_per_core + 1)
                group1_ranges = num_cores_to_grid_ranges(
                    (0,) * len(grid_size), num_cores_with_more_work, grid_size, row_wise
                )

                # Find the last core of group 1 to determine where group 2 starts
                last_range_group1 = group1_ranges[-1]
                last_coord_group1 = last_range_group1[1]  # end coord of last range

                # Calculate starting position for group 2
                num_cores_x = grid_size[-1]
                num_cores_y = grid_size[-2]
                last_x = last_coord_group1[-1]
                last_y = last_coord_group1[-2]

                if row_wise:
                    # Start in the same row if possible
                    if last_x != num_cores_x - 1:
                        start_x_group2 = last_x + 1
                        start_y_group2 = last_y
                    # Otherwise start in the next row
                    else:
                        start_x_group2 = 0
                        start_y_group2 = last_y + 1
                else:
                    # Column-wise: Start in the same column if possible
                    if last_y != num_cores_y - 1:
                        start_x_group2 = last_x
                        start_y_group2 = last_y + 1
                    # Otherwise start in the next column
                    else:
                        start_x_group2 = last_x + 1
                        start_y_group2 = 0

                prefix = (0,) * (len(grid_size) - 2)
                start_coord_group2 = prefix + (start_y_group2, start_x_group2)

                num_cores_group2 = total_cores - num_cores_with_more_work
                group2_ranges = num_cores_to_grid_ranges(
                    start_coord_group2, num_cores_group2, grid_size, row_wise
                )

                # For simplified return, we'll return the bounding boxes
                # Group 1: from (0,0,...) to last coord of group 1
                group1_bbox = (start_coord, last_coord_group1)

                # Group 2: from start to last coord of group 2
                last_coord_group2 = group2_ranges[-1][1]
                group2_bbox = (start_coord_group2, last_coord_group2)

                return (
                    total_cores,
                    group1_bbox,
                    group2_bbox,
                    work_per_core + 1,
                    work_per_core,
                )


def get_prime_factors_with_multiplicity(n: int) -> List[int]:
    factors_with_multiplicity = []
    factor_dict = sympy.factorint(n)
    # Extend the list for each factor according to its count
    for prime, count in factor_dict.items():
        factors_with_multiplicity.extend([prime] * count)
    # Sort the list for a standard representation
    return sorted(factors_with_multiplicity)


def get_possible_products(factors: List[int]) -> List[int]:
    """
    Generate all possible products from a list of factors.
    This function computes all unique products that can be formed by taking
    subsets of the factors (including taking factors multiple times if they
    appear multiple times in the input).
    Args:
        factors: A list of prime factors
    Returns:
        A sorted list of all unique products that can be formed from the factors
    """
    unique_products = {1}
    for p in factors:
        new_products = {prod * p for prod in unique_products}
        unique_products.update(new_products)
    sorted_products = sorted(list(unique_products))
    return sorted_products


def get_maximum_block_dim(block_dim: int, in0_block_w: int) -> int:
    """
    Compute maximum size for the other block dimension given circular buffer (CB) capacity constraints.

    Args:
        block_dim: M or N dimension block size in tiles
        in0_block_w: K dimension block width in tiles

    Returns:
        Maximum value for the other block dimension that fits in L1 memory.
        Returns 0 if constraints cannot be satisfied.

    Notes:
        Uses a heuristic of 400 tiles for CB budget in L1 memory (double-buffered A and  B + C).
        Constraint: DB*in0_block_w*block_dim + DB*in0_block_w*other_dim + block_dim*other_dim ≤ 400
    """
    other_dim = (400 - 2 * in0_block_w * block_dim) // (2 * in0_block_w + block_dim)

    if other_dim > 0:
        return other_dim

    return 0


# Subblock choices in priority order
SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
]


# Named tuple for matmul parameters
MatmulParams = namedtuple(
    "MatmulParams", ["block_h", "block_w", "subblock_h", "subblock_w"]
)


def get_large_matmul_params(
    Mt: int, Nt: int, num_cores_y: int, num_cores_x: int, in0_block_w: int
) -> MatmulParams:
    """
    Compute optimal matrix multiplication parameters for multi-core execution.
    This function determines the per-core block sizes (Mpc, Npc) and subblock
    dimensions (subblock_h, subblock_w) for distributing a matrix multiplication
    across multiple cores while respecting memory and compute constraints.
    Args:
        Mt: Total number of tiles in M dimension (output rows)
        Nt: Total number of tiles in N dimension (output columns)
        num_cores_y: Number of available cores in Y dimension
        num_cores_x: Number of available cores in X dimension
        in0_block_w: Inner dimension block width (K dimension in tiles)
    Returns:
        MatmulParams named tuple with fields:
        - block_h: Number of M tiles per core
        - block_w: Number of N tiles per core
        - subblock_h: Subblock height for compute kernel
        - subblock_w: Subblock width for compute kernel
        Returns MatmulParams(0, 0, 0, 0) if no valid configuration is found.
    """
    # Get prime factorizations
    Nt_fac = get_prime_factors_with_multiplicity(Nt)
    Mt_fac = get_prime_factors_with_multiplicity(Mt)

    Npc_min = 1
    Mpc_min = 1

    # Remove factors larger than available cores from Nt_fac
    # These must be handled per-core (Npc_min)
    large_factors_N = [f for f in Nt_fac if f > num_cores_x]
    Nt_fac = [f for f in Nt_fac if f <= num_cores_x]
    Npc_min = math.prod(large_factors_N) if large_factors_N else 1

    # Remove factors larger than available cores from Mt_fac
    # These must be handled per-core (Mpc_min)
    large_factors_M = [f for f in Mt_fac if f > num_cores_y]
    Mt_fac = [f for f in Mt_fac if f <= num_cores_y]
    Mpc_min = math.prod(large_factors_M) if large_factors_M else 1

    # Check if minimum Npc violates memory constraints
    if Npc_min > get_maximum_block_dim(Mpc_min, in0_block_w):
        return MatmulParams(0, 0, 0, 0)

    Mpc = Mpc_min
    Npc = Npc_min

    # Case 1: Mpc_min > 1 (M dimension has large prime factors)
    if Mpc_min > 1:
        Npc_choices = get_possible_products(Nt_fac)
        Npc_max = get_maximum_block_dim(Mpc_min, in0_block_w)

        # Maximize Npc within memory constraints
        for ele in Npc_choices:
            if ele * Npc_min <= Npc_max:
                Npc = ele * Npc_min
            else:
                break

        # Check if this fits within the core grid
        if Mt // Mpc > num_cores_y or Nt // Npc > num_cores_x:
            return MatmulParams(0, 0, 0, 0)

        # Find compatible subblock dimensions
        for subblock_h, subblock_w in SUBBLOCK_HW_CHOICES:
            if Mpc % subblock_h == 0 and Npc % subblock_w == 0:
                return MatmulParams(Mpc, Npc, subblock_h, subblock_w)

    # Case 2: Npc_min > 1 (N dimension has large prime factors)
    elif Npc_min > 1:
        Mpc_choices = get_possible_products(Mt_fac)
        Mpc_max = get_maximum_block_dim(Npc_min, in0_block_w)

        # Maximize Mpc within memory constraints
        for ele in Mpc_choices:
            if ele * Mpc_min <= Mpc_max:
                Mpc = ele * Mpc_min
            else:
                break

        # Check if this fits within the core grid
        if Mt // Mpc > num_cores_y or Nt // Npc > num_cores_x:
            return MatmulParams(0, 0, 0, 0)

        # Find compatible subblock dimensions
        for subblock_h, subblock_w in SUBBLOCK_HW_CHOICES:
            if Mpc % subblock_h == 0 and Npc % subblock_w == 0:
                return MatmulParams(Mpc, Npc, subblock_h, subblock_w)

    # Case 3: Both Mpc_min == 1 and Npc_min == 1
    else:
        Mpc_choices = get_possible_products(Mt_fac)
        Npc_choices = get_possible_products(Nt_fac)

        # Try different Npc values and find the largest compatible Mpc
        for Npc in Npc_choices:
            Mpc_max = get_maximum_block_dim(Npc, in0_block_w)

            # Find largest Mpc that fits in memory
            for ele in Mpc_choices:
                if ele <= Mpc_max:
                    Mpc = ele

            # Check if this configuration fits within the core grid
            if Mt // Mpc > num_cores_y or Nt // Npc > num_cores_x:
                continue

            # Find compatible subblock dimensions
            for subblock_h, subblock_w in SUBBLOCK_HW_CHOICES:
                if Mpc % subblock_h == 0 and Npc % subblock_w == 0:
                    return MatmulParams(Mpc, Npc, subblock_h, subblock_w)

    # No valid configuration found
    return MatmulParams(0, 0, 0, 0)
