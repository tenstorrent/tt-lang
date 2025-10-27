# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Templates for common kernel patterns (matmul, elementwise operations)."""

matmul_template = {
    "grid": (1, 1),
    "block_factors": [1, 1, 1],
    "indexing_maps": [
        lambda m, n, k: (m, k),
        lambda m, n, k: (k, n),
        lambda m, n, k: (m, n),
    ],
    "iterator_types": [
        "parallel",
        "parallel",
        "reduction",
    ],
}


def matmul_fused_template(args=3):
    """
    Generate a matmul template with additional fused operands.

    Args:
        args: Total number of arguments (must be >= 3)

    Returns:
        Dictionary with grid, block_factors, indexing_maps, and iterator_types
    """
    assert args >= 3
    return {
        "grid": (1, 1),
        "block_factors": [1, 1, 1],
        "indexing_maps": [
            lambda m, n, k: (m, k),
            lambda m, n, k: (k, n),
            lambda m, n, k: (m, n),
        ]
        + [lambda m, n: (m, n)] * (args - 3),
        "iterator_types": [
            "parallel",
            "parallel",
            "reduction",
        ],
    }


eltwise_template = {
    "grid": (1, 1),
    "block_factors": [1, 1],
    "indexing_maps": [
        lambda m, n: (m, n),
        lambda m, n: (m, n),
        lambda m, n: (m, n),
    ],
    "iterator_types": [
        "parallel",
        "parallel",
    ],
}


def eltwise_fused_template(args=1):
    """
    Generate an elementwise template with multiple operands.

    Args:
        args: Total number of arguments (must be >= 1)

    Returns:
        Dictionary with grid, block_factors, indexing_maps, and iterator_types
    """
    assert args >= 1
    return {
        "grid": (1, 1),
        "block_factors": [1, 1],
        "indexing_maps": [lambda m, n: (m, n)] * args,
        "iterator_types": [
            "parallel",
            "parallel",
        ],
    }


explicit_template = {
    "grid": (1, 1),
    "block_factors": None,
    "indexing_maps": None,
    "iterator_types": None,
}
