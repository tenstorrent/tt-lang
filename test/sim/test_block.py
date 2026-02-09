# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for Block class operations."""

import pytest
import torch

from sim.block import Block
from sim.ttnnsim import Tensor


def test_matmul_1x4_times_4x1_shape():
    """Test matmul with (1,4) @ (4,1) produces (1,1) result, not (4,4).

    This is a regression test for a bug where Block.__matmul__ was using
    broadcasting logic (result_shape = (max(1,4), max(4,1)) = (4,4))
    instead of matmul logic (result_shape = (1,1)).
    """
    # Create (1,4) block - 1 row, 4 columns of tiles
    a_tensors = [Tensor(torch.ones((32, 32))) for _ in range(4)]
    a_block = Block.from_list(a_tensors, shape=(1, 4))

    # Create (4,1) block - 4 rows, 1 column of tiles
    b_tensors = [Tensor(torch.ones((32, 32))) for _ in range(4)]
    b_block = Block.from_list(b_tensors, shape=(4, 1))

    # Perform matmul
    result = a_block @ b_block

    # Result should be (1,1), not (4,4)
    assert result.shape == (1, 1), f"Expected (1,1) but got {result.shape}"
    assert (
        len(result.to_list()) == 1
    ), f"Expected 1 tile but got {len(result.to_list())}"


def test_matmul_2x3_times_3x2_shape():
    """Test matmul with (2,3) @ (3,2) produces (2,2) result."""
    # Create (2,3) block
    a_tensors = [Tensor(torch.ones((32, 32))) for _ in range(6)]
    a_block = Block.from_list(a_tensors, shape=(2, 3))

    # Create (3,2) block
    b_tensors = [Tensor(torch.ones((32, 32))) for _ in range(6)]
    b_block = Block.from_list(b_tensors, shape=(3, 2))

    # Perform matmul
    result = a_block @ b_block

    # Result should be (2,2)
    assert result.shape == (2, 2), f"Expected (2,2) but got {result.shape}"
    assert (
        len(result.to_list()) == 4
    ), f"Expected 4 tiles but got {len(result.to_list())}"


def test_matmul_1x1_times_1x4_shape():
    """Test matmul with (1,1) @ (1,4) produces (1,4) result."""
    # Create (1,1) block
    a_tensors = [Tensor(torch.ones((32, 32)))]
    a_block = Block.from_list(a_tensors, shape=(1, 1))

    # Create (1,4) block
    b_tensors = [Tensor(torch.ones((32, 32))) for _ in range(4)]
    b_block = Block.from_list(b_tensors, shape=(1, 4))

    # Perform matmul
    result = a_block @ b_block

    # Result should be (1,4)
    assert result.shape == (1, 4), f"Expected (1,4) but got {result.shape}"
    assert (
        len(result.to_list()) == 4
    ), f"Expected 4 tiles but got {len(result.to_list())}"


def test_matmul_4x1_times_1x4_shape():
    """Test matmul with (4,1) @ (1,4) produces (4,4) result."""
    # Create (4,1) block
    a_tensors = [Tensor(torch.ones((32, 32))) for _ in range(4)]
    a_block = Block.from_list(a_tensors, shape=(4, 1))

    # Create (1,4) block
    b_tensors = [Tensor(torch.ones((32, 32))) for _ in range(4)]
    b_block = Block.from_list(b_tensors, shape=(1, 4))

    # Perform matmul
    result = a_block @ b_block

    # Result should be (4,4) - this case broadcasting and matmul agree
    assert result.shape == (4, 4), f"Expected (4,4) but got {result.shape}"
    assert (
        len(result.to_list()) == 16
    ), f"Expected 16 tiles but got {len(result.to_list())}"


def test_matmul_1x4_times_4x1_values():
    """Test matmul correctness for (1,4) @ (4,1) grid."""
    # Create (1,4) block with known values
    a_tensors = [Tensor(torch.full((32, 32), float(i + 1))) for i in range(4)]
    a_block = Block.from_list(a_tensors, shape=(1, 4))

    # Create (4,1) block with known values
    b_tensors = [Tensor(torch.full((32, 32), float(i + 1))) for i in range(4)]
    b_block = Block.from_list(b_tensors, shape=(4, 1))

    # Perform matmul
    result = a_block @ b_block

    # Result should be (1,1)
    assert result.shape == (1, 1)

    # Each tile in result is sum of: tile[0,0] @ tile[0,0] + tile[0,1] @ tile[1,0] + ...
    # = (1*32 @ 1*32) + (2*32 @ 2*32) + (3*32 @ 3*32) + (4*32 @ 4*32)
    # Each matmul produces 32*(value_a * value_b) for all elements
    # So result = sum(i^2 * 32 for i in [1,2,3,4]) = 32 * (1 + 4 + 9 + 16) = 32 * 30 = 960
    result_tensor = result.to_list()[0].to_torch()
    expected_value = 32.0 * (1 * 1 + 2 * 2 + 3 * 3 + 4 * 4)  # 960
    assert torch.allclose(
        result_tensor, torch.full((32, 32), expected_value)
    ), f"Expected all values to be {expected_value}, got {result_tensor[0, 0]}"
