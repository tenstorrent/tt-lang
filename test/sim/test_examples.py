# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test case for element-wise addition using the simulation framework.
Imports and tests the eltwise_add.py example.
"""

import torch

# Import the example functions
from eltwise_add import eltwise_add
from eltwise_pipe import eltwise_pipe

# Import validation utilities and CircularBuffer for resetting
from python.sim import assert_pcc


class TestExamples:
    """Test cases for example simulations."""

    # TODO: Make sure all cores are utilized in the tests and they use the correct chunks
    def test_eltwise_add_example(self):
        """Test that the eltwise_add example runs without assertions being hit."""
        # Use the same parameters as the original example
        dim = 256
        a_in = torch.randn(dim, dim)  # type: ignore
        b_in = torch.randn(dim, dim)  # type: ignore
        out = torch.zeros(dim, dim)  # type: ignore

        eltwise_add(a_in, b_in, out)

        golden = a_in + b_in
        assert_pcc(golden, out)

    def test_eltwise_pipe_example(self):
        """Test that the eltwise_pipe example runs without assertions being hit."""
        # Use parameters that match the eltwise_pipe requirements
        dim = 128
        a_in = torch.randn(dim, dim)  # type: ignore
        b_in = torch.randn(dim, dim)  # type: ignore
        c_in = torch.randn(1, 1)  # type: ignore (single tile)
        out = torch.zeros(dim, dim)  # type: ignore

        eltwise_pipe(a_in, b_in, c_in, out)
        print(out)
        golden = a_in * b_in + c_in
        assert_pcc(golden, out)
