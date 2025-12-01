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
from eltwise_mcast import eltwise_mcast

# Import validation utilities and CircularBuffer for resetting
from python.sim import assert_pcc


class TestExamples:
    """Test cases for example simulations."""

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

    def test_eltwise_mcast_example(self):
        """Test that the eltwise_mcast example runs without assertions being hit."""
        # Use parameters that match the eltwise_mcast requirements
        dim = 128
        a_in = torch.randn(dim, dim)  # type: ignore
        b_in = torch.randn(dim, dim)  # type: ignore
        c_in = torch.randn(1, 1)  # type: ignore (single tile)
        out = torch.zeros(dim, dim)  # type: ignore

        eltwise_mcast(a_in, b_in, c_in, out)
        print(out)
        golden = a_in * b_in + c_in
        assert_pcc(golden, out)
