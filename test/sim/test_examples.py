# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test case for element-wise addition using the simulation framework.
Imports and tests the eltwise_add.py example.
"""

import sys
from pathlib import Path

# Add examples/sim to path so we can import the moved example modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples" / "sim"))

# Import the example functions
from eltwise_add import eltwise_add
from eltwise_pipe import eltwise_pipe
from eltwise_pipe_core3 import eltwise_pipe_core3
from singlecore_matmul import tt_lang_singlecore_matmul
from multicore_matmul import tt_lang_multicore_matmul

from sim import ttnn
from sim.testing import assert_pcc


class TestExamples:
    """Test cases for example simulations."""

    # TODO: Make sure all cores are utilized in the tests and they use the correct chunks
    def test_eltwise_add_example(self):
        """Test that the eltwise_add example runs without assertions being hit."""
        # Use the same parameters as the original example
        dim = 256
        a_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
        b_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
        out = ttnn.empty((dim, dim), dtype=ttnn.float32)

        # Test default cooperative mode
        eltwise_add(a_in, b_in, out)

        # Calculate expected result using ttnn operations
        golden = a_in + b_in
        assert_pcc(golden, out)

    def test_eltwise_pipe_example(self):
        """Test that the eltwise_pipe example runs without assertions being hit."""
        # Use parameters that match the eltwise_pipe requirements
        dim = 128
        a_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
        b_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
        c_in = ttnn.rand((1, 1), dtype=ttnn.float32)
        out = ttnn.empty((dim, dim), dtype=ttnn.float32)

        # Test default cooperative mode
        eltwise_pipe(a_in, b_in, c_in, out)

        # Calculate expected result using ttnn operations
        golden = a_in * b_in + c_in
        assert_pcc(golden, out)

    def test_eltwise_pipe_core3_example(self):
        """Test that the eltwise_pipe_core3 example runs without assertions being hit."""
        # Use parameters that match the eltwise_pipe_core3 requirements
        dim = 128
        a_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
        b_in = ttnn.rand((dim, dim), dtype=ttnn.float32)
        c_in = ttnn.rand((1, 1), dtype=ttnn.float32)
        out = ttnn.empty((dim, dim), dtype=ttnn.float32)

        # Test default cooperative mode
        eltwise_pipe_core3(a_in, b_in, c_in, out)

        # Calculate expected result using ttnn operations
        golden = a_in * b_in + c_in
        assert_pcc(golden, out)

    def test_singlecore_matmul_example(self):
        """Test that the singlecore_matmul example runs without assertions being hit."""
        # Use parameters that match the singlecore_matmul requirements
        dim_m = 128
        dim_k = 256
        dim_n = 64
        a_in = ttnn.rand((dim_m, dim_k), dtype=ttnn.float32)
        b_in = ttnn.rand((dim_k, dim_n), dtype=ttnn.float32)
        out = ttnn.empty((dim_m, dim_n), dtype=ttnn.float32)

        # Test default cooperative mode
        tt_lang_singlecore_matmul(a_in, b_in, out)

        # Calculate expected result using ttnn operations
        golden = a_in @ b_in
        assert_pcc(golden, out, rtol=1e-4, atol=1e-4)

    def test_multicore_matmul_example(self):
        """Test that the multicore_matmul example runs without assertions being hit."""
        # Use parameters that are tile-divisible
        dim_m = 128
        dim_k = 256
        dim_n = 64

        a_in = ttnn.rand((dim_m, dim_k), dtype=ttnn.float32)
        b_in = ttnn.rand((dim_k, dim_n), dtype=ttnn.float32)
        out = ttnn.empty((dim_m, dim_n), dtype=ttnn.float32)

        # Test default cooperative mode
        tt_lang_multicore_matmul(a_in, b_in, out)

        # Calculate expected result using ttnn operations
        golden = a_in @ b_in
        assert_pcc(golden, out, rtol=1e-4, atol=1e-4)
