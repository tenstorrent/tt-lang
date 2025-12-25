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
from eltwise_pipe_core3 import eltwise_pipe_core3
from singlecore_matmul import tt_lang_singlecore_matmul
from multicore_matmul import tt_lang_multicore_matmul

# Import validation utilities and CircularBuffer for resetting
from python.sim import assert_pcc
from python.sim.program import ExecutionMode


class TestExamples:
    """Test cases for example simulations."""

    # TODO: Make sure all cores are utilized in the tests and they use the correct chunks
    def test_eltwise_add_example(self):
        """Test that the eltwise_add example runs without assertions being hit."""
        # Use the same parameters as the original example
        dim = 256
        a_in = torch.randn(dim, dim)
        b_in = torch.randn(dim, dim)
        out_threaded = torch.zeros(dim, dim)
        out_cooperative = torch.zeros(dim, dim)

        # Test threaded mode
        eltwise_add(a_in, b_in, out_threaded, mode=ExecutionMode.THREADED)

        # Test cooperative mode
        eltwise_add(a_in, b_in, out_cooperative, mode=ExecutionMode.COOPERATIVE)

        golden = a_in + b_in
        assert_pcc(golden, out_threaded)
        assert_pcc(golden, out_cooperative)

    def test_eltwise_pipe_example(self):
        """Test that the eltwise_pipe example runs without assertions being hit."""
        # Use parameters that match the eltwise_pipe requirements
        dim = 128
        a_in = torch.randn(dim, dim)
        b_in = torch.randn(dim, dim)
        c_in = torch.randn(1, 1)
        out_threaded = torch.zeros(dim, dim)
        out_cooperative = torch.zeros(dim, dim)

        # Test threaded mode
        eltwise_pipe(a_in, b_in, c_in, out_threaded, mode=ExecutionMode.THREADED)

        # Test cooperative mode
        eltwise_pipe(a_in, b_in, c_in, out_cooperative, mode=ExecutionMode.COOPERATIVE)

        golden = a_in * b_in + c_in
        assert_pcc(golden, out_threaded)
        assert_pcc(golden, out_cooperative)

    def test_eltwise_pipe_core3_example(self):
        """Test that the eltwise_pipe_core3 example runs without assertions being hit."""
        # Use parameters that match the eltwise_pipe_core3 requirements
        dim = 128
        a_in = torch.randn(dim, dim)
        b_in = torch.randn(dim, dim)
        c_in = torch.randn(1, 1)
        out_threaded = torch.zeros(dim, dim)
        out_cooperative = torch.zeros(dim, dim)

        # Test threaded mode
        eltwise_pipe_core3(a_in, b_in, c_in, out_threaded, mode=ExecutionMode.THREADED)

        # Test cooperative mode
        eltwise_pipe_core3(
            a_in, b_in, c_in, out_cooperative, mode=ExecutionMode.COOPERATIVE
        )

        golden = a_in * b_in + c_in
        assert_pcc(golden, out_threaded)
        assert_pcc(golden, out_cooperative)

    def test_singlecore_matmul_example(self):
        """Test that the singlecore_matmul example runs without assertions being hit."""
        # Use parameters that match the singlecore_matmul requirements
        dim_m = 128
        dim_k = 256
        dim_n = 64
        a_in = torch.randn(dim_m, dim_k)
        b_in = torch.randn(dim_k, dim_n)
        out_threaded = torch.zeros(dim_m, dim_n)
        out_cooperative = torch.zeros(dim_m, dim_n)

        # Test threaded mode
        tt_lang_singlecore_matmul(a_in, b_in, out_threaded, mode=ExecutionMode.THREADED)

        # Test cooperative mode
        tt_lang_singlecore_matmul(
            a_in, b_in, out_cooperative, mode=ExecutionMode.COOPERATIVE
        )

        golden = torch.matmul(a_in, b_in)
        assert_pcc(golden, out_threaded, rtol=1e-4, atol=1e-4)
        assert_pcc(golden, out_cooperative, rtol=1e-4, atol=1e-4)

    def test_multicore_matmul_example(self):
        """Test that the multicore_matmul example runs without assertions being hit."""
        # Use parameters that are tile-divisible
        dim_m = 128
        dim_k = 256
        dim_n = 64

        a_in = torch.randn(dim_m, dim_k)
        b_in = torch.randn(dim_k, dim_n)
        out_threaded = torch.zeros(dim_m, dim_n)
        out_cooperative = torch.zeros(dim_m, dim_n)

        # Test threaded mode
        tt_lang_multicore_matmul(a_in, b_in, out_threaded, mode=ExecutionMode.THREADED)

        # Test cooperative mode
        tt_lang_multicore_matmul(
            a_in, b_in, out_cooperative, mode=ExecutionMode.COOPERATIVE
        )

        golden = torch.matmul(a_in, b_in)
        assert_pcc(golden, out_threaded, rtol=1e-4, atol=1e-4)
        assert_pcc(golden, out_cooperative, rtol=1e-4, atol=1e-4)
