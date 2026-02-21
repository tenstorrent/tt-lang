# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for intermediate dataflow buffer pattern.

Regression test for a bug where kernels with intermediate circular buffers
(CBs not backed by input/output tensors) failed to compile with:

    static assertion failed: Index out of range
    static_assert(Idx < kernel_compile_time_args.size(), "Index out of range");
    note: the comparison reduces to '(2 < 2)'

The root cause was that compile-time args and DFB descriptors were only created
for tensor-backed CBs, not intermediate CBs. The fix ensures all CBs are
included in compile-time args and DFB descriptors by using the actual DFB count
from dfb_configs rather than the tensor argument count.
"""

import pytest
import torch
import ttl
from ttlang_test_utils import assert_allclose, to_l1

pytestmark = pytest.mark.requires_device


@ttl.kernel(grid=(1, 1))
def intermediate_dfb_kernel(x, out):
    """
    Compute exp(relu(x)) using intermediate DFB to break fusion.

    Uses 3 CBs:
    - x_dfb (index 0): input
    - intermediate_dfb (index 1): stores relu result
    - out_dfb (index 2): output
    """
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    intermediate_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv:
            with intermediate_dfb.reserve() as inter:
                relu_result = ttl.math.relu(xv)
                inter.store(relu_result)

        with intermediate_dfb.wait() as rv:
            with out_dfb.reserve() as o:
                result = ttl.math.exp(rv)
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_intermediate_dfb(device):
    """Test intermediate DFB pattern computes exp(relu(x)) correctly."""
    try:
        import ttnn
    except ImportError:
        pytest.skip("TTNN not available")

    x_torch = torch.tensor(
        [[-1.0, 0.0, 1.0, 2.0, 3.0] + [1.0] * 27] * 32, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    x = to_l1(x_torch, device)
    out = to_l1(out_torch, device)

    expected = torch.exp(torch.relu(x_torch))

    intermediate_dfb_kernel(x, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
