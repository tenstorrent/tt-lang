# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for intermediate circular buffer pattern.

Regression test for a bug where kernels with intermediate circular buffers
(CBs not backed by input/output tensors) failed to compile with:

    static assertion failed: Index out of range
    static_assert(Idx < kernel_compile_time_args.size(), "Index out of range");
    note: the comparison reduces to '(2 < 2)'

The root cause was that compile-time args and CB descriptors were only created
for tensor-backed CBs, not intermediate CBs. The fix ensures all CBs are
included in compile-time args and CB descriptors by using the actual CB count
from cb_configs rather than the tensor argument count.
"""

import pytest
import torch
import ttl
from ttlang_test_utils import assert_allclose, to_l1

pytestmark = pytest.mark.requires_device


@ttl.kernel(grid=(1, 1))
def intermediate_cb_kernel(x, out):
    """
    Compute exp(relu(x)) using intermediate CB to break fusion.

    Uses 3 CBs:
    - x_cb (index 0): input
    - intermediate_cb (index 1): stores relu result
    - out_cb (index 2): output
    """
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    intermediate_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv:
            with intermediate_cb.reserve() as inter:
                relu_result = ttl.math.relu(xv)
                inter.store(relu_result)

        with intermediate_cb.wait() as rv:
            with out_cb.reserve() as o:
                result = ttl.math.exp(rv)
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_intermediate_cb(device):
    """Test intermediate CB pattern computes exp(relu(x)) correctly."""
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

    intermediate_cb_kernel(x, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
