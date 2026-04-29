# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for issue #541: pipe matmuls must produce correct results
when the work shape (M_BLOCKS x N_BLOCKS) does not exactly fill the device's
compute grid.

Before the fix, `@ttl.operation(grid="auto")` resolved to the full device
grid; idle cores executed the kernel with out-of-bounds indices, corrupting
DRAM and the multicast handshake (PCC dropped to 0.06-0.89 across the
configurations swept here). After the fix, kernels in test_mcast_matmul.py
use a host-side callable (_matmul_grid) that returns the active subgrid
sized so the work divides evenly.

This test exercises the small-work / non-fill regime that previously failed
across all three multicast variants (mcast, balanced, balanced+relu) and
verifies that:
  1. PCC is now >= 0.99 on the previously-broken configurations.
  2. The compile-time guard in ttl_api.py rejects `grid="auto"` when the
     kernel constructs multicast pipes (negative test).
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram
from test_mcast_matmul import (
    BLOCK_SIZE,
    make_balanced_kernel,
    make_balanced_relu_kernel,
    make_mcast_kernel,
)


# Shapes chosen so that M_BLOCKS x N_BLOCKS does not exactly fill an 8x7
# (wormhole_b0) or 13x10 (blackhole) compute grid. These previously failed
# with PCC < 0.90 under grid="auto".
SHAPES = [
    pytest.param((256, 256, 256), id="256x256x256"),
    pytest.param((512, 512, 512), id="512x512x512"),
    pytest.param((1024, 1024, 1024), id="1024x1024x1024"),
    pytest.param((1792, 256, 2048), id="1792x256x2048"),
]

KERNELS = [
    pytest.param(make_mcast_kernel, id="mcast"),
    pytest.param(make_balanced_kernel, id="balanced"),
    pytest.param(make_balanced_relu_kernel, id="balanced_relu"),
]


@pytest.mark.parametrize("kernel_factory", KERNELS)
@pytest.mark.parametrize("MKN", SHAPES)
def test_mcast_matmul_subgrid_bf16(device, kernel_factory, MKN):
    """Pipe matmul produces correct results when work does not fill the device grid."""
    M, K, N = MKN
    a_torch = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_torch = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

    a_tt = to_dram(a_torch, device)
    w_tt = to_dram(w_torch, device)
    out_tt = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    kernel = kernel_factory(M, K, N)
    kernel(a_tt, w_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    if kernel_factory is make_balanced_relu_kernel:
        expected = ttnn.to_torch(ttnn.relu(ttnn.matmul(a_tt, w_tt)))
    else:
        expected = ttnn.to_torch(ttnn.matmul(a_tt, w_tt))
    assert_pcc(expected.float(), result.float(), threshold=0.99)


def test_grid_auto_with_multicast_raises(device):
    """grid='auto' combined with a multicast PipeNet must be rejected at compile time."""

    @ttl.operation(grid="auto")
    def bad_kernel(inp, out):
        NUM_COLS, NUM_ROWS = ttl.grid_size(dims=2)
        net = ttl.PipeNet(
            [ttl.Pipe(src=(0, 0), dst=(slice(0, NUM_COLS), slice(0, NUM_ROWS)))]
        )
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
                tile_out.store(ttl.math.abs(tile_in))

        @ttl.datamovement()
        def dm_read():
            with inp_cb.reserve() as blk:

                def src(pipe):
                    ttl.copy(inp[0, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

                net.if_src(src)

                def dst(pipe):
                    ttl.copy(pipe, blk).wait()

                net.if_dst(dst)

        @ttl.datamovement()
        def dm_write():
            x, y = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[y, x]).wait()

    inp = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    with pytest.raises(ValueError, match="grid='auto'"):
        bad_kernel(inp, out)
