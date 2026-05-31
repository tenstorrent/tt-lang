# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFB accumulation behavior under tensor recurrence strategy options."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32
N_ITERS = 3

_DTYPE_TOL = {
    torch.bfloat16: dict(rtol=5e-2, atol=1.0),
    torch.float32: dict(rtol=1e-3, atol=1e-3),
}


def _make_dfb_accumulation_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(inp, out):
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=N_ITERS)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with inp_dfb.wait() as inp_blk:
                        out_blk += inp_blk

        @ttl.datamovement()
        def reader():
            for _ in range(N_ITERS):
                with inp_dfb.reserve() as inp_blk:
                    ttl.copy(inp[0:1, 0:1], inp_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_dfb_accumulation_ignores_dst_tensor_strategy(device, dtype):
    """`--ttl-accumulation-strategy=dst` selects tensor recurrence storage.
    User-written DFB accumulation still lowers to L1 packer accumulation."""
    inp = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_ITERS * inp.float()

    inp_dev = to_dram(inp, device)
    out_dev = to_dram(out, device)

    kernel = _make_dfb_accumulation_kernel()
    kernel(inp_dev, out_dev, options="--ttl-accumulation-strategy=dst")
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])
