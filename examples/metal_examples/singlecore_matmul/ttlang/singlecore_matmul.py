# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# up to tt-lang spec, not intended to compile or run currently
import sys
from pathlib import Path
import ttnn
import pytest
import torch

import ttl
from ttl import Program, make_circular_buffer_like, copy

from utils.correctness import assert_with_ulp


@ttl.kernel(grid=(1, 1))
def tt_lang_singlecore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    buffering_factor = 2
    a_cb = make_circular_buffer_like(a, shape=(1, 1), buffer_factor=buffering_factor)
    b_cb = make_circular_buffer_like(b, shape=(1, 1), buffer_factor=buffering_factor)
    out_cb = make_circular_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )

    @ttl.compute()
    def mm_compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with out_cb.reserve() as out_blk:
                    for _ in range(Kt):
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                            out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                        a_wr = copy(a[m, k], a_blk)
                        b_wr = copy(b[k, n], b_blk)
                        a_wr.wait()
                        b_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        for m in range(Mt):
            for n in range(Nt):
                with out_cb.wait() as out_blk:
                    out_wr = copy(out_blk, out[m, n])
                    out_wr.wait()

    return Program(mm_compute, mm_reader, mm_writer)(a, b, out)


def test_singlecore_matmul_tt_lang():
    """Test singlecore matmul kernel."""
    device = ttnn.open_device(device_id=0)
    M, K, N = 256, 256, 256
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_singlecore_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)

    ttnn.close_device(device)


if __name__ == "__main__":
    test_singlecore_matmul_tt_lang()
    print("Singlecore matmul tt-lang test passed.")
