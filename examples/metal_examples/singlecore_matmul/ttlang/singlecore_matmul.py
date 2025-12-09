# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest
import torch

from ttl import Program, make_circular_buffer_like, copy
from metal_examples.utils import assert_with_ulp


@ttl.kernel(grid=(1, 1))
def tt_lang_singlecore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    blk_size = ttnn.TILE_SIZE
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // blk_size
    Kt = K // blk_size
    Nt = N // blk_size
    buffering_factor = 2
    a_cb = make_circular_buffer_like(
        a, shape=(blk_size, blk_size), buffer_factor=buffering_factor
    )
    b_cb = make_circular_buffer_like(
        b, shape=(blk_size, blk_size), buffer_factor=buffering_factor
    )
    out_cb = make_circular_buffer_like(
        out, shape=(blk_size, blk_size), buffer_factor=buffering_factor
    )

    @ttl.compute()
    def mm_compute():
        for _ in range(Mt):
            for _ in range(Nt):
                with out_cb.reserve() as out_blk:
                    for _ in range(Kt):
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                            out_blk += a_blk @ b_blk

    @ttl.datamovement()
    def mm_reader():
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                        a_wr = copy(a[m : (m + 1), k : (k + 1)], a_blk)
                        b_wr = copy(b[k : (k + 1), n : (n + 1)], b_blk)
                        a_wr.wait()
                        b_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        for m in range(Mt):
            for n in range(Nt):
                with out_cb.wait() as out_blk:
                    out_wr = copy(out_blk, out[m : (m + 1), n : (n + 1)])
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
