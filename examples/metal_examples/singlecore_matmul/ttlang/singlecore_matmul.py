# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
import ttnn
import pytest
import torch

import ttl
from ttl import Program, make_circular_buffer_like, copy, matmul

from utils.correctness import assert_with_ulp


@ttl.kernel(grid=(1, 1))
def tt_lang_singlecore_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
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
                with out_cb.reserve() as o:
                    for _ in range(Kt):
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                            result = matmul(a_blk, b_blk, o)
                            o.store(result)

    @ttl.datamovement()
    def mm_reader():
        for _ in range(Mt):
            for _ in range(Nt):
                for _ in range(Kt):
                    # TODO: Use [m, k] and [k, n] indices when supported
                    with a_cb.reserve(), b_cb.reserve():
                        tx_a = copy(a[0, 0], a_cb)
                        tx_a.wait()
                        tx_b = copy(b[0, 0], b_cb)
                        tx_b.wait()

    @ttl.datamovement()
    def mm_writer():
        for _ in range(Mt):
            for _ in range(Nt):
                # TODO: Use [m, n] index when supported
                with out_cb.wait():
                    tx = copy(out_cb, out[0, 0])
                    tx.wait()

    return Program(mm_compute, mm_reader, mm_writer)(a, b, out)

def test_singlecore_matmul_tt_lang():
    """Test singlecore matmul kernel."""
    device = ttnn.open_device(device_id=0)

    try:
        M, K, N = 32, 32, 32

        # Create torch tensors first
        a_torch = torch.rand((M, K), dtype=torch.bfloat16)
        b_torch = torch.rand((K, N), dtype=torch.bfloat16)
        c_torch = torch.zeros((M, N), dtype=torch.bfloat16)

        # Convert to ttnn tensors with proper device and memory config
        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        c = ttnn.from_torch(
            c_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
        assert a.shape[0] == c.shape[0], "Output matrix has incorrect number of rows."

        tt_lang_singlecore_matmul(a, b, c)

        golden = torch.matmul(a_torch, b_torch)
        result = ttnn.to_torch(c).to(torch.bfloat16)
        print(result)
        assert_with_ulp(golden, result)

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    test_singlecore_matmul_tt_lang()
