# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Flash Attention - Incremental testing of operation combinations
"""

from ttlang.d2m_api import *
import torch
import os

os.environ["SYSTEM_DESC_PATH"] = "/Users/zcarver/Downloads/system_desc.ttsys"


def test_pattern_1():
    """matmul + rowmax"""
    @pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space="L1", tiled=True)
    def kernel(Q, K, out, block_factors=None, grid=None):
        Q_stream, K_stream = Stream(Q), Stream(K)
        @compute()
        async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
            Q_block = Q_cb.pop()
            K_block = K_cb.pop()
            out_block = out_cb.reserve()

            K_T = K_block.transpose()
            S = Q_block @ K_T
            m = S.rowmax()

            out_block.store(m)
            out_cb.pop()

        @datamovement()
        async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
            idx = core_index(0) * 1 + core_index(1)
            dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
            dma(K_stream[idx, 0], K_cb.reserve()).wait()

        return Program(comp, dm)(Q, K, out)

    Q, K, out = torch.randn(32, 32), torch.randn(32, 32), torch.zeros(32, 32)
    kernel(Q, K, out)


def test_pattern_2():
    """matmul + subtract + exp + rowsum"""
    @pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space="L1", tiled=True)
    def kernel(Q, K, out, block_factors=None, grid=None):
        Q_stream, K_stream = Stream(Q), Stream(K)
        @compute()
        async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
            Q_block = Q_cb.pop()
            K_block = K_cb.pop()
            out_block = out_cb.reserve()

            K_T = K_block.transpose()
            S = Q_block @ K_T
            S_stable = S - Q_block  # Approximates S - rowmax(S)
            P = S_stable.exp()
            l = P.rowsum()

            out_block.store(l)
            out_cb.pop()

        @datamovement()
        async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
            idx = core_index(0) * 1 + core_index(1)
            dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
            dma(K_stream[idx, 0], K_cb.reserve()).wait()

        return Program(comp, dm)(Q, K, out)

    Q, K, out = torch.randn(32, 32), torch.randn(32, 32), torch.zeros(32, 32)
    kernel(Q, K, out)


def test_pattern_3():
    """Full softmax approximation: matmul + rowmax + subtract + exp + rowsum"""
    @pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space="L1", tiled=True)
    def kernel(Q, K, out, block_factors=None, grid=None):
        Q_stream, K_stream = Stream(Q), Stream(K)
        @compute()
        async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
            Q_block = Q_cb.pop()
            K_block = K_cb.pop()
            out_block = out_cb.reserve()

            K_T = K_block.transpose()
            S = Q_block @ K_T
            m = S.rowmax()
            S_stable = S - m
            P = S_stable.exp()
            l = P.rowsum()

            out_block.store(l)
            out_cb.pop()

        @datamovement()
        async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
            idx = core_index(0) * 1 + core_index(1)
            dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
            dma(K_stream[idx, 0], K_cb.reserve()).wait()

        return Program(comp, dm)(Q, K, out)

    Q, K, out = torch.randn(32, 32), torch.randn(32, 32), torch.zeros(32, 32)
    kernel(Q, K, out)


if __name__ == "__main__":
    print("=== Pattern 1: matmul + rowmax ===")
    try:
        test_pattern_1()
        print("✓ Pattern 1 works!\n")
    except Exception as e:
        print(f"✗ Pattern 1 failed: {e}\n")

    print("=== Pattern 2: matmul + subtract + exp + rowsum ===")
    try:
        test_pattern_2()
        print("✓ Pattern 2 works!\n")
    except Exception as e:
        print(f"✗ Pattern 2 failed: {e}\n")

    print("=== Pattern 3: matmul + rowmax + subtract + exp + rowsum ===")
    try:
        test_pattern_3()
        print("✓ Pattern 3 works!\n")
    except Exception as e:
        print(f"✗ Pattern 3 failed: {e}\n")
        import traceback
        traceback.print_exc()
