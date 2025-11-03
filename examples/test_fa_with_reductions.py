# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Flash Attention with rowmax and rowsum reductions
Testing incrementally to see what works
"""

from ttlang.d2m_api import *
import torch
import os

os.environ["SYSTEM_DESC_PATH"] = "/Users/zcarver/Downloads/system_desc.ttsys"
os.environ["TTLANG_INITIAL_MLIR"] = "/tmp/fa_reductions_initial.mlir"
os.environ["TTLANG_FINAL_MLIR"] = "/tmp/fa_reductions_final.mlir"


@pykernel_gen(
    block_factors=[(1, 1), (1, 1), (1, 1)],
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def fa_with_reductions(Q, K, out, block_factors=None, grid=None):
    """
    Test FA pattern with rowmax and rowsum.
    Incrementally adding ops to find what breaks.
    """
    Q_stream = Stream(Q)
    K_stream = Stream(K)

    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        out_block = out_cb.reserve()

        # Pattern 1: Just matmul + rowmax
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


if __name__ == "__main__":
    print("=== Test 1: matmul + rowmax ===")
    Q = torch.randn(32, 32)
    K = torch.randn(32, 32)
    out = torch.zeros(32, 32)

    try:
        fa_with_reductions(Q, K, out)
        print("✓ Pattern 1 compiled!")
    except Exception as e:
        print(f"✗ Pattern 1 failed: {e}")
        import traceback
        traceback.print_exc()
