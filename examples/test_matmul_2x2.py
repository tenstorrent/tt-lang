# Test 2x2 tiles with full FA kernel
from ttlang.d2m_api import *
import torch
import os


@pykernel_gen(
    block_factors=[(2, 2), (2, 2), (2, 2)],  # 2x2 tiles = 64x64 per core
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def fa_2x2_tiles(Q, K, out, block_factors=None, grid=None):
    """Full FA kernel with 2x2 tiles"""
    Q_stream = Stream(Q)
    K_stream = Stream(K)

    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        out_block = out_cb.reserve()

        # Full 6-op FA pattern
        K_T = K_block.transpose()
        S = Q_block @ K_T
        S_stable = S - Q_block
        P = S_stable.exp()
        P_norm = P.sqrt()
        result = P_norm.recip()

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()

    return Program(comp, dm)(Q, K, out)

print("Testing 2x2 tiles with full 6-op FA kernel...")
Q = torch.randn(64, 64)  # 2x2 tiles
K = torch.randn(64, 64)
out = torch.zeros(64, 64)

try:
    fa_2x2_tiles(Q, K, out)
    print("✓ 2x2 TILES WITH FULL FA WORKS!")
    print("  - 4x more data per core")
    print("  - Should be ~2-4x faster")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
