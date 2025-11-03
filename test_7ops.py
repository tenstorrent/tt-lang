# Test 7 operations to find crash point
from ttlang.d2m_api import *
import torch
import os

os.environ["SYSTEM_DESC_PATH"] = "/Users/zcarver/Downloads/system_desc.ttsys"

@pykernel_gen(grid=(1,1), block_factors=[(1,1),(1,1),(1,1)], memory_space="L1", tiled=True)
def test_7ops(Q, K, out, block_factors=None, grid=None):
    Q_stream = Stream(Q)
    K_stream = Stream(K)

    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        out_block = out_cb.reserve()

        # Copy EXACT FA pattern plus one more op
        K_T = K_block.transpose()
        S = Q_block @ K_T
        S_stable = S - Q_block
        P = S_stable.exp()
        P_norm = P.sqrt()
        almost = P_norm.recip()     # 6th op (FA stops here)
        result = almost.exp()       # 7th op - NEW

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()

    return Program(comp, dm)(Q, K, out)

print("Testing 7 ops...")
Q = torch.randn(32, 32)
K = torch.randn(32, 32)
out = torch.zeros(32, 32)

try:
    test_7ops(Q, K, out)
    print("✓ 7 ops compiled!")
except Exception as e:
    print(f"✗ 7 ops failed: {e}")
    import traceback
    traceback.print_exc()
