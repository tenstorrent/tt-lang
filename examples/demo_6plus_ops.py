# Demonstrate 6+ operation fusion works
from ttlang.d2m_api import *
import torch

@pykernel_gen(
    block_factors=[(1, 1), (1, 1), (1, 1)],
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def test_10ops(Q, K, out, block_factors=None, grid=None):
    """10-operation kernel proving NO operation limit"""
    Q_stream = Stream(Q)
    K_stream = Stream(K)

    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        out_block = out_cb.reserve()

        # 10 operations - proves no 6-op limit!
        K_T = K_block.transpose()    # 1
        S = Q_block @ K_T             # 2
        r = S - Q_block               # 3
        r = r.exp()                   # 4
        r = r.sqrt()                  # 5
        r = r.recip()                 # 6
        r = r.exp()                   # 7
        r = r.sqrt()                  # 8
        r = r.recip()                 # 9
        result = r.exp()              # 10

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()

    return Program(comp, dm)(Q, K, out)

print("Testing 10-operation kernel...")
Q = torch.randn(32, 32)
K = torch.randn(32, 32)
out = torch.zeros(32, 32)

test_10ops(Q, K, out)
print("✓ 10 OPERATIONS WORK - No operation limit!")
