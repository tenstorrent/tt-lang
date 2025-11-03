from ttlang.d2m_api import *
import torch, os
os.environ['SYSTEM_DESC_PATH'] = '/Users/zcarver/Downloads/system_desc.ttsys'

@pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space='L1', tiled=True)
def test_5ops(Q, K, out, block_factors=None, grid=None):
    """5 ops: transpose + matmul + rowmax + subtract + exp"""
    Q_stream, K_stream = Stream(Q), Stream(K)
    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        out_block = out_cb.reserve()

        K_T = K_block.transpose()  # 1
        S = Q_block @ K_T           # 2
        m = S.rowmax()              # 3
        S_stable = S - m            # 4
        P = S_stable.exp()          # 5

        out_block.store(P)
        out_cb.pop()

    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) * 1 + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()

    return Program(comp, dm)(Q, K, out)

test_5ops(torch.randn(32, 32), torch.randn(32, 32), torch.zeros(32, 32))
print('✓ 5 ops work!')
