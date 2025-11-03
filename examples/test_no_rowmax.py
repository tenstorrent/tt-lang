from ttlang.d2m_api import *
import torch, os
os.environ['SYSTEM_DESC_PATH'] = '/Users/zcarver/Downloads/system_desc.ttsys'

# Test WITHOUT rowmax - just matmul + exp
print("Test: block_factors=(2,2), grid=(1,1), NO rowmax")
@pykernel_gen(block_factors=[(2,2),(2,2),(2,2)], grid=(1,1), memory_space='L1', tiled=True)
def test_no_rowmax(Q, K, out, block_factors=None, grid=None):
    Q_stream, K_stream = Stream(Q), Stream(K)
    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        out_block = out_cb.reserve()
        K_T = K_block.transpose()
        S = Q_block @ K_T
        P = S.exp()  # Skip rowmax
        out_block.store(P)
        out_cb.pop()
    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        dma(Q_stream[0, 0], Q_cb.reserve()).wait()
        dma(K_stream[0, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    test_no_rowmax(torch.randn(64, 64), torch.randn(64, 64), torch.zeros(64, 64))
    print("  ✓ WORKS without rowmax")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
print()

# Test WITH rowmax on 1×1 grid
print("Test: block_factors=(1,1), grid=(1,1), WITH rowmax")
@pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space='L1', tiled=True)
def test_1x1_rowmax(Q, K, out, block_factors=None, grid=None):
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
        dma(Q_stream[0, 0], Q_cb.reserve()).wait()
        dma(K_stream[0, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    test_1x1_rowmax(torch.randn(32, 32), torch.randn(32, 32), torch.zeros(32, 32))
    print("  ✓ WORKS with 1×1 rowmax")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
print()

# Test WITH rowmax on 2×2 blocks
print("Test: block_factors=(2,2), grid=(1,1), WITH rowmax")
@pykernel_gen(block_factors=[(2,2),(2,2),(2,2)], grid=(1,1), memory_space='L1', tiled=True)
def test_2x2_rowmax(Q, K, out, block_factors=None, grid=None):
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
        dma(Q_stream[0, 0], Q_cb.reserve()).wait()
        dma(K_stream[0, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    test_2x2_rowmax(torch.randn(64, 64), torch.randn(64, 64), torch.zeros(64, 64))
    print("  ✓ WORKS with 2×2 rowmax")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
