from ttlang.d2m_api import *
import torch, os
os.environ['SYSTEM_DESC_PATH'] = '/Users/zcarver/Downloads/system_desc.ttsys'

# Test 1: 1×1 blocks (minimal)
print("Test 1: block_factors=(1,1), grid=(1,1), tensor=32×32")
@pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space='L1', tiled=True)
def test_1x1(Q, K, out, block_factors=None, grid=None):
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
    test_1x1(torch.randn(32, 32), torch.randn(32, 32), torch.zeros(32, 32))
    print("  ✓ WORKS: 1×1 blocks")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
print()

# Test 2: 2×2 blocks (square, moderate)
print("Test 2: block_factors=(2,2), grid=(1,1), tensor=64×64")
@pykernel_gen(block_factors=[(2,2),(2,2),(2,2)], grid=(1,1), memory_space='L1', tiled=True)
def test_2x2(Q, K, out, block_factors=None, grid=None):
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
    test_2x2(torch.randn(64, 64), torch.randn(64, 64), torch.zeros(64, 64))
    print("  ✓ WORKS: 2×2 blocks")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
print()

# Test 3: 2×1 blocks (rectangular)
print("Test 3: block_factors=(2,1), grid=(1,1), tensor=64×32")
@pykernel_gen(block_factors=[(2,1),(2,1),(2,1)], grid=(1,1), memory_space='L1', tiled=True)
def test_2x1(Q, K, out, block_factors=None, grid=None):
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
    test_2x1(torch.randn(64, 32), torch.randn(64, 32), torch.zeros(64, 32))
    print("  ✓ WORKS: 2×1 blocks")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:100]}")
print()

# Test 4: 8×8 blocks (large square)
print("Test 4: block_factors=(8,8), grid=(1,1), tensor=256×256")
@pykernel_gen(block_factors=[(8,8),(8,8),(8,8)], grid=(1,1), memory_space='L1', tiled=True)
def test_8x8(Q, K, out, block_factors=None, grid=None):
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
    test_8x8(torch.randn(256, 256), torch.randn(256, 256), torch.zeros(256, 256))
    print("  ✓ WORKS: 8×8 blocks")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:100]}")
print()

print("="*80)
print("Summary: Testing which block_factors work with rowmax")
print("="*80)
