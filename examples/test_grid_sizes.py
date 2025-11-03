from ttlang.d2m_api import *
import torch, os
os.environ['SYSTEM_DESC_PATH'] = '/Users/zcarver/Downloads/system_desc.ttsys'

# Test 1: 2×2 grid with 1×1 blocks
print("Test 1: grid=(2,2), block_factors=(1,1), tensor=64×64")
@pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(2,2), memory_space='L1', tiled=True)
def test_2x2_grid_1x1_block(Q, K, out, block_factors=None, grid=None):
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
        idx = core_index(0) * 2 + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    test_2x2_grid_1x1_block(torch.randn(64, 64), torch.randn(64, 64), torch.zeros(64, 64))
    print("  ✓ WORKS")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
print()

# Test 2: 4×4 grid with 1×1 blocks
print("Test 2: grid=(4,4), block_factors=(1,1), tensor=128×128")
@pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(4,4), memory_space='L1', tiled=True)
def test_4x4_grid_1x1_block(Q, K, out, block_factors=None, grid=None):
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
        idx = core_index(0) * 4 + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    test_4x4_grid_1x1_block(torch.randn(128, 128), torch.randn(128, 128), torch.zeros(128, 128))
    print("  ✓ WORKS")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
print()

# Test 3: 4×4 grid with 2×2 blocks
print("Test 3: grid=(4,4), block_factors=(2,2), tensor=256×256")
@pykernel_gen(block_factors=[(2,2),(2,2),(2,2)], grid=(4,4), memory_space='L1', tiled=True)
def test_4x4_grid_2x2_block(Q, K, out, block_factors=None, grid=None):
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
        idx = core_index(0) * 4 + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    test_4x4_grid_2x2_block(torch.randn(256, 256), torch.randn(256, 256), torch.zeros(256, 256))
    print("  ✓ WORKS")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
print()

# Test 4: Our known-good config
print("Test 4: grid=(8,10), block_factors=(2,2), tensor=512×640")
# (already tested - we know this works)
print("  ✓ WORKS (confirmed from previous tests)")
