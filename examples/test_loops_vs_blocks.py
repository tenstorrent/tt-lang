from ttlang.d2m_api import *
import torch, os
os.environ['SYSTEM_DESC_PATH'] = '/Users/zcarver/Downloads/system_desc.ttsys'

print("="*80)
print("Question: Can we use LOOPS instead of large block_factors?")
print("="*80)
print()

print("APPROACH 1: Large block_factors (8×8)")
print("  grid=(1,1), block_factors=(8,8)")
print("  Total work: 8×8 = 64 tiles")
print("  Each tile: 32×32 = 1024 elements")
print("  Total: 256×256 = 65,536 elements")
print("  Parallelism: ALL 64 tiles computed in parallel (if hardware supports)")
print()

@pykernel_gen(block_factors=[(8,8),(8,8),(8,8)], grid=(1,1), memory_space='L1', tiled=True)
def approach1_large_blocks(Q, K, out, block_factors=None, grid=None):
    Q_stream, K_stream = Stream(Q), Stream(K)
    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()  # Gets 8×8 tiles at once
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
    Q1 = torch.randn(256, 256)
    K1 = torch.randn(256, 256)
    out1 = torch.zeros(256, 256)
    approach1_large_blocks(Q1, K1, out1)
    print("  ✓ Large blocks (8×8) WORK")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
print()

print("APPROACH 2: Loops with small blocks (1×1 blocks, loop 8×8 times)")
print("  grid=(1,1), block_factors=(1,1)")
print("  Total work: Same 64 tiles")
print("  Each tile: 32×32 = 1024 elements")
print("  Total: 256×256 = 65,536 elements")
print("  Parallelism: Process 1 tile at a time sequentially")
print()

@pykernel_gen(block_factors=[(1,1),(1,1),(1,1)], grid=(1,1), memory_space='L1', tiled=True)
def approach2_loops(Q, K, out, block_factors=None, grid=None):
    Q_stream, K_stream = Stream(Q), Stream(K)
    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        # NOTE: Loops in compute may have issues (from docs)
        # This would process tiles sequentially
        for i in range(8):
            for j in range(8):
                Q_block = Q_cb.pop()  # Get 1 tile
                K_block = K_cb.pop()
                out_block = out_cb.reserve()
                K_T = K_block.transpose()
                S = Q_block @ K_T
                m = S.rowmax()
                out_block.store(m)
                out_cb.pop()
    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer, out_cb: CircularBuffer):
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
                dma(K_stream[idx, 0], K_cb.reserve()).wait()
    return Program(comp, dm)(Q, K, out)

try:
    Q2 = torch.randn(256, 256)
    K2 = torch.randn(256, 256)
    out2 = torch.zeros(256, 256)
    approach2_loops(Q2, K2, out2)
    print("  ✓ Loops (1×1 with 8×8 loop) WORK")
except Exception as e:
    print(f"  ✗ FAILED: {str(e)[:150]}")
print()

print("="*80)
print("ANALYSIS: Work Comparison")
print("="*80)
print()
print("Square blocks (8×8):")
print("  - Total tiles: 8×8 = 64")
print("  - Each tile: 32×32 = 1,024 FLOPs for matmul")
print("  - Parallelism: ALL tiles in parallel")
print("  - Hardware bottleneck: DST register slots (16 max)")
print()
print("Loops (1×1 × 64 iterations):")
print("  - Total tiles: 64 (same)")
print("  - Each tile: 1,024 FLOPs (same)")
print("  - Parallelism: Sequential (1 tile at a time)")
print("  - Pro: Uses fewer DST slots (1 instead of 64)")
print("  - Con: NO parallelism within core")
print()
print("For 512×128 from benchmark:")
print("  Option A: grid=(8,4), block=(4,1) = 32 cores × 4 tiles = 128 tiles total")
print("           16 seq tiles × 4 head tiles = same 128 tiles")
print("           Each core does 4 tiles in parallel")
print()
print("  Option B: grid=(8,4), block=(1,1) + loop 4×1 times")
print("           32 cores × 4 iterations = same 128 tiles")
print("           Each core does 4 tiles SEQUENTIALLY")
print("           4× slower per core, but easier on DST")
print()
print("Verdict:")
print("  - Large blocks = FASTER (parallelism within core)")
print("  - Square blocks with unneeded dims = WASTEFUL")
print("    512×128 needs 16×4 tiles, NOT 16×16")
print("    Using (4,4) block = 256 tiles, but only 128 needed!")
print("    That's 2× extra work!")
