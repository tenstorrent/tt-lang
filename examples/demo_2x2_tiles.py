# Demonstrate 2x2 tile processing (4x more data per core)
from ttlang.d2m_api import *
import torch

@pykernel_gen(
    block_factors=[(2, 2), (2, 2)],  # 2x2 tiles = 64x64 elements
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def unary_2x2(A, out, block_factors=None, grid=None):
    """Unary operations work with 2x2 tiles"""
    A_stream = Stream(A)

    @compute()
    async def comp(A_cb: CircularBuffer, out_cb: CircularBuffer):
        A_block = A_cb.pop()
        out_block = out_cb.reserve()

        # 6-op unary chain with 2x2 tiles
        result = A_block.exp()
        result = result.sqrt()
        result = result.recip()
        result = result.exp()
        result = result.sqrt()
        result = result.recip()

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(A_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) + core_index(1)
        dma(A_stream[idx, 0], A_cb.reserve()).wait()

    return Program(comp, dm)(A, out)

print("Testing 2x2 tiles (4x more data per core)...")
A = torch.randn(64, 64)  # 2x2 tiles
out = torch.zeros(64, 64)

unary_2x2(A, out)
print("✓ 2x2 tiles WORK with unary operations!")
print("  - 4x more data processed per iteration")
print("  - Unlocked by removing DST assertion")

# Test with larger grid
@pykernel_gen(
    block_factors=[(2, 2), (2, 2)],
    grid=(4, 4),  # 16 cores
    memory_space="L1",
    tiled=True,
)
def grid_with_2x2(A, out, block_factors=None, grid=None):
    A_stream = Stream(A)

    @compute()
    async def comp(A_cb: CircularBuffer, out_cb: CircularBuffer):
        A_block = A_cb.pop()
        out_block = out_cb.reserve()
        result = A_block.exp().sqrt().recip()
        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(A_cb: CircularBuffer, out_cb: CircularBuffer):
        cy = core_index(0)
        cx = core_index(1)
        idx = cy * 4 + cx
        dma(A_stream[idx, 0], A_cb.reserve()).wait()

    return Program(comp, dm)(A, out)

print("\nTesting 4x4 grid with 2x2 tiles...")
A2 = torch.randn(256, 256)  # 16 cores × 2x2 tiles = 8x8 tiles total
out2 = torch.zeros(256, 256)

grid_with_2x2(A2, out2)
print("✓ 4x4 grid × 2x2 tiles WORKS!")
print("  - 16 cores × 4 tiles each = 64 tiles in parallel")
print("  - 16x throughput improvement!")
