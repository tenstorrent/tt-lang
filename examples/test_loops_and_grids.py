# Test Python loops and different grid sizes
from ttlang.d2m_api import *
import torch
import os

os.environ["SYSTEM_DESC_PATH"] = "/Users/zcarver/Downloads/system_desc.ttsys"

# Test 1: Python for loop in datamovement
@pykernel_gen(
    block_factors=[(1, 1), (1, 1)],
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def test_dm_loop(inputs, out, block_factors=None, grid=None):
    """Test for loop in datamovement thread"""
    input_stream = Stream(inputs)

    @compute()
    async def comp(input_cb: CircularBuffer, out_cb: CircularBuffer):
        # Simple pass-through to test DM loop
        data = input_cb.pop()
        out_block = out_cb.reserve()
        out_block.store(data)
        out_cb.pop()

    @datamovement()
    async def dm(input_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) + core_index(1)

        # Python for loop - load multiple blocks
        for i in range(3):
            dma(input_stream[idx, i], input_cb.reserve()).wait()

    return Program(comp, dm)(inputs, out)

# Test 2: Grid size 2x2
@pykernel_gen(
    block_factors=[(1, 1), (1, 1)],
    grid=(2, 2),  # 2x2 = 4 cores
    memory_space="L1",
    tiled=True,
)
def test_2x2_grid(A, out, block_factors=None, grid=None):
    """Test 2x2 grid"""
    A_stream = Stream(A)

    @compute()
    async def comp(A_cb: CircularBuffer, out_cb: CircularBuffer):
        A_block = A_cb.pop()
        out_block = out_cb.reserve()

        # Simple operation
        result = A_block.exp()

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(A_cb: CircularBuffer, out_cb: CircularBuffer):
        cy = core_index(0)
        cx = core_index(1)
        grid_x = 2
        idx = cy * grid_x + cx  # Linear index for 2x2 grid

        dma(A_stream[idx, 0], A_cb.reserve()).wait()

    return Program(comp, dm)(A, out)

# Test 3: Grid size 4x4
@pykernel_gen(
    block_factors=[(1, 1), (1, 1)],
    grid=(4, 4),  # 4x4 = 16 cores
    memory_space="L1",
    tiled=True,
)
def test_4x4_grid(A, out, block_factors=None, grid=None):
    """Test 4x4 grid"""
    A_stream = Stream(A)

    @compute()
    async def comp(A_cb: CircularBuffer, out_cb: CircularBuffer):
        A_block = A_cb.pop()
        out_block = out_cb.reserve()

        # Chain operations
        result = A_block.exp().sqrt()

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(A_cb: CircularBuffer, out_cb: CircularBuffer):
        cy = core_index(0)
        cx = core_index(1)
        grid_x = 4
        idx = cy * grid_x + cx  # Linear index for 4x4 grid

        dma(A_stream[idx, 0], A_cb.reserve()).wait()

    return Program(comp, dm)(A, out)


print("=== Test 1: Python for loops in datamovement ===")
inputs = torch.randn(96, 32)  # 3 blocks of 32x32
out1 = torch.zeros(32, 32)
try:
    test_dm_loop(inputs, out1)
    print("✓ Python for loops WORK in datamovement!\n")
except Exception as e:
    print(f"✗ DM loops failed: {e}\n")

print("=== Test 2: 2x2 Grid (4 cores) ===")
A_2x2 = torch.randn(64, 64)  # 2x2 grid of 32x32 tiles
out2 = torch.zeros(64, 64)
try:
    test_2x2_grid(A_2x2, out2)
    print("✓ 2x2 grid WORKS!\n")
except Exception as e:
    print(f"✗ 2x2 grid failed: {e}\n")

print("=== Test 3: 4x4 Grid (16 cores) ===")
A_4x4 = torch.randn(128, 128)  # 4x4 grid of 32x32 tiles
out3 = torch.zeros(128, 128)
try:
    test_4x4_grid(A_4x4, out3)
    print("✓ 4x4 grid WORKS!\n")
except Exception as e:
    print(f"✗ 4x4 grid failed: {e}\n")

print("=== SUMMARY ===")
print("Testing Python loops and grid scaling capabilities")
