# Test double matmul (Q @ K) @ V pattern
from ttlang.d2m_api import *
import torch
import os

os.environ["TTLANG_INITIAL_MLIR"] = "/tmp/double_matmul_initial.mlir"
os.environ["TTLANG_FINAL_MLIR"] = "/tmp/double_matmul_final.mlir"
os.environ["SYSTEM_DESC_PATH"] = "/Users/zcarver/Downloads/system_desc.ttsys"

@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)],
    memory_space="L1",
    tiled=True,
)
def test_double_matmul(Q, K, V, out, block_factors=None, grid=None):
    """Test two matmuls: (Q @ K) @ V"""
    Q_stream = Stream(Q)
    K_stream = Stream(K)
    V_stream = Stream(V)

    @compute()
    async def comp(Q_cb: CircularBuffer, K_cb: CircularBuffer,
                   V_cb: CircularBuffer, out_cb: CircularBuffer):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        V_block = V_cb.pop()
        out_block = out_cb.reserve()

        # Two matmuls - temp alloc elimination should handle this
        S = Q_block @ K_block    # First matmul
        O = S @ V_block           # Second matmul

        out_block.store(O)
        out_cb.pop()

    @datamovement()
    async def dm(Q_cb: CircularBuffer, K_cb: CircularBuffer,
                 V_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) + core_index(1)
        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()
        dma(V_stream[idx, 0], V_cb.reserve()).wait()

    return Program(comp, dm)(Q, K, V, out)


print("Testing double matmul with temp alloc elimination...")
Q = torch.randn(32, 32)
K = torch.randn(32, 32)
V = torch.randn(32, 32)
out = torch.zeros(32, 32)

try:
    test_double_matmul(Q, K, V, out)
    print("✓ Double matmul WORKS!")
    print("Check /tmp/double_matmul_final.mlir for generated code")
except Exception as e:
    print(f"✗ Double matmul failed: {e}")
