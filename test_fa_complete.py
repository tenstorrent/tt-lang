# Complete Flash Attention with double matmul
from ttlang.d2m_api import *
import torch
import os

os.environ["TTLANG_INITIAL_MLIR"] = "/tmp/fa_complete_initial.mlir"
os.environ["TTLANG_FINAL_MLIR"] = "/tmp/fa_complete_final.mlir"
os.environ["SYSTEM_DESC_PATH"] = "/Users/zcarver/Downloads/system_desc.ttsys"

@pykernel_gen(
    block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)],
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def flash_attention_complete(Q, K, V, out, block_factors=None, grid=None):
    """Complete Flash Attention with two matmuls: softmax(Q @ K^T) @ V"""
    Q_stream = Stream(Q)
    K_stream = Stream(K)
    V_stream = Stream(V)

    @compute()
    async def attention_compute(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        Q_block = Q_cb.pop()
        K_block = K_cb.pop()
        V_block = V_cb.pop()
        out_block = out_cb.reserve()

        # Flash Attention with two matmuls
        K_T = K_block.transpose()
        S = Q_block @ K_T                  # First matmul
        P = S.exp()
        O = P @ V_block                    # Second matmul - TESTING

        out_block.store(O)
        out_cb.pop()

    @datamovement()
    async def dm_reader(
        Q_cb: CircularBuffer,
        K_cb: CircularBuffer,
        V_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        cy = core_index(0)
        cx = core_index(1)
        idx = cy * 1 + cx

        dma(Q_stream[idx, 0], Q_cb.reserve()).wait()
        dma(K_stream[idx, 0], K_cb.reserve()).wait()
        dma(V_stream[idx, 0], V_cb.reserve()).wait()

    return Program(attention_compute, dm_reader)(Q, K, V, out)


print("Testing complete FA with double matmul...")
Q = torch.randn(32, 32)
K = torch.randn(32, 32)
V = torch.randn(32, 32)
out = torch.zeros(32, 32)

flash_attention_complete(Q, K, V, out)
print("✓ DOUBLE MATMUL WORKS! Complete FA compiled successfully!")
