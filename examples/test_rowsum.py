# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Minimal test for rowsum operator
"""

from ttlang.d2m_api import *
import torch
import os

os.environ["TTLANG_INITIAL_MLIR"] = "/tmp/rowsum_initial.mlir"
os.environ["TTLANG_FINAL_MLIR"] = "/tmp/rowsum_final.mlir"


@pykernel_gen(
    block_factors=[(1, 1), (1, 1)],
    grid=(1, 1),
    memory_space="L1",
    tiled=True,
)
def test_rowsum(inp, out, block_factors=None, grid=None):
    """Test rowsum reduction operator."""
    inp_stream = Stream(inp)

    @compute()
    async def comp(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        inp_block = inp_cb.pop()
        out_block = out_cb.reserve()

        # Test rowsum
        result = inp_block.rowsum()

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) * 1 + core_index(1)
        dma(inp_stream[idx, 0], inp_cb.reserve()).wait()

    return Program(comp, dm)(inp, out)


if __name__ == "__main__":
    print("Testing rowsum operator...")
    inp = torch.randn(32, 32)
    out = torch.zeros(32, 32)

    try:
        test_rowsum(inp, out)
        print("✓ rowsum compiled successfully!")
    except Exception as e:
        print(f"✗ rowsum failed: {e}")
        import traceback
        traceback.print_exc()
