# Test transpose operation standalone
from ttlang.d2m_api import *
import torch

@pykernel_gen(grid=(1,1), block_factors=[(1,1), (1,1)])
def test_transpose(inp, out):
    inp_stream = Stream(inp)

    @compute()
    async def comp(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        inp_block = inp_cb.pop()
        out_block = out_cb.reserve()

        result = inp_block.transpose()

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0)
        shard = inp_cb.reserve()
        dma(inp_stream[idx, 0], shard).wait()

    return Program(comp, dm)(inp, out)

test_transpose(torch.randn(32,32), torch.zeros(32,32))
print('✓ Transpose works standalone!')
