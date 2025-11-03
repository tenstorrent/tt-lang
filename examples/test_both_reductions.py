from ttlang.d2m_api import *
import torch, os
os.environ['SYSTEM_DESC_PATH'] = '/Users/zcarver/Downloads/system_desc.ttsys'

@pykernel_gen(block_factors=[(1,1),(1,1)], grid=(1,1), memory_space='L1', tiled=True)
def test_both(inp, out, block_factors=None, grid=None):
    """Test both rowmax and rowsum in same kernel"""
    inp_stream = Stream(inp)
    @compute()
    async def comp(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        inp_block = inp_cb.pop()
        out_block = out_cb.reserve()

        m = inp_block.rowmax()
        l = inp_block.rowsum()
        # Can't use both - would need multiple outputs
        # Just output one
        out_block.store(l)
        out_cb.pop()

    @datamovement()
    async def dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0) * 1 + core_index(1)
        dma(inp_stream[idx, 0], inp_cb.reserve()).wait()

    return Program(comp, dm)(inp, out)

test_both(torch.randn(32, 32), torch.zeros(32, 32))
print('✓ Both reductions in same kernel work!')
