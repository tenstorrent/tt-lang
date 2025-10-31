# Test exp using direct linalg generation (bypass method call)
from ttlang.d2m_api import *
from ttlang.operators import _create_linalg_generic_unary
from ttmlir.ir import *
from ttmlir.dialects import d2m
import torch

@pykernel_gen(grid=(1,1), block_factors=[(1,1), (1,1)])
def test_exp(inp, out):
    inp_stream = Stream(inp)

    @compute()
    async def comp(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        inp_block = inp_cb.pop()
        out_block = out_cb.reserve()

        # Try direct MLIR call instead of method
        ctx = inp_block.type.context
        rank = len(inp_block.type.shape)
        identity_map = AffineMap.get_identity(rank, ctx)

        result = _create_linalg_generic_unary(
            inp_block,
            output_shape=list(inp_block.type.shape),
            affine_maps=[identity_map, identity_map],
            iterator_types=["parallel"] * rank,
            tile_op_builder=lambda result_type, inp_arg, out_arg: d2m.tile_exp(
                result_type, inp_arg
            ),
        )

        out_block.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        idx = core_index(0)
        shard = inp_cb.reserve()
        dma(inp_stream[idx, 0], shard).wait()

    return Program(comp, dm)(inp, out)

test_exp(torch.randn(32,32), torch.zeros(32,32))
print('✓ Exp direct call works!')
