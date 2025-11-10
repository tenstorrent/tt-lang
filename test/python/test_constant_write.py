# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Simplest possible test: write a constant to output

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1)])
def test_constant_write(out):
    @compute()
    async def compute_constant(out_cb: CircularBuffer):
        o = out_cb.reserve()
        # Just write some constant value (all zeros for now since we can't create constants easily)
        # TODO: Need a way to create constant tensors in DSL
        o.store(o)  # Store whatever is in the reserved buffer
        out_cb.pop()

    @datamovement()
    async def dm(out_cb: CircularBuffer):
        pass

    return Program(compute_constant, dm)(out)


out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"out[0:2, 0:2] = \n{out[0:2, 0:2]}")

test_constant_write(out)

print("\n=== AFTER KERNEL ===")
print(f"out[0:2, 0:2] = \n{out[0:2, 0:2]}")
print(f"out min/max/mean: {out.min().item():.4f} / {out.max().item():.4f} / {out.mean().item():.4f}")
