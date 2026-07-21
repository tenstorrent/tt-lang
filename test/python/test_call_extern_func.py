# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for ttl.call_extern_func with a struct-based compute header.

Calls an external negate-tile operation that follows the config-struct + Op-class
pattern used by external op libraries.  The compute thread invokes a shim
parameterised by DFB IDs via ttl.call_extern_func; data movement threads handle
CB synchronisation.
"""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


HEADER = os.path.join(os.path.dirname(__file__), "include", "negate_tile_op.hpp")


@ttl.operation(grid=(1, 1))
def negate_extern(inp, out):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
            ttl.call_extern_func(
                HEADER,
                "negate_tile_shim",
                template_args=[
                    ttl.get_dfb_id(in_dfb),
                    ttl.get_dfb_id(out_dfb),
                ],
            )

    @ttl.datamovement()
    def dm_read():
        blk = in_dfb.reserve()
        tx = ttl.copy(inp[0, 0], blk)
        tx.wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        tx = ttl.copy(blk, out[0, 0])
        tx.wait()
        blk.pop()


def test_negate_extern(device):
    inp_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros_like(inp_torch), device)

    negate_extern(inp, out)

    result = ttnn.to_torch(out)
    expected = -inp_torch
    assert_allclose(result, expected)
