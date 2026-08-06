# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for ttl.call_extern_func.

Covers:
- Struct-based compute header with DFB IDs as template args and direct DFB
  operands (negate).
- int/bool/float values as both template_args and func_args.
- A DFB passed directly as a func_arg (CB index via get_compile_time_arg_val).
- A tensor passed as a func_arg (runtime buffer base address).
"""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


NEGATE_HEADER = os.path.join(os.path.dirname(__file__), "include", "negate_tile_op.hpp")
TYPED_ARGS_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "typed_args_op.hpp"
)
TENSOR_ADDRESS_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "tensor_address_op.hpp"
)


@ttl.operation(grid=(1, 1))
def negate_extern(inp, out):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
            ttl.call_extern_func(
                NEGATE_HEADER,
                "negate_tile_shim",
                template_args=[
                    ttl.get_dfb_id(in_dfb),
                    ttl.get_dfb_id(out_dfb),
                ],
                func_args=[in_dfb, out_dfb],
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


@ttl.operation(grid=(1, 1))
def typed_args_extern(inp, out):
    """Exercise int/bool/float template+func args and a DFB as func_arg.

    template_args:
      OutCB=get_dfb_id(out), IntScale=2, NegateTpl=True, ScaleTpl=0.5
    func_args:
      in_dfb and out_dfb (DFBs), scale_f=3.0, int_factor=2,
      also_negate=False

    expected = -inp * 2 * 0.5 * 3.0 * 2 = -inp * 6
    """
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
            ttl.call_extern_func(
                TYPED_ARGS_HEADER,
                "typed_args_shim",
                template_args=[
                    ttl.get_dfb_id(out_dfb),
                    2,  # int
                    True,  # bool
                    0.5,  # float (IEEE bits in template)
                ],
                func_args=[
                    in_dfb,  # DFB -> CB index
                    out_dfb,  # DFB -> CB index
                    3.0,  # float
                    2,  # int
                    False,  # bool
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


@ttl.operation(grid=(1, 1))
def tensor_address_extern(inp, out):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
            ttl.call_extern_func(
                TENSOR_ADDRESS_HEADER,
                "tensor_address_alias_shim",
                template_args=[ttl.get_dfb_id(out_dfb)],
                func_args=[inp, inp, in_dfb, out_dfb],
            )

    @ttl.datamovement()
    def dm_read():
        blk = in_dfb.reserve()
        ttl.copy(inp[0, 0], blk).wait()
        blk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        ttl.copy(blk, out[0, 0]).wait()
        blk.pop()


def test_negate_extern(device):
    inp_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros_like(inp_torch), device)

    negate_extern(inp, out)

    result = ttnn.to_torch(out)
    expected = -inp_torch
    assert_allclose(result, expected)


def test_typed_args_extern(device):
    inp_torch = torch.full((32, 32), 4.0, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros_like(inp_torch), device)

    typed_args_extern(inp, out)

    result = ttnn.to_torch(out)
    # -inp * IntScale(2) * ScaleTpl(0.5) * scale_f(3.0) * int_factor(2)
    expected = -inp_torch * 6.0
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    (
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ),
    ids=("bf16", "fp32"),
)
@pytest.mark.parametrize(
    "memory_config",
    (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ids=("l1", "dram"),
)
def test_tensor_address_extern(device, torch_dtype, ttnn_dtype, memory_config):
    inp_torch = torch.full((32, 32), 4.0, dtype=torch_dtype)
    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    out = ttnn.from_torch(
        torch.zeros_like(inp_torch),
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tensor_address_extern(inp, out)

    assert_allclose(ttnn.to_torch(out), inp_torch)
