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
- A real ttnn GlobalSemaphore captured into template_args and func_args.
"""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl
from ttl.diagnostics import TTLangCompileError


NEGATE_HEADER = os.path.join(os.path.dirname(__file__), "include", "negate_tile_op.hpp")
TYPED_ARGS_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "typed_args_op.hpp"
)
SEMAPHORE_TEMPLATE_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "semaphore_template_op.hpp"
)
MODULE_GLOBAL_SEMAPHORE = None


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


def _make_semaphore_template_extern(global_sem):
    """Create an operation whose C++ template identity includes a semaphore."""

    @ttl.operation(grid=(1, 1))
    def semaphore_template_extern(inp, out):
        in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
                ttl.call_extern_func(
                    SEMAPHORE_TEMPLATE_HEADER,
                    "semaphore_template_negate_shim",
                    template_args=[
                        ttl.get_dfb_id(in_dfb),
                        ttl.get_dfb_id(out_dfb),
                        global_sem,
                    ],
                    func_args=[in_dfb, out_dfb, global_sem],
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

    return semaphore_template_extern


@ttl.operation(grid=(1, 1))
def _module_global_semaphore_extern(inp, out):
    in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        with in_dfb.wait() as in_blk, out_dfb.reserve() as out_blk:
            ttl.call_extern_func(
                SEMAPHORE_TEMPLATE_HEADER,
                "semaphore_template_negate_shim",
                template_args=[
                    ttl.get_dfb_id(in_dfb),
                    ttl.get_dfb_id(out_dfb),
                    MODULE_GLOBAL_SEMAPHORE,
                ],
                func_args=[in_dfb, out_dfb, MODULE_GLOBAL_SEMAPHORE],
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


def test_global_semaphore_template_arg(device):
    """Distinct captures produce distinct compiled template identities."""

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    first_sem = ttnn.create_global_semaphore(device, core_ranges, 0)
    second_sem = ttnn.create_global_semaphore(device, core_ranges, 0)
    assert ttnn.get_global_semaphore_address(
        first_sem
    ) != ttnn.get_global_semaphore_address(second_sem)

    first_operation = _make_semaphore_template_extern(first_sem)
    second_operation = _make_semaphore_template_extern(second_sem)

    inp_torch = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    first_out = to_l1(torch.zeros_like(inp_torch), device)
    second_out = to_l1(torch.zeros_like(inp_torch), device)
    first_operation(inp, first_out)
    second_operation(inp, second_out)

    expected = -inp_torch
    assert_allclose(ttnn.to_torch(first_out), expected)
    assert_allclose(ttnn.to_torch(second_out), expected)


def test_global_semaphore_getter_failure(device, monkeypatch):
    """TTNN getter failures remain visible without selecting another getter."""

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    global_sem = ttnn.create_global_semaphore(device, core_ranges, 0)
    operation = _make_semaphore_template_extern(global_sem)
    inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros_like(inp_torch), device)

    def fail_address_lookup(_semaphore):
        raise RuntimeError("test GlobalSemaphore address failure")

    monkeypatch.setattr(ttnn, "get_global_semaphore_address", fail_address_lookup)
    with pytest.raises(RuntimeError, match="test GlobalSemaphore address failure"):
        operation(inp, out)


def test_global_semaphore_address_is_unsigned(device, monkeypatch):
    """Addresses above INT32_MAX remain unsigned template and function values."""

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    global_sem = ttnn.create_global_semaphore(device, core_ranges, 0)
    operation = _make_semaphore_template_extern(global_sem)
    high_address = 0xF0000000
    monkeypatch.setattr(
        ttnn, "get_global_semaphore_address", lambda _semaphore: high_address
    )
    inp_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros_like(inp_torch), device)

    operation(inp, out)

    assert_allclose(ttnn.to_torch(out), -inp_torch)


def test_module_global_semaphore_is_rejected(device, monkeypatch):
    """Module globals cannot provide stable operation or cache identity."""

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    global_sem = ttnn.create_global_semaphore(device, core_ranges, 0)
    monkeypatch.setitem(globals(), "MODULE_GLOBAL_SEMAPHORE", global_sem)
    inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros_like(inp_torch), device)

    with pytest.raises(
        TTLangCompileError,
        match="ttnn.GlobalSemaphore must be captured by an operation factory",
    ):
        _module_global_semaphore_extern(inp, out)
