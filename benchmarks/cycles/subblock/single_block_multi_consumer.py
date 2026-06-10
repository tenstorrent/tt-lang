# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-consumer DST kernels over ONE block, compute-isolated.

The five patterns from ``test/python/test_dst_multi_consumer.py`` -- block args
or op results consumed by multiple ops, stressing copy_tile insertion against
register clobbering:

  mc_silu         : out = x * sigmoid(x)            unary + binary on a block arg
  mc_unary_binary : out = abs(x) + (x+y) + (x*y)    1 unary + 2 binary on x
  mc_three        : sigmoid(a), exp(a), a+b         2 unary + 1 binary, 3 outputs
  mc_square       : out = x * x                     same value both operands
  mc_branch       : exp(abs(a)), abs(a)+b           op result feeds unary + binary

All stripped to the bare compute (same style as single_block_add): compute
reserves every block (inputs are uninitialized L1; only cycles matter) and the
data-movement threads are empty.
"""

from __future__ import annotations

import ttl


def _decorator_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options):
    kw = dict(grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en)
    if compiler_options is not None:
        kw["options"] = compiler_options
    return kw


def make_single_block_mc_silu_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """out = x * sigmoid(x)."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_decorator_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __mc_silu(x, out) -> None:
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (x_dfb.reserve() as xv, out_dfb.reserve() as o):
                o.store(xv * ttl.math.sigmoid(xv))

        @ttl.datamovement()
        def read():
            pass

        @ttl.datamovement()
        def write():
            pass

    return __mc_silu


def make_single_block_mc_unary_binary_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """out = abs(x) + (x + y) + (x * y)."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_decorator_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __mc_unary_binary(x, y, out) -> None:
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(R, C), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (x_dfb.reserve() as xv, y_dfb.reserve() as yv, out_dfb.reserve() as o):
                abs_x = ttl.math.abs(xv)
                add_result = xv + yv
                mul_result = xv * yv
                o.store(abs_x + add_result + mul_result)

        @ttl.datamovement()
        def read():
            pass

        @ttl.datamovement()
        def write():
            pass

    return __mc_unary_binary


def make_single_block_mc_three_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """sigmoid(a), exp(a), a+b into three outputs."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_decorator_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __mc_three(a, b, out_sig, out_exp, out_add) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        sig_dfb = ttl.make_dataflow_buffer_like(out_sig, shape=(R, C), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(out_exp, shape=(R, C), block_count=2)
        add_dfb = ttl.make_dataflow_buffer_like(out_add, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                b_dfb.reserve() as bv,
                sig_dfb.reserve() as o_sig,
                exp_dfb.reserve() as o_exp,
                add_dfb.reserve() as o_add,
            ):
                o_sig.store(ttl.math.sigmoid(av))
                o_exp.store(ttl.math.exp(av))
                o_add.store(av + bv)

        @ttl.datamovement()
        def read():
            pass

        @ttl.datamovement()
        def write():
            pass

    return __mc_three


def make_single_block_mc_square_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """out = x * x (same value both operands)."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_decorator_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __mc_square(x, out) -> None:
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (x_dfb.reserve() as xv, out_dfb.reserve() as o):
                o.store(xv * xv)

        @ttl.datamovement()
        def read():
            pass

        @ttl.datamovement()
        def write():
            pass

    return __mc_square


def make_single_block_mc_branch_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """abs(a) branches: exp(abs(a)) and abs(a)+b into two outputs."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_decorator_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __mc_branch(a, b, out_exp, out_add) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        exp_dfb = ttl.make_dataflow_buffer_like(out_exp, shape=(R, C), block_count=2)
        add_dfb = ttl.make_dataflow_buffer_like(out_add, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                b_dfb.reserve() as bv,
                exp_dfb.reserve() as o_exp,
                add_dfb.reserve() as o_add,
            ):
                abs_val = ttl.math.abs(av)
                o_exp.store(ttl.math.exp(abs_val))
                o_add.store(abs_val + bv)

        @ttl.datamovement()
        def read():
            pass

        @ttl.datamovement()
        def write():
            pass

    return __mc_branch
