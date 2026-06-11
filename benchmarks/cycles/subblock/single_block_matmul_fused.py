# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused matmul + post-op over exactly ONE M x N block, compute-isolated.

The two fused patterns from ``test/python/test_matmul_subblock.py``, on the
matmul kind's geometry (y[M,N] = a[M,K] @ b[K,N], K reduced in DST):

  matmul_bias : out = (a @ b) + c     bias block added after the K reduction
  matmul_relu : out = relu(a @ b)     relu applied to DST before pack

Compiler forces fp32 dest acc for matmul, so the budget matches the plain
matmul kind; the post-op runs on the result tiles already in DST.

Stripped to the bare compute (same style as single_block_matmul): the compute
thread *reserves* all blocks itself (inputs are uninitialized L1) and the
data-movement threads are empty.
"""

from __future__ import annotations

import ttl


def make_single_block_matmul_bias_no_dram(
    *,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block matmul + bias: out = (a @ b) + c."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    M, N, K = m_tiles, n_tiles, k_tiles

    @ttl.operation(**decorator_kwargs)
    def __matmul_bias_single_block_no_dram(a, b, c, out) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(M, K), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K, N), block_count=2)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(M, N), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(M, N), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
                c_dfb.reserve() as c_blk,
                out_dfb.reserve() as o,
            ):
                o.store((a_blk @ b_blk) + c_blk)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __matmul_bias_single_block_no_dram


def make_single_block_matmul_relu_no_dram(
    *,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block matmul + relu: out = relu(a @ b)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    M, N, K = m_tiles, n_tiles, k_tiles

    @ttl.operation(**decorator_kwargs)
    def __matmul_relu_single_block_no_dram(a, b, out) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(M, K), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K, N), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(M, N), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
                out_dfb.reserve() as o,
            ):
                o.store(ttl.math.relu(a_blk @ b_blk))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __matmul_relu_single_block_no_dram