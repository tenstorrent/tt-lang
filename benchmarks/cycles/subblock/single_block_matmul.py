# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""y = a @ b over exactly ONE M x N output block, with no output-block loops.

A single core does one M x N output block reducing over K, as a single matmul
(``y_blk.store(a_blk @ b_blk)``) -- K accumulates in DST, no L1 accumulation and
no `for` over output blocks. Parallel dims are (M, N); K is the reduction.
matmul forces fp32 dest accumulation, which halves the usable DST capacity, so a
forced subblock (sM, sN) needs sM*sN <= capacity/2.

  - make_single_block_matmul          : real DRAM read/matmul/write (correctness).
  - make_single_block_matmul_no_dram  : compute-isolated (data movement gutted);
                                        used by the cycle benchmark.
"""

from __future__ import annotations

import ttnn
import ttl


def make_single_block_matmul(
    *,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Real single-block matmul: read one M x K and K x N block, matmul, write
    one M x N block. No output-block loops."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    M, N, K = m_tiles, n_tiles, k_tiles

    @ttl.operation(**decorator_kwargs)
    def __matmul_single_block(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(M, K), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K, N), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(M, N), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                y_dfb.reserve() as y_blk,
            ):
                y_blk.store(a_blk @ b_blk)

        @ttl.datamovement()
        def read():
            with (a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk):
                tx_a = ttl.copy(a[0:M, 0:K], a_blk)
                tx_b = ttl.copy(b[0:K, 0:N], b_blk)
                tx_a.wait()
                tx_b.wait()

        @ttl.datamovement()
        def write():
            with y_dfb.wait() as y_blk:
                tx = ttl.copy(y_blk, y[0:M, 0:N])
                tx.wait()

    return __matmul_single_block


def make_single_block_matmul_no_dram(
    *,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block matmul, stripped to the bare compute. The
    compute thread *reserves* all three blocks itself and does the matmul (a/b
    hold uninitialized L1 -- correctness is irrelevant, only the compute cycles
    are), so there is no CB handshake at all: the data-movement threads do nothing
    (no reserve/wait), leaving the measured cycles as pure compute."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    M, N, K = m_tiles, n_tiles, k_tiles

    @ttl.operation(**decorator_kwargs)
    def __matmul_single_block_no_dram(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(M, K), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K, N), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(M, N), block_count=2)

        # Compute owns all three blocks (no producer/consumer handshake): reserve
        # a/b/y, matmul, store. a_blk/b_blk are uninitialized L1 -- we only measure
        # the compute, so the data is don't-care.
        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
                y_dfb.reserve() as y_blk,
            ):
                y_blk.store(a_blk @ b_blk)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __matmul_single_block_no_dram
