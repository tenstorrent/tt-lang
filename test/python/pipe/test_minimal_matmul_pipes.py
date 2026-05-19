# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal-matmul mirror with PipeNets.

Mirrors the dataflow shape and the canonical test shape of tt-metal's
[`ttnn/cpp/ttnn/operations/experimental/minimal_matmul/`][mm-tree]
at commit `c296ef469fe6aab65ab0d359e164b14b62d92bfc`. The shape and
block size match the tt-metal nightly unit test
[`tests/ttnn/nightly/.../test_minimal_matmul.py::test_linear`][mm-test]:
M = K = N = 4096, M_block = K_block = N_block = 8 tiles.

Output is sharded across the device grid via `_even_split`. Every core
owns `m_blocks_per_node x n_blocks_per_node` output blocks of
`BLOCK_M x BLOCK_N` tiles each. For each output block the K-loop runs
`K_BLOCKS` accumulation steps; at every step:
  - `A[mb, kb]` is broadcast across row `mb` (one multicast Pipe per
    row).
  - `B[kb, nb]` is broadcast across column `nb` (one multicast Pipe
    per column).
  - Compute does one block-level matmul accumulation (`out_blk += a_blk
    @ b_blk`) into a tile-level DST register.

Two-NOC split: A read+broadcast runs on `dm_read`, B read+broadcast and
output write run on `dm_write`. Matches the channel assignment in
[`minimal_matmul_program_factory.cpp:229-247`][factory-noc-policy]
(small input + output on `RISCV_1`/`NOC_1`, large input on
`RISCV_0`/`NOC_0`).

In tt-metal `minimal_matmul`, A and B are propagated across each
row/column via a unicast forwarding chain with pre-push, not via a
hardware multicast. tt-lang today lowers `Pipe(src=p, dst=slice(...))`
to `noc_async_write_multicast`. The chain rewrite that would close
that gap is captured in `docs/development/PipeOptimizations.md`. The
PipeNet shape here is the same; only the lowering primitive differs.

[mm-tree]: https://github.com/tenstorrent/tt-metal/tree/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul
[mm-test]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py#L146-L167
[factory-noc-policy]: https://github.com/tenstorrent/tt-metal/blob/c296ef469fe6aab65ab0d359e164b14b62d92bfc/ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp#L229-L247
"""

import pytest
import torch
import ttl
from ttl import ttl_api

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32
BLOCK_M = 8
BLOCK_N = 8
BLOCK_K = 8
BLOCK_SIZE = BLOCK_M * TILE  # 256

M_DIM = 4096
K_DIM = 4096
N_DIM = 4096

M_BLOCKS = M_DIM // BLOCK_SIZE
N_BLOCKS = N_DIM // BLOCK_SIZE
K_BLOCKS = K_DIM // BLOCK_SIZE


def _even_split(n_blocks, max_grid):
    """Pick blocks_per_node that divides n_blocks evenly without exceeding
    `max_grid` cores along that axis."""
    bpn = -(-n_blocks // max_grid)
    while n_blocks % bpn != 0:
        bpn += 1
    return bpn, n_blocks // bpn


@ttl.operation(grid="auto")
def minimal_matmul_kernel(a, b, out):
    num_cols, num_rows = ttl.grid_size(dims=2)
    m_blocks_per_node, num_rows_used = _even_split(M_BLOCKS, num_rows)
    n_blocks_per_node, num_cols_used = _even_split(N_BLOCKS, num_cols)

    # One row-broadcast Pipe per used row: source is column 0 of that row,
    # destinations cover columns 0 through num_cols_used.
    a_pipes = [
        ttl.Pipe(src=(0, row), dst=(slice(0, num_cols_used), row))
        for row in range(num_rows_used)
    ]
    a_net = ttl.PipeNet(a_pipes)

    # One column-broadcast Pipe per used column: source is row 0 of that
    # column, destinations cover rows 0 through num_rows_used.
    b_pipes = [
        ttl.Pipe(src=(col, 0), dst=(col, slice(0, num_rows_used)))
        for col in range(num_cols_used)
    ]
    b_net = ttl.PipeNet(b_pipes)

    a_cb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_M, BLOCK_K), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(BLOCK_K, BLOCK_N), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_M, BLOCK_N), block_count=2)

    @ttl.compute()
    def compute():
        if a_net.is_active():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                for local_nb in range(n_blocks_per_node):
                    with out_cb.reserve() as out_blk:
                        out_blk.store(ttl.block.fill(0, shape=out_blk.shape))
                        for _ in range(K_BLOCKS):
                            a_blk = a_cb.wait()
                            b_blk = b_cb.wait()
                            out_blk += a_blk @ b_blk
                            a_blk.pop()
                            b_blk.pop()

    @ttl.datamovement()
    def dm_read():
        if a_net.is_active():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with a_cb.reserve() as a_blk:

                            def read_a(pipe):
                                ttl.copy(
                                    a[mr : mr + BLOCK_M, kc : kc + BLOCK_K], a_blk
                                ).wait()
                                ttl.copy(a_blk, pipe).wait()

                            a_net.if_src(read_a)

                            def recv_a(pipe):
                                ttl.copy(pipe, a_blk).wait()

                            a_net.if_dst(recv_a)

    @ttl.datamovement()
    def dm_write():
        if b_net.is_active():
            node_n, node_m = ttl.node(dims=2)
            for local_mb in range(m_blocks_per_node):
                mb = node_m * m_blocks_per_node + local_mb
                mr = mb * BLOCK_M
                for local_nb in range(n_blocks_per_node):
                    nb = node_n * n_blocks_per_node + local_nb
                    nc = nb * BLOCK_N
                    for kb in range(K_BLOCKS):
                        kc = kb * BLOCK_K
                        with b_cb.reserve() as b_blk:

                            def read_b(pipe):
                                ttl.copy(
                                    b[kc : kc + BLOCK_K, nc : nc + BLOCK_N], b_blk
                                ).wait()
                                ttl.copy(b_blk, pipe).wait()

                            b_net.if_src(read_b)

                            def recv_b(pipe):
                                ttl.copy(pipe, b_blk).wait()

                            b_net.if_dst(recv_b)

                    with out_cb.wait() as out_blk:
                        ttl.copy(
                            out_blk, out[mr : mr + BLOCK_M, nc : nc + BLOCK_N]
                        ).wait()


def test_minimal_matmul_pipes(device):
    """4096x4096x4096 matmul with row-broadcast A and column-broadcast B,
    matching the canonical tt-metal `minimal_matmul::test_linear` shape."""
    # TODO(#585): re-enable on Wormhole once the dispatch-FW residue
    # triggered by the 8-dest a-pipe multicast lowering is fixed. The test
    # itself passes, but leaves ethernet dispatch core (x=25,y=17) in
    # RUN_MSG_INIT (0x40) so the next OpenDevice times out waiting for
    # physical cores. Same error signature as tt-metal#43511 (Galaxy logs
    # 32 fatals and continues; single-device WH cannot recover).
    arch = ttl_api._detect_device_arch(device)
    if arch and "wormhole" in arch:
        pytest.skip(
            "WH dispatch hang from 8-dest a-pipe multicast (RUN_MSG_INIT "
            "residue) — see #585"
        )

    a_torch = torch.randn(M_DIM, K_DIM, dtype=torch.bfloat16) / K_BLOCKS
    b_torch = torch.randn(K_DIM, N_DIM, dtype=torch.bfloat16) / K_BLOCKS

    a_tt = to_dram(a_torch, device)
    b_tt = to_dram(b_torch, device)
    out_tt = to_dram(torch.zeros(M_DIM, N_DIM, dtype=torch.bfloat16), device)

    minimal_matmul_kernel(a_tt, b_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()
    expected = a_torch.float() @ b_torch.float()
    assert_pcc(expected, result)
