# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gated Delta Rule step over exactly ONE block, compute-isolated.

The recurrent GDN timestep from ``test/python/test_gdn_kernel.py`` scaled from
1x1 tile (D=32) to a D x D tile block so the subblock can sweep:

  S_scaled = alpha * S
  e        = S_scaled^T @ k
  delta    = v - e
  S_new    = S_scaled + beta * (k @ delta^T)
  o        = S_new^T @ q

Matmuls reduce over the D tile dim in DST (compiler forces fp32 dest acc, so
the matmul regions cap the budget); the transposes and elementwise chains share
the same global forced subblock. ``sn_local`` keeps its in-compute store->wait
handshake from the original to bridge S_new between the two phases.

Stripped to the bare compute (same style as single_block_add): the compute
thread *reserves* the six inputs itself (uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do
nothing. block_count=1: nine D x D CBs double-buffered would exceed L1.
"""

from __future__ import annotations

import ttl


def make_single_block_gdn_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,  # unused: GDN state is square (D x D tiles)
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block GDN step. D = row_tiles_per_block."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    D = row_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __gdn_single_block_no_dram(state_in, q_in, k_in, v_in, alpha_in, beta_in,
                                   state_out, out) -> None:
        si = ttl.make_dataflow_buffer_like(state_in, shape=(D, D), block_count=1)
        qi = ttl.make_dataflow_buffer_like(q_in, shape=(D, D), block_count=1)
        ki = ttl.make_dataflow_buffer_like(k_in, shape=(D, D), block_count=1)
        vi = ttl.make_dataflow_buffer_like(v_in, shape=(D, D), block_count=1)
        ai = ttl.make_dataflow_buffer_like(alpha_in, shape=(D, D), block_count=1)
        bi = ttl.make_dataflow_buffer_like(beta_in, shape=(D, D), block_count=1)
        so = ttl.make_dataflow_buffer_like(state_out, shape=(D, D), block_count=1)
        oo = ttl.make_dataflow_buffer_like(out, shape=(D, D), block_count=1)
        # Thread-local DFB bridging S_new between the two phases (as in the test).
        sn_local = ttl.make_dataflow_buffer_like(state_in, shape=(D, D), block_count=1)

        @ttl.compute()
        def compute():
            # Phase 1: S_new = alpha*S + beta * (k @ (v - (alpha*S)^T @ k)^T)
            with (
                si.reserve() as s,
                ai.reserve() as a,
                ki.reserve() as k,
                vi.reserve() as v,
                bi.reserve() as b,
            ):
                s_scaled = a * s
                st = ttl.transpose(s_scaled)
                e = st @ k
                delta = v - e
                dt = ttl.transpose(delta)
                outer = k @ dt
                s_new = s_scaled + b * outer

                with so.reserve() as so_blk:
                    so_blk.store(s_new)
                with sn_local.reserve() as snl:
                    snl.store(s_new)

            # Phase 2: o = S_new^T @ q
            with sn_local.wait() as sn, qi.reserve() as q:
                snt = ttl.transpose(sn)
                with oo.reserve() as oo_blk:
                    oo_blk.store(snt @ q)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __gdn_single_block_no_dram
