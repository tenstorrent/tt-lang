# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ttl-insert-cb-sync auto-injection edge cases.

Each test exercises a distinct shape that the auto pop/push placement must
handle, including the issue #536 follow-up case_a and case_b reproducers
(deferred consumer uses across multiple consecutive cb.wait() calls on the
same DFB).
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402

TILE = 32


# ---------------------------------------------------------------------------
# Deferred consumer uses across multiple consecutive cb.wait() calls.
#
# The auto-pop pass clamps each wait's owned-use search at the next wait on
# the same DFB. If the consumer use of an earlier wait's tile lives past the
# later waits (e.g., 4 waits followed by 4 stores), the pass fails to find
# the use, places the pop right after the wait, and the read pointer
# advances before the data is consumed. See issue #536 follow-up comment.
# ---------------------------------------------------------------------------


def _run(device, kernel, num_out_tiles, expected):
    out_t = to_dram(
        torch.full((TILE, num_out_tiles * TILE), -42.0, dtype=torch.bfloat16),
        device,
    )
    kernel(out_t)
    ttnn.synchronize_device(device)
    out_h = ttnn.to_torch(out_t)
    actual = [out_h[0, i * TILE].item() for i in range(num_out_tiles)]
    assert actual == expected, f"actual={actual} expected={expected}"


@pytest.mark.requires_device
def test_issue_536_followup_case_a_three_waits_no_loop(device):
    """case_a from issue #536 follow-up: 3 consecutive cb.wait() calls in
    compute() with no enclosing loop, all consumer stores after the last
    wait."""

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 11.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 22.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 33.0))

            t1 = cb.wait()
            t2 = cb.wait()
            t3 = cb.wait()

            with out_cb.reserve() as o:
                o.store(t1)
            with out_cb.reserve() as o:
                o.store(t2)
            with out_cb.reserve() as o:
                o.store(t3)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 2]).wait()

    _run(device, repro, 3, [11.0, 22.0, 33.0])


@pytest.mark.requires_device
def test_issue_536_followup_case_b_four_waits_in_loop(device):
    """case_b from issue #536 follow-up: 4 consecutive cb.wait() calls
    inside a for-loop in compute(), 3 iterations, all consumer stores
    after the four waits in each iteration."""

    N_ITERS = 3
    N_PER_ITER = 4
    TOTAL = N_ITERS * N_PER_ITER

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=TOTAL)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 1.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 2.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 3.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 4.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 5.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 6.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 7.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 8.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 9.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 10.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 11.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 12.0))

            for _ in range(N_ITERS):
                t1 = cb.wait()
                t2 = cb.wait()
                t3 = cb.wait()
                t4 = cb.wait()
                with out_cb.reserve() as o:
                    o.store(t1)
                with out_cb.reserve() as o:
                    o.store(t2)
                with out_cb.reserve() as o:
                    o.store(t3)
                with out_cb.reserve() as o:
                    o.store(t4)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            for col in range(TOTAL):
                blk = out_cb.wait()
                ttl.copy(blk, out[0, col]).wait()

    _run(device, repro, TOTAL, [float(i + 1) for i in range(TOTAL)])


@pytest.mark.requires_device
def test_interleaved_wait_consume_pop_baseline(device):
    """Sanity check: the safe shape (consume each wait before the next wait)
    works after the #536 fix. This is the form the auto-pop pass currently
    reasons about correctly."""

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 1.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 2.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 3.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 4.0))

            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 2]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 3]).wait()

    _run(device, repro, 4, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Reused Python variable name ("tx-name collision"): the second assignment
# rebinds the local but the first acquire's SSA value still has uses. The
# auto-pop pass operates on SSA values, so this should be unaffected.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_python_name_reuse_does_not_alias_ssa(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_cb.reserve() as v:
                v.store(ttl.math.fill(v, 5.0))
            with out_cb.reserve() as v:
                v.store(ttl.math.fill(v, 6.0))

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            tx = out_cb.wait()
            tx = ttl.copy(tx, out[0, 0])
            tx.wait()
            tx = out_cb.wait()
            tx = ttl.copy(tx, out[0, 1])
            tx.wait()

    _run(device, repro, 2, [5.0, 6.0])


# ---------------------------------------------------------------------------
# Nested scf.for with independent acquires in the inner and outer bodies.
# updateBoundary() only treats acquires that share a common ancestor block
# as boundaries; an inner-loop acquire never bounds an outer-loop acquire.
# Verify that auto-pop placement remains correct across the loop boundary.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_nested_for_independent_acquires_per_loop(device):
    OUTER = 2
    INNER = 3
    TOTAL = OUTER * INNER

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=TOTAL)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 1.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 2.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 3.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 4.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 5.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 6.0))

            for _outer in range(OUTER):
                for _inner in range(INNER):
                    with cb.wait() as src, out_cb.reserve() as dst:
                        dst.store(src)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            for col in range(TOTAL):
                blk = out_cb.wait()
                ttl.copy(blk, out[0, col]).wait()

    _run(device, repro, TOTAL, [float(i + 1) for i in range(TOTAL)])


# ---------------------------------------------------------------------------
# Mixed immediate + deferred consumer uses. Some cb.wait results are consumed
# before the next wait; others are consumed after multiple subsequent waits.
# Boundary handling must be correct for both shapes simultaneously.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_mixed_immediate_and_deferred_consumes(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 100.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 200.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 300.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 400.0))

            # First wait + immediate consume.
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            # Three more waits with deferred consumes after all of them.
            t2 = cb.wait()
            t3 = cb.wait()
            t4 = cb.wait()
            with out_cb.reserve() as o:
                o.store(t2)
            with out_cb.reserve() as o:
                o.store(t3)
            with out_cb.reserve() as o:
                o.store(t4)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 2]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 3]).wait()

    _run(device, repro, 4, [100.0, 200.0, 300.0, 400.0])


# ---------------------------------------------------------------------------
# Long chain of consecutive cb.wait acquires with deferred consumes. Stresses
# the boundary-relaxed walk on a wider chain than case_a / case_b.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_eight_consecutive_waits_deferred_consumes(device):
    N = 8

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 1.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 2.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 3.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 4.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 5.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 6.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 7.0))
            with cb.reserve() as v:
                v.store(ttl.math.fill(v, 8.0))

            t1 = cb.wait()
            t2 = cb.wait()
            t3 = cb.wait()
            t4 = cb.wait()
            t5 = cb.wait()
            t6 = cb.wait()
            t7 = cb.wait()
            t8 = cb.wait()
            with out_cb.reserve() as o:
                o.store(t1)
            with out_cb.reserve() as o:
                o.store(t2)
            with out_cb.reserve() as o:
                o.store(t3)
            with out_cb.reserve() as o:
                o.store(t4)
            with out_cb.reserve() as o:
                o.store(t5)
            with out_cb.reserve() as o:
                o.store(t6)
            with out_cb.reserve() as o:
                o.store(t7)
            with out_cb.reserve() as o:
                o.store(t8)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 2]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 3]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 4]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 5]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 6]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 7]).wait()

    _run(device, repro, N, [float(i + 1) for i in range(N)])


# ---------------------------------------------------------------------------
# Two distinct CBs interleaved: each wait pair has deferred consumes. The
# next-acquire boundary is per-CB; this test verifies independence.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_two_cbs_interleaved_deferred_consumes(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb_a = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        cb_b = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            with cb_a.reserve() as v:
                v.store(ttl.math.fill(v, 10.0))
            with cb_a.reserve() as v:
                v.store(ttl.math.fill(v, 20.0))
            with cb_b.reserve() as v:
                v.store(ttl.math.fill(v, 30.0))
            with cb_b.reserve() as v:
                v.store(ttl.math.fill(v, 40.0))

            # Interleave waits across two CBs; defer consumes for all four.
            a1 = cb_a.wait()
            b1 = cb_b.wait()
            a2 = cb_a.wait()
            b2 = cb_b.wait()
            with out_cb.reserve() as o:
                o.store(a1)
            with out_cb.reserve() as o:
                o.store(b1)
            with out_cb.reserve() as o:
                o.store(a2)
            with out_cb.reserve() as o:
                o.store(b2)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 2]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 3]).wait()

    _run(device, repro, 4, [10.0, 30.0, 20.0, 40.0])


# ---------------------------------------------------------------------------
# Producer-side deferred reserves: 3 cb.reserve handles acquired, then 3
# stores fired after all reserves. Mirror of case_a for the producer side.
# Pattern is the explicit reserve-handle form used in test_layernorm.py and
# simple_bcast.py rather than the `with cb.reserve() as v` form.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_three_consecutive_reserves_deferred_stores(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            r1 = cb.reserve()
            r2 = cb.reserve()
            r3 = cb.reserve()
            r1.store(ttl.math.fill(r1, 7.0))
            r2.store(ttl.math.fill(r2, 8.0))
            r3.store(ttl.math.fill(r3, 9.0))

            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 2]).wait()

    _run(device, repro, 3, [7.0, 8.0, 9.0])
