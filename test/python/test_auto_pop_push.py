# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ttl-insert-cb-sync auto-injection edge cases.

Each test exercises a distinct pattern that the auto pop/push placement
must handle, including the issue #536 follow-up case_a and case_b reproducers
(deferred consumer uses across multiple consecutive cb.wait() calls on the
same DFB).

Several tests are marked xfail(strict). Each describes a real pattern
that currently produces wrong runtime output (or fails to compile) and
will start passing once a tracked compiler follow-up is merged. The
explanation for each is at the test site.
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
                v.store(ttl.block.fill(11.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(22.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(33.0, shape=v.shape))

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
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(3.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(4.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(5.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(6.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(7.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(8.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(9.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(10.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(11.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(12.0, shape=v.shape))

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
    """Sanity check: the safe form (consume each wait before the next wait)
    works after the #536 fix. This is the pattern the auto-pop pass
    currently reasons about correctly."""

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(3.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(4.0, shape=v.shape))

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
                v.store(ttl.block.fill(5.0, shape=v.shape))
            with out_cb.reserve() as v:
                v.store(ttl.block.fill(6.0, shape=v.shape))

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
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(3.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(4.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(5.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(6.0, shape=v.shape))

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
# Boundary handling must be correct for both patterns simultaneously.
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
                v.store(ttl.block.fill(100.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(200.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(300.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(400.0, shape=v.shape))

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
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(3.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(4.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(5.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(6.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(7.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(8.0, shape=v.shape))

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
                v.store(ttl.block.fill(10.0, shape=v.shape))
            with cb_a.reserve() as v:
                v.store(ttl.block.fill(20.0, shape=v.shape))
            with cb_b.reserve() as v:
                v.store(ttl.block.fill(30.0, shape=v.shape))
            with cb_b.reserve() as v:
                v.store(ttl.block.fill(40.0, shape=v.shape))

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
            r1.store(ttl.block.fill(7.0, shape=r1.shape))
            r2.store(ttl.block.fill(8.0, shape=r2.shape))
            r3.store(ttl.block.fill(9.0, shape=r3.shape))

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


# ---------------------------------------------------------------------------
# Wait-result fanout. A single cb.wait() result is consumed by multiple
# downstream stores; the SSA walk must discover every transitive use, not
# just the first one. If it stops early, a later store reads from a slot
# that has already been popped.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_wait_result_fanout_multiple_consumers(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(42.0, shape=v.shape))

            t = cb.wait()
            with out_cb.reserve() as o1:
                o1.store(t)
            with out_cb.reserve() as o2:
                o2.store(t)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()

    _run(device, repro, 2, [42.0, 42.0])


# ---------------------------------------------------------------------------
# xfail (#555). DM-thread producer with three consecutive reserves
# followed by three ttl.copy completions. ttl.copy takes a !ttl.cb operand
# directly rather than a tensor SSA value derived from cb_reserve, so the
# IR carries no def-use edge identifying which copy fills which reserve.
# The pass falls back to op-order reasoning and attributes all three
# copies to the last reserve. The push for the earlier reserves is
# emitted before any data is written; the buffer's write pointer advances past
# empty slots. Lifted by #555 (encode DFB ownership in SSA on ttl.copy).
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
@pytest.mark.xfail(
    strict=True,
    reason="Batched DM-thread reserve/copy/wait/push pattern. "
    "Lifted by #555 (encode DFB ownership in SSA on ttl.copy).",
)
def test_dm_read_three_consecutive_reserves_deferred_copies(device):
    @ttl.operation(grid=(1, 1))
    def repro(inp, out):
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=3)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            with inp_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with inp_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with inp_cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)

        @ttl.datamovement()
        def dm_read():
            r1 = inp_cb.reserve()
            r2 = inp_cb.reserve()
            r3 = inp_cb.reserve()
            tx1 = ttl.copy(inp[0, 0], r1)
            tx2 = ttl.copy(inp[1, 0], r2)
            tx3 = ttl.copy(inp[2, 0], r3)
            tx1.wait()
            tx2.wait()
            tx3.wait()
            r1.push()
            r2.push()
            r3.push()

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[1, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[2, 0]).wait()

    torch.manual_seed(11536)
    inp_t = to_dram(torch.randn((3 * TILE, TILE), dtype=torch.bfloat16), device)
    out_t = to_dram(torch.full((3 * TILE, TILE), -42.0, dtype=torch.bfloat16), device)
    repro(inp_t, out_t)
    ttnn.synchronize_device(device)
    inp_h = ttnn.to_torch(inp_t)
    out_h = ttnn.to_torch(out_t)
    assert torch.equal(out_h, inp_h)


# ---------------------------------------------------------------------------
# xfail (#555). DM-thread consumer with three consecutive cb.wait()
# acquires followed by three ttl.copy completions. Consumer-side mirror
# of the dm_read case above; ttl.copy reads from the bare !ttl.cb operand
# instead of the cb_wait result, so the pass cannot tell which copy
# consumes which acquired slot and pops the earlier slots before the
# corresponding copies read them. Lifted by #555.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
@pytest.mark.xfail(
    strict=True,
    reason="Batched DM-thread wait/copy/wait/pop pattern. "
    "Lifted by #555 (encode DFB ownership in SSA on ttl.copy).",
)
def test_dm_write_three_consecutive_waits_deferred_copies(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            with out_cb.reserve() as v:
                v.store(ttl.block.fill(50.0, shape=v.shape))
            with out_cb.reserve() as v:
                v.store(ttl.block.fill(60.0, shape=v.shape))
            with out_cb.reserve() as v:
                v.store(ttl.block.fill(70.0, shape=v.shape))

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            b1 = out_cb.wait()
            b2 = out_cb.wait()
            b3 = out_cb.wait()
            tx1 = ttl.copy(b1, out[0, 0])
            tx2 = ttl.copy(b2, out[0, 1])
            tx3 = ttl.copy(b3, out[0, 2])
            tx1.wait()
            tx2.wait()
            tx3.wait()
            b1.pop()
            b2.pop()
            b3.pop()

    _run(device, repro, 3, [50.0, 60.0, 70.0])


# ---------------------------------------------------------------------------
# xfail (#555). Cross-thread chain: dm_read reserves 4 slots up front then
# writes them, compute waits 4 then consumes them, dm_write waits 4 then
# writes them out. The compute-side auto-injection works (SSA def-use
# anchors ownership), but both DM threads inherit the same batched
# reserve/copy or wait/copy miscompile as the two tests above. Lifted by
# #555.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
@pytest.mark.xfail(
    strict=True,
    reason="Inherits the batched DM-thread reserve/wait miscompile in "
    "both dm_read and dm_write halves. Lifted by #555.",
)
def test_cross_thread_deferred_chain(device):
    @ttl.operation(grid=(1, 1))
    def repro(inp, out):
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=4)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            t1 = inp_cb.wait()
            t2 = inp_cb.wait()
            t3 = inp_cb.wait()
            t4 = inp_cb.wait()
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
            r1 = inp_cb.reserve()
            r2 = inp_cb.reserve()
            r3 = inp_cb.reserve()
            r4 = inp_cb.reserve()
            tx1 = ttl.copy(inp[0, 0], r1)
            tx2 = ttl.copy(inp[1, 0], r2)
            tx3 = ttl.copy(inp[2, 0], r3)
            tx4 = ttl.copy(inp[3, 0], r4)
            tx1.wait()
            tx2.wait()
            tx3.wait()
            tx4.wait()
            r1.push()
            r2.push()
            r3.push()
            r4.push()

        @ttl.datamovement()
        def dm_write():
            b1 = out_cb.wait()
            b2 = out_cb.wait()
            b3 = out_cb.wait()
            b4 = out_cb.wait()
            tx1 = ttl.copy(b1, out[0, 0])
            tx2 = ttl.copy(b2, out[1, 0])
            tx3 = ttl.copy(b3, out[2, 0])
            tx4 = ttl.copy(b4, out[3, 0])
            tx1.wait()
            tx2.wait()
            tx3.wait()
            tx4.wait()
            b1.pop()
            b2.pop()
            b3.pop()
            b4.pop()

    torch.manual_seed(536)
    inp_t = to_dram(torch.randn((4 * TILE, TILE), dtype=torch.bfloat16), device)
    out_t = to_dram(torch.full((4 * TILE, TILE), -42.0, dtype=torch.bfloat16), device)
    repro(inp_t, out_t)
    ttnn.synchronize_device(device)
    inp_h = ttnn.to_torch(inp_t)
    out_h = ttnn.to_torch(out_t)
    assert torch.equal(out_h, inp_h)


# ---------------------------------------------------------------------------
# Reordered consumes: consumer reads tile 2 before tile 1 (out of
# declaration order). Consecutive cb_wait acquires coalesce into one
# multi-tile `cb_wait_front(N*k)` plus per-block `tensor.extract_slice`
# views, so consume order is decoupled from release order; both tiles
# are present from the single coalesced wait, and the slice offsets
# index each block at lowering.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_reordered_consumes_decoupled_from_fifo(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))

            t1 = cb.wait()
            t2 = cb.wait()
            # Consume t2 BEFORE t1 -- requires per-tile src_idx to be correct.
            with out_cb.reserve() as o:
                o.store(t2)
            with out_cb.reserve() as o:
                o.store(t1)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()

    _run(device, repro, 2, [2.0, 1.0])


# ---------------------------------------------------------------------------
# Multi-tile block shape. shape=(1,2) means each CB slot holds two tiles.
# Consecutive cb.wait()s with deferred consumes verify the boundary
# handling does not assume single-tile geometry.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_multi_tile_block_shape_deferred_consumes(device):
    @ttl.operation(grid=(1, 1))
    def repro(inp, out):
        cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 2), block_count=2)

        @ttl.compute()
        def compute():
            t1 = cb.wait()
            t2 = cb.wait()
            with out_cb.reserve() as o:
                o.store(t1)
            with out_cb.reserve() as o:
                o.store(t2)

        @ttl.datamovement()
        def dm_read():
            r1 = cb.reserve()
            tx1 = ttl.copy(inp[0:1, 0:2], r1)
            tx1.wait()
            r1.push()
            r2 = cb.reserve()
            tx2 = ttl.copy(inp[0:1, 2:4], r2)
            tx2.wait()
            r2.push()

        @ttl.datamovement()
        def dm_write():
            b1 = out_cb.wait()
            ttl.copy(b1, out[0:1, 0:2]).wait()
            b1.pop()
            b2 = out_cb.wait()
            ttl.copy(b2, out[0:1, 2:4]).wait()
            b2.pop()

    torch.manual_seed(909)
    inp_t = to_dram(torch.randn((TILE, 4 * TILE), dtype=torch.bfloat16), device)
    out_t = to_dram(torch.full((TILE, 4 * TILE), -42.0, dtype=torch.bfloat16), device)
    repro(inp_t, out_t)
    ttnn.synchronize_device(device)
    inp_h = ttnn.to_torch(inp_t)
    out_h = ttnn.to_torch(out_t)
    assert torch.equal(out_h, inp_h)


# ---------------------------------------------------------------------------
# Tight block_count -- block_count exactly equal to the
# consecutive-acquire count, no slack. Producer must push all 4 before the
# consumer can read; ordering bugs that block_count slack would mask are
# exposed here.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_tight_block_count_four_consecutive_waits(device):
    N = 4

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(3.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(4.0, shape=v.shape))

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
            for col in range(N):
                blk = out_cb.wait()
                ttl.copy(blk, out[0, col]).wait()

    _run(device, repro, N, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Producer-side mixed -- a reserve consumed immediately followed by two
# reserves with deferred stores. Mirror of the mixed-immediate-deferred
# consumer test above on the producer side.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_producer_mixed_immediate_and_deferred_stores(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            # Immediate reserve + store (the with-block form).
            with cb.reserve() as v:
                v.store(ttl.block.fill(11.0, shape=v.shape))
            # Two deferred reserves with stores after both reserves.
            r2 = cb.reserve()
            r3 = cb.reserve()
            r2.store(ttl.block.fill(22.0, shape=r2.shape))
            r3.store(ttl.block.fill(33.0, shape=r3.shape))

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

    _run(device, repro, 3, [11.0, 22.0, 33.0])


# ---------------------------------------------------------------------------
# block_count=1 (single-slot CB). Degenerate but legal: every
# producer-consumer pair must serialize through the single slot. Tests
# the pass on the smallest legal CB topology.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_single_slot_cb_serialized(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(99.0, shape=v.shape))
            with cb.wait() as src, out_cb.reserve() as dst:
                dst.store(src)
            with cb.reserve() as v:
                v.store(ttl.block.fill(88.0, shape=v.shape))
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

    _run(device, repro, 2, [99.0, 88.0])


# ---------------------------------------------------------------------------
# Long DM-thread loop with many iterations. Exercises per-iteration pop
# placement under wider iteration counts than case_b's 12.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_long_dm_thread_loop_64_iterations(device):
    N = 64

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            for _ in range(N):
                with cb.reserve() as v:
                    v.store(ttl.block.fill(17.0, shape=v.shape))

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            for col in range(N):
                blk = cb.wait()
                ttl.copy(blk, out[0, col]).wait()

    _run(device, repro, N, [17.0] * N)


# ---------------------------------------------------------------------------
# Multiple direct CB uses on a single DM-thread acquire.
#
# A single cb.wait() followed by two ttl.copy() reads from the same slot to
# different output positions. Both copies are direct CB operands on the same
# acquire (criterion-b ownership). The pop must be inserted after the last copy; if
# findLastOwnedUse stopped at the first copy, the pop would advance the read
# pointer before the second copy reads, producing stale data in row 1.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_dm_write_two_copies_same_acquire(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(5.0, shape=v.shape))

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            ttl.copy(blk, out[0, 1]).wait()

    out_t = to_dram(torch.full((TILE, 2 * TILE), -42.0, dtype=torch.bfloat16), device)
    repro(out_t)
    ttnn.synchronize_device(device)
    out_h = ttnn.to_torch(out_t)
    assert out_h[0, 0].item() == 5.0
    assert out_h[0, TILE].item() == 5.0


# ---------------------------------------------------------------------------
# Producer-side analog of case_b: 3 consecutive cb.reserve() per iteration
# of an scf.for, with the matching stores deferred until after the third
# reserve. Each push must be inserted after its own slot's store, inside the loop
# body. Symmetric coverage to test 28 in insert_cb_sync.mlir for producers.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_producer_three_reserves_deferred_stores_in_loop(device):
    N_ITERS = 3
    N_PER_ITER = 3
    TOTAL = N_ITERS * N_PER_ITER

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=TOTAL)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

        @ttl.compute()
        def compute():
            for _ in range(N_ITERS):
                r1 = cb.reserve()
                r2 = cb.reserve()
                r3 = cb.reserve()
                r1.store(ttl.block.fill(1.0, shape=r1.shape))
                r2.store(ttl.block.fill(2.0, shape=r2.shape))
                r3.store(ttl.block.fill(3.0, shape=r3.shape))

            for _ in range(TOTAL):
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

    expected = [1.0, 2.0, 3.0] * N_ITERS
    _run(device, repro, TOTAL, expected)


@pytest.mark.requires_device
def test_wait_result_through_for_iter_args(device):
    """Auto-pop must preserve the waited tile through a tensor recurrence."""
    N = 4

    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(1.0, shape=v.shape))
            acc = cb.wait()
            for _ in range(N):
                acc = acc + acc
            with out_cb.reserve() as o:
                o.store(acc)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()

    _run(device, repro, 1, [float(2**N)])


# ---------------------------------------------------------------------------
# A third same-DFB acquire is interposed between two coalescable waits
# and their releases. Auto-pop places pop_t1 right after t1's last use,
# t3's wait runs before t2's last use, then pop_t2 is emitted. The coalescing
# rewrite collapses pop_t1 and pop_t2 into a single coalesced pop that
# now sits past the interposed t3 wait; this verifies correctness of
# that placement.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_third_acquire_interposed_between_coalesced_pops(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(1.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(2.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(3.0, shape=v.shape))

            t1 = cb.wait()
            t2 = cb.wait()
            with out_cb.reserve() as o:
                o.store(t1)
            t3 = cb.wait()
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

    _run(device, repro, 3, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Producer-side multi-tile block shape. Three consecutive cb.reserve()
# handles, each shape=(1, 2), with deferred stores on the block-shaped
# views. Verifies that the producer-side coalesce + per-block
# extract_slice + dst_idx fold line up for k > 1.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_producer_three_reserves_multi_tile_block_shape(device):
    @ttl.operation(grid=(1, 1))
    def repro(inp, out):
        cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 2), block_count=3)

        @ttl.compute()
        def compute():
            t = cb.wait()
            r1 = out_cb.reserve()
            r2 = out_cb.reserve()
            r3 = out_cb.reserve()
            r1.store(t)
            r2.store(t)
            r3.store(t)

        @ttl.datamovement()
        def dm_read():
            r = cb.reserve()
            tx = ttl.copy(inp[0:1, 0:2], r)
            tx.wait()
            r.push()

        @ttl.datamovement()
        def dm_write():
            for col in range(3):
                blk = out_cb.wait()
                ttl.copy(blk, out[0:1, 2 * col : 2 * col + 2]).wait()
                blk.pop()

    torch.manual_seed(424)
    inp_t = to_dram(torch.randn((TILE, 2 * TILE), dtype=torch.bfloat16), device)
    out_t = to_dram(torch.full((TILE, 6 * TILE), -42.0, dtype=torch.bfloat16), device)
    repro(inp_t, out_t)
    ttnn.synchronize_device(device)
    inp_h = ttnn.to_torch(inp_t)
    out_h = ttnn.to_torch(out_t)
    for col in range(3):
        col_slice = out_h[:, 2 * TILE * col : 2 * TILE * (col + 1)]
        assert torch.equal(col_slice, inp_h), f"output block {col} differs from input"


# ---------------------------------------------------------------------------
# Two deferred waits where t1 has fan-out (used twice) before the
# auto-pop pop point. After coalescing, replaceAllUsesWith must update
# every t1 use, not just one.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_deferred_waits_with_t1_fanout(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            with cb.reserve() as v:
                v.store(ttl.block.fill(5.0, shape=v.shape))
            with cb.reserve() as v:
                v.store(ttl.block.fill(7.0, shape=v.shape))

            t1 = cb.wait()
            t2 = cb.wait()
            with out_cb.reserve() as o:
                o.store(t1)
            with out_cb.reserve() as o:
                o.store(t1)
            with out_cb.reserve() as o:
                o.store(t2)

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

    _run(device, repro, 3, [5.0, 5.0, 7.0])


# ---------------------------------------------------------------------------
# Matmul-style pattern: 2 waits on cb_a interleaved with 2 waits on cb_b
# (a1, b1, a2, b2). Each CB has its own pair of deferred consumes, but
# the source pairs them across CBs. cb_a's two waits coalesce
# independently of cb_b's two waits.
# ---------------------------------------------------------------------------


@pytest.mark.requires_device
def test_matmul_style_two_cb_interleaved_deferred_acquires(device):
    @ttl.operation(grid=(1, 1))
    def repro(out):
        cb_a = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        cb_b = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with cb_a.reserve() as v:
                v.store(ttl.block.fill(11.0, shape=v.shape))
            with cb_a.reserve() as v:
                v.store(ttl.block.fill(22.0, shape=v.shape))
            with cb_b.reserve() as v:
                v.store(ttl.block.fill(33.0, shape=v.shape))
            with cb_b.reserve() as v:
                v.store(ttl.block.fill(44.0, shape=v.shape))

            a1 = cb_a.wait()
            b1 = cb_b.wait()
            a2 = cb_a.wait()
            b2 = cb_b.wait()
            with out_cb.reserve() as o:
                o.store(a1 + b1)
            with out_cb.reserve() as o:
                o.store(a2 + b2)

        @ttl.datamovement()
        def dm_read():
            pass

        @ttl.datamovement()
        def dm_write():
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 0]).wait()
            blk = out_cb.wait()
            ttl.copy(blk, out[0, 1]).wait()

    _run(device, repro, 2, [11.0 + 33.0, 22.0 + 44.0])
