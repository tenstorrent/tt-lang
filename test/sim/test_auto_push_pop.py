# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic push/pop insertion (auto_push_pop.py).

Tests cover:
- AST analysis: InjectionPoint detection for reserve/wait without explicit release
- Explicit push/pop preserved (no double-insert)
- copy -> tx.wait() one-hop use detection
- Context-manager (with) acquires are skipped
- Runtime: kernel runs correctly without any explicit push/pop calls
- Runtime: explicit push/pop still works alongside auto-insertion
- Runtime: sequential reserve then wait on same DFB (the deadlock scenario)
"""

import pytest

from python.sim import ttl, ttnn
from python.sim.auto_push_pop import (
    InjectionPoint,
    PatternViolation,
    ThreadAnalysis,
    analyze_thread_function,
    validate_thread_function,
)
from python.sim.context import get_context, reset_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset():
    reset_context()
    yield
    reset_context()


# ---------------------------------------------------------------------------
# Unit tests: AST analysis
# ---------------------------------------------------------------------------


class TestAnalyzeThreadFunction:
    """Verify InjectionPoint detection from function source."""

    def test_reserve_without_push_detected(self):
        """A bare reserve() with no push() produces a push injection point."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(data, blk).wait()  # noqa: F821

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "blk"
        assert ips[0].action == "push"

    def test_wait_without_pop_detected(self):
        """A bare wait() with no pop() produces a pop injection point."""

        def compute():
            blk = dfb.wait()  # noqa: F821
            result = blk + blk  # noqa: F821

        ips = analyze_thread_function(compute).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "blk"
        assert ips[0].action == "pop"

    def test_explicit_push_suppresses_injection(self):
        """When an explicit push() is present no injection point is generated."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(data, blk).wait()  # noqa: F821
            blk.push()

        ips = analyze_thread_function(dm).injection_points
        assert ips == ()

    def test_explicit_pop_suppresses_injection(self):
        """When an explicit pop() is present no injection point is generated."""

        def compute():
            blk = dfb.wait()  # noqa: F821
            _ = blk + blk  # noqa: F821
            blk.pop()

        ips = analyze_thread_function(compute).injection_points
        assert ips == ()

    def test_with_acquire_skipped(self):
        """Acquires inside with-statements are already handled by __exit__."""

        def dm():
            with dfb.reserve() as blk:  # noqa: F821
                ttl.copy(data, blk).wait()  # noqa: F821

        ips = analyze_thread_function(dm).injection_points
        assert ips == ()

    def test_multiple_acquires_without_release(self):
        """Two sequential acquires on different DFBs each get an injection point."""

        def dm():
            a = a_dfb.reserve()  # noqa: F821
            ttl.copy(src, a).wait()  # noqa: F821
            b = b_dfb.reserve()  # noqa: F821
            ttl.copy(src2, b).wait()  # noqa: F821

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 2
        names = {ip.var_name for ip in ips}
        assert names == {"a", "b"}

    def test_scope_boundary_separates_sequential_reserves(self):
        """The first reserve's injection fires before the second reserve."""

        def dm():
            a = dfb.reserve()  # noqa: F821
            ttl.copy(src, a).wait()  # noqa: F821
            b = dfb.reserve()  # noqa: F821  <- scope boundary for a
            ttl.copy(src, b).wait()  # noqa: F821

        ips = analyze_thread_function(dm).injection_points
        # a -> inject at dfb.reserve() (scope boundary)
        # b -> inject on return (last reserve, no boundary)
        a_ip = next(ip for ip in ips if ip.var_name == "a")
        b_ip = next(ip for ip in ips if ip.var_name == "b")
        assert not a_ip.trigger_on_return
        assert b_ip.trigger_on_return

    def test_trigger_on_return_when_last_reserve(self):
        """Last acquire in function triggers on return, not on a line."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(data, blk).wait()  # noqa: F821

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].trigger_on_return

    def test_copy_hop_detects_tx_wait(self):
        """tx = ttl.copy(src, blk); tx.wait() — last use is tx.wait(), not copy."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821  <- direct use of blk
            tx.wait()  # noqa: F821                 <- one-hop use

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].action == "push"
        assert ips[0].trigger_on_return  # tx.wait() is the last stmt -> return

    def test_trigger_after_last_use_not_at_boundary(self):
        """Trigger placed after last use, not at the conservative scope boundary.

        blk is last used at ttl.copy().wait(); the next reserve is two lines
        later with idle code in between.  The trigger should fire on the line
        immediately after tx.wait(), not at the next reserve line.
        """
        import inspect

        def dm():
            blk = dfb.reserve()  # noqa: F821  <- line A
            tx = ttl.copy(src, blk)  # noqa: F821
            tx.wait()  # noqa: F821             <- last use of blk
            x = 1 + 1  # noqa: F841            <- trigger should be HERE
            blk2 = dfb.reserve()  # noqa: F821 <- scope boundary (conservative)
            ttl.copy(src, blk2).wait()  # noqa: F821

        src_lines, _ = inspect.getsourcelines(dm)
        # Locate line offsets within the function source.
        last_use_offset = next(i for i, l in enumerate(src_lines) if "tx.wait()" in l)
        trigger_offset = last_use_offset + 1  # line immediately after last use
        boundary_offset = next(
            i for i, l in enumerate(src_lines) if "blk2 = dfb.reserve()" in l
        )
        # The trigger must be earlier than the boundary.
        assert trigger_offset < boundary_offset

        ips = analyze_thread_function(dm).injection_points
        blk_ip = next(ip for ip in ips if ip.var_name == "blk")
        assert not blk_ip.trigger_on_return
        # Trigger line is not the scope boundary — it is before it.
        blk2_ip = next(ip for ip in ips if ip.var_name == "blk2")
        assert blk2_ip.trigger_on_return  # blk2 is last, no subsequent acquire
        assert blk_ip.trigger_lineno is not None
        assert blk2_ip.trigger_lineno is None

    def test_trigger_after_tx_wait_not_at_copy_line(self):
        """For a two-step copy, the trigger fires after tx.wait(), not at ttl.copy().

        If copy-handle tracking is broken, ``tx = ttl.copy(src, blk)`` would
        be the last seen use of ``blk`` and the trigger would land on the
        ``tx.wait()`` line (firing before the copy completes).  With correct
        tracking ``tx.wait()`` is the last use, so the trigger lands on the
        statement after it.
        """

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821  <- blk loaded here
            tx.wait()  # noqa: F821                 <- real last use via handle
            blk2 = dfb.reserve()  # noqa: F821      <- scope boundary

        import inspect

        src_lines, start = inspect.getsourcelines(dm)
        copy_lineno = start + next(
            i for i, l in enumerate(src_lines) if "ttl.copy" in l
        )
        tx_wait_lineno = start + next(
            i for i, l in enumerate(src_lines) if "tx.wait()" in l
        )

        ips = analyze_thread_function(dm).injection_points
        blk_ip = next(ip for ip in ips if ip.var_name == "blk")
        assert not blk_ip.trigger_on_return
        assert blk_ip.trigger_lineno is not None
        # Trigger must be strictly after tx.wait() — not at or before it.
        # (If copy-handle tracking were broken the trigger would be copy_lineno+1
        # which equals tx_wait_lineno.)
        assert blk_ip.trigger_lineno > tx_wait_lineno
        assert blk_ip.trigger_lineno > copy_lineno


# ---------------------------------------------------------------------------
# Integration tests: runtime behaviour
# ---------------------------------------------------------------------------


def _run_kernel(op_fn, inp, out):
    """Run a ttl.operation that takes inp and out, return (exit_code, output)."""
    op_fn(inp, out)


class TestRuntimeAutoPushPop:
    """Verify that kernels run correctly with auto push/pop."""

    def test_kernel_without_push_pop_succeeds(self):
        """A kernel that omits push/pop should complete without errors."""
        inp = ttnn.rand((32, 32))
        out = ttnn.empty((32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

            @ttl.compute()
            def compute():
                blk = dfb.wait()
                out_blk = out_dfb.reserve()
                out_blk.store(blk + blk)
                # no pop / push

            @ttl.datamovement()
            def dm_read():
                blk = dfb.reserve()
                tx = ttl.copy(a[0, 0], blk)
                tx.wait()
                # no push

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                tx = ttl.copy(blk, o[0, 0])
                tx.wait()
                # no pop

        op(inp, out)  # Should not raise

    def test_kernel_result_correct_without_push_pop(self):
        """Auto-inserted push/pop produce the correct output."""
        import torch

        inp = ttnn.rand((32, 32))
        out = ttnn.empty((32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

            @ttl.compute()
            def compute():
                blk = dfb.wait()
                out_blk = out_dfb.reserve()
                out_blk.store(blk)  # passthrough

            @ttl.datamovement()
            def dm_read():
                blk = dfb.reserve()
                ttl.copy(a[0, 0], blk).wait()

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, o[0, 0]).wait()

        op(inp, out)
        assert torch.allclose(
            ttnn.to_torch(inp).float(), ttnn.to_torch(out).float(), atol=1e-2
        )

    def test_explicit_push_pop_not_double_fired(self):
        """Explicit push/pop are preserved and no double push/pop occurs."""
        inp = ttnn.rand((32, 32))
        out = ttnn.empty((32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

            @ttl.compute()
            def compute():
                blk = dfb.wait()
                out_blk = out_dfb.reserve()
                out_blk.store(blk)
                blk.pop()
                out_blk.push()

            @ttl.datamovement()
            def dm_read():
                blk = dfb.reserve()
                ttl.copy(a[0, 0], blk).wait()
                blk.push()

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, o[0, 0]).wait()
                blk.pop()

        op(inp, out)  # Should not raise (no double push/pop)

    def test_sequential_reserve_then_wait_same_dfb(self):
        """Producer reserves, writes, then consumer waits on the same DFB thread.

        This is the critical deadlock scenario: push must fire BEFORE the
        subsequent wait, not at end of scope.
        """
        inp = ttnn.rand((32, 32))
        out = ttnn.empty((32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            # Single shared DFB used by all three threads.
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

            @ttl.compute()
            def compute():
                # reserve then wait on different DFBs — no cross-DFB deadlock here
                in_blk = dfb.wait()
                out_blk = out_dfb.reserve()
                out_blk.store(in_blk)
                # auto pop / push

            @ttl.datamovement()
            def dm_read():
                blk = dfb.reserve()
                ttl.copy(a[0, 0], blk).wait()
                # auto push -> must fire before any wait on dfb

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, o[0, 0]).wait()
                # auto pop

        op(inp, out)  # must not deadlock

    def test_multi_iteration_loop_auto_push_pop(self):
        """Auto push/pop fires correctly on every iteration of a loop."""
        import torch

        ITERS = 3
        inp = ttnn.rand((ITERS * 32, 32))
        out = ttnn.empty((ITERS * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for _ in range(ITERS):
                    blk = dfb.wait()
                    out_blk = out_dfb.reserve()
                    out_blk.store(blk)
                    # auto pop / push — must fire each iteration

            @ttl.datamovement()
            def dm_read():
                for i in range(ITERS):
                    blk = dfb.reserve()
                    ttl.copy(a[i, 0], blk).wait()
                    # auto push each iteration

            @ttl.datamovement()
            def dm_write():
                for i in range(ITERS):
                    blk = out_dfb.wait()
                    ttl.copy(blk, o[i, 0]).wait()
                    # auto pop each iteration

        op(inp, out)
        assert torch.allclose(
            ttnn.to_torch(inp).float(), ttnn.to_torch(out).float(), atol=1e-2
        )

    def test_sequential_reserves_same_dfb_in_loop(self):
        """Multiple reserves on the same DFB within one loop iteration.

        dm_read reserves blk1 then blk2 from the same DFB in each iteration.
        With block_count=2 there is only one free slot after blk1 is reserved,
        so the auto-push for blk1 must fire at the blk2.reserve() line (same
        iteration) rather than at the start of the next iteration.
        The auto-push for blk2 fires at the blk1.reserve() line next iteration.
        """
        import torch

        ITERS = 3
        # dm_read produces 2*ITERS blocks total.
        inp = ttnn.rand((ITERS * 32, 32))
        out = ttnn.empty((ITERS * 2 * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            # block_count=2: only two slots; correct ordering is mandatory.
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Consume 2*ITERS blocks from dfb, one at a time.
                for _ in range(ITERS * 2):
                    blk = dfb.wait()
                    ob = out_dfb.reserve()
                    ob.store(blk)
                    # auto pop/push each iteration

            @ttl.datamovement()
            def dm_read():
                for i in range(ITERS):
                    blk1 = dfb.reserve()
                    ttl.copy(a[i, 0], blk1).wait()
                    # auto push fires at blk2 = dfb.reserve() line (same iter)
                    blk2 = dfb.reserve()
                    ttl.copy(a[i, 0], blk2).wait()
                    # auto push fires at blk1 = dfb.reserve() next iteration

            @ttl.datamovement()
            def dm_write():
                for i in range(ITERS * 2):
                    blk = out_dfb.wait()
                    ttl.copy(blk, o[i, 0]).wait()
                    # auto pop each iteration

        op(inp, out)
        # Each output row is a copy of the corresponding input row (floor(i/2)).
        out_t = ttnn.to_torch(out).float()
        inp_t = ttnn.to_torch(inp).float()
        for i in range(ITERS * 2):
            assert torch.allclose(
                out_t[i * 32 : (i + 1) * 32],
                inp_t[(i // 2) * 32 : (i // 2 + 1) * 32],
                atol=1e-2,
            )


# ---------------------------------------------------------------------------
# Copy-wait tests
# ---------------------------------------------------------------------------


class TestCopyWaitAnalysis:
    """Verify AST detection of missing tx.wait() on ttl.copy() calls."""

    def test_assigned_copy_with_explicit_wait_not_detected(self):
        """tx = ttl.copy(...); tx.wait() — no injection needed."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            tx.wait()

        ips = analyze_thread_function(dm).injection_points
        # push for blk is auto-inserted; no extra wait injection
        wait_ips = [ip for ip in ips if ip.action == "wait"]
        assert wait_ips == []

    def test_assigned_copy_without_wait_detected(self):
        """tx = ttl.copy(...) with no tx.wait() produces a wait injection."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            # no tx.wait()

        ips = analyze_thread_function(dm).injection_points
        wait_ips = [ip for ip in ips if ip.action == "wait"]
        assert len(wait_ips) == 1
        assert wait_ips[0].var_name == "tx"

    def test_assigned_copy_wait_triggers_on_return_when_last_stmt(self):
        """If the copy is the last statement, trigger_on_return is True."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821

        result = analyze_thread_function(dm)
        wait_ips = [ip for ip in result.injection_points if ip.action == "wait"]
        assert len(wait_ips) == 1
        assert wait_ips[0].trigger_on_return is True

    def test_assigned_copy_wait_triggers_on_next_line(self):
        """If there is a statement after the copy, trigger is on that line."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            blk.push()  # next statement

        result = analyze_thread_function(dm)
        wait_ips = [ip for ip in result.injection_points if ip.action == "wait"]
        assert len(wait_ips) == 1
        assert wait_ips[0].trigger_on_return is False
        assert wait_ips[0].trigger_lineno is not None

    def test_bare_copy_lineno_detected(self):
        """Bare ttl.copy(...) call (no assignment) records the absolute lineno."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk)  # noqa: F821  bare call — Case A

        result = analyze_thread_function(dm)
        assert len(result.bare_copy_linenos) == 1

    def test_bare_copy_not_in_injection_points(self):
        """Bare copy is handled via auto_wait_copy_lines, not injection points."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk)  # noqa: F821

        result = analyze_thread_function(dm)
        wait_ips = [ip for ip in result.injection_points if ip.action == "wait"]
        assert wait_ips == []

    def test_non_ttl_copy_not_detected(self):
        """copy() from a different namespace is not treated as ttl.copy."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = copy(src, blk)  # noqa: F821  plain 'copy', not ttl.copy
            tx.wait()

        result = analyze_thread_function(dm)
        wait_ips = [ip for ip in result.injection_points if ip.action == "wait"]
        assert wait_ips == []
        assert result.bare_copy_linenos == frozenset()


class TestCopyWaitRuntime:
    """Verify that copy-wait auto-insertion allows kernels to run correctly."""

    def test_assigned_copy_without_wait_auto_waited(self, reset_simulator_context):
        """Kernel with tx = ttl.copy(...) (no tx.wait()) runs successfully."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )  # noqa: F841
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )  # noqa: F841

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()  # noqa: F821
                o = out_dfb.reserve()  # noqa: F821
                o.store(blk)  # noqa: F821

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()  # noqa: F821
                tx = ttl.copy(inp[0, 0], blk)  # noqa: F821
                # no tx.wait() — auto-waited by injection

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()  # noqa: F821
                tx = ttl.copy(blk, out[0, 0])  # noqa: F821
                # no tx.wait() — auto-waited by injection

        op(inp, out)
        result = ttnn.to_torch(out).float()
        assert torch.allclose(result, torch.ones(32, 32).float())

    def test_bare_copy_auto_waited(self, reset_simulator_context):
        """Kernel with bare ttl.copy(...) (no assignment) runs successfully."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )  # noqa: F841
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )  # noqa: F841

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()  # noqa: F821
                o = out_dfb.reserve()  # noqa: F821
                o.store(blk)  # noqa: F821

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()  # noqa: F821
                ttl.copy(inp[0, 0], blk)  # noqa: F821  bare — Case A

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()  # noqa: F821
                ttl.copy(blk, out[0, 0])  # noqa: F821  bare — Case A

        op(inp, out)
        result = ttnn.to_torch(out).float()
        assert torch.allclose(result, torch.ones(32, 32).float())


# ---------------------------------------------------------------------------
# Unit tests: AST analysis — inline DFB acquires
# ---------------------------------------------------------------------------


class TestInlineAcquireAnalysis:
    """Verify InjectionPoint detection for dfb.wait()/reserve() inline in ttl.copy()."""

    def test_inline_wait_in_copy_detected(self):
        """ttl.copy(dfb.wait(), dst) produces a pop_dfb injection point."""

        def dm():
            ttl.copy(out_dfb.wait(), dst)  # noqa: F821

        result = analyze_thread_function(dm)
        pop_ips = [ip for ip in result.injection_points if ip.action == "pop_dfb"]
        assert len(pop_ips) == 1
        assert pop_ips[0].var_name == "out_dfb"
        assert pop_ips[0].trigger_on_return is True

    def test_inline_reserve_in_copy_detected(self):
        """ttl.copy(src, dfb.reserve()) produces a push_dfb injection point."""

        def dm():
            ttl.copy(src, inp_dfb.reserve())  # noqa: F821

        result = analyze_thread_function(dm)
        push_ips = [ip for ip in result.injection_points if ip.action == "push_dfb"]
        assert len(push_ips) == 1
        assert push_ips[0].var_name == "inp_dfb"
        assert push_ips[0].trigger_on_return is True

    def test_inline_acquire_with_next_stmt_uses_scope_boundary(self):
        """With a second acquire on the same DFB, trigger is at the boundary line."""

        def dm():
            ttl.copy(out_dfb.wait(), dst[0])  # noqa: F821
            ttl.copy(out_dfb.wait(), dst[1])  # noqa: F821

        result = analyze_thread_function(dm)
        pop_ips = [ip for ip in result.injection_points if ip.action == "pop_dfb"]
        # First inline acquire: scope boundary is the second acquire's line.
        # Second inline acquire: no next acquire -> trigger_on_return.
        assert len(pop_ips) == 2
        first = min(pop_ips, key=lambda ip: ip.trigger_lineno or float("inf"))
        second = max(pop_ips, key=lambda ip: ip.trigger_lineno or float("inf"))
        assert first.trigger_on_return is False
        assert first.trigger_lineno is not None
        assert second.trigger_on_return is True

    def test_named_acquire_scope_constrained_by_inline(self):
        """Named acquire scope boundary is correctly constrained by inline acquire."""

        def dm():
            blk = out_dfb.wait()  # noqa: F821
            ttl.copy(out_dfb.wait(), dst)  # noqa: F821

        result = analyze_thread_function(dm)
        pop_ips = [ip for ip in result.injection_points if ip.action == "pop"]
        # The named acquire's scope boundary should be the inline acquire's line.
        assert len(pop_ips) == 1
        assert pop_ips[0].trigger_on_return is False

    def test_inline_in_assigned_copy_detected(self):
        """tx = ttl.copy(dfb.wait(), dst) produces both a wait and pop_dfb injection."""

        def dm():
            tx = ttl.copy(out_dfb.wait(), dst)  # noqa: F821

        result = analyze_thread_function(dm)
        wait_ips = [ip for ip in result.injection_points if ip.action == "wait"]
        pop_ips = [ip for ip in result.injection_points if ip.action == "pop_dfb"]
        assert len(wait_ips) == 1  # Case B: tx.wait() injection
        assert len(pop_ips) == 1  # Inline acquire: pop_dfb injection
        assert wait_ips[0].var_name == "tx"
        assert pop_ips[0].var_name == "out_dfb"


# ---------------------------------------------------------------------------
# Runtime tests: inline DFB acquires
# ---------------------------------------------------------------------------


class TestInlineAcquireRuntime:
    """Verify that kernels using inline dfb.wait()/reserve() in ttl.copy() run
    correctly with the auto pop_block()/push_block() injection."""

    def test_inline_wait_copy_pop_auto_injected(self, reset_simulator_context):
        """Kernel using ttl.copy(dfb.wait(), dst) in dm_write succeeds without explicit pop."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )  # noqa: F841
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )  # noqa: F841

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()  # noqa: F821
                o = out_dfb.reserve()  # noqa: F821
                o.store(blk)  # noqa: F821

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()  # noqa: F821
                tx = ttl.copy(inp[0, 0], blk)  # noqa: F821
                tx.wait()  # noqa: F821

            @ttl.datamovement()
            def dm_write():
                # Inline wait: dfb.wait() passed directly as copy src; pop auto-injected.
                ttl.copy(out_dfb.wait(), out[0, 0])  # noqa: F821 bare + inline wait

        op(inp, out)
        result = ttnn.to_torch(out).float()
        assert torch.allclose(result, torch.ones(32, 32).float())

    def test_inline_reserve_copy_push_auto_injected(self, reset_simulator_context):
        """Kernel using ttl.copy(src, dfb.reserve()) in dm_read succeeds without explicit push."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )  # noqa: F841
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )  # noqa: F841

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()  # noqa: F821
                o = out_dfb.reserve()  # noqa: F821
                o.store(blk)  # noqa: F821

            @ttl.datamovement()
            def dm_read():
                # Inline reserve: dfb.reserve() passed directly as copy dst; push auto-injected.
                ttl.copy(
                    inp[0, 0], inp_dfb.reserve()
                )  # noqa: F821 bare + inline reserve

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()  # noqa: F821
                tx = ttl.copy(blk, out[0, 0])  # noqa: F821
                tx.wait()  # noqa: F821

        op(inp, out)
        result = ttnn.to_torch(out).float()
        assert torch.allclose(result, torch.ones(32, 32).float())

    def test_full_inline_pipeline(self, reset_simulator_context):
        """Full pipeline: dm_read uses inline reserve, dm_write uses inline wait."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )  # noqa: F841
            out_dfb = ttl.make_dataflow_buffer_like(
                out, shape=(1, 1), block_count=2
            )  # noqa: F841

            @ttl.compute()
            def compute():
                blk = dfb.wait()  # noqa: F821
                o = out_dfb.reserve()  # noqa: F821
                o.store(blk)  # noqa: F821

            @ttl.datamovement()
            def dm_read():
                # Inline reserve — push auto-injected.
                ttl.copy(inp[0, 0], dfb.reserve())  # noqa: F821

            @ttl.datamovement()
            def dm_write():
                # Inline wait — pop auto-injected.
                ttl.copy(out_dfb.wait(), out[0, 0])  # noqa: F821

        op(inp, out)
        result = ttnn.to_torch(out).float()
        assert torch.allclose(result, torch.ones(32, 32).float())


# ---------------------------------------------------------------------------
# Unit tests: pattern validation
# ---------------------------------------------------------------------------


class TestValidateThreadFunction:
    """Verify that validate_thread_function catches unsupported patterns."""

    # ---- DFB acquire violations ----

    def test_bare_reserve_is_violation(self):
        """dfb.reserve() as a bare statement (return value discarded) is flagged."""

        def dm():
            dfb.reserve()  # noqa: F821  bare — discards the block

        violations = validate_thread_function(dm)
        assert len(violations) == 1
        assert "reserve()" in violations[0].message
        assert isinstance(violations[0], PatternViolation)

    def test_reserve_passed_to_function_is_violation(self):
        """dfb.reserve() passed to a non-ttl.copy() function is flagged."""

        def dm():
            some_func(dfb.reserve())  # noqa: F821

        violations = validate_thread_function(dm)
        assert len(violations) == 1
        assert "reserve()" in violations[0].message

    def test_wait_passed_to_function_is_violation(self):
        """dfb.wait() passed to a non-ttl.copy() function is flagged."""

        def dm():
            some_func(dfb.wait())  # noqa: F821

        violations = validate_thread_function(dm)
        assert len(violations) == 1
        assert "wait()" in violations[0].message

    def test_named_assign_reserve_is_ok(self):
        """blk = dfb.reserve() is a supported pattern; no violation."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk).wait()  # noqa: F821

        assert validate_thread_function(dm) == []

    def test_with_reserve_is_ok(self):
        """with dfb.reserve() as blk: is a supported pattern; no violation."""

        def dm():
            with dfb.reserve() as blk:  # noqa: F821
                ttl.copy(src, blk).wait()  # noqa: F821

        assert validate_thread_function(dm) == []

    def test_inline_reserve_in_copy_is_ok(self):
        """ttl.copy(src, dfb.reserve()) is a supported pattern; no violation."""

        def dm():
            ttl.copy(src, dfb.reserve())  # noqa: F821

        assert validate_thread_function(dm) == []

    def test_tx_wait_not_flagged(self):
        """tx.wait() (CopyTransaction) shares the wait() shape but is not flagged."""

        def dm():
            tx = ttl.copy(src, dst)  # noqa: F821
            tx.wait()  # noqa: F821

        assert validate_thread_function(dm) == []

    def test_multiple_violations_all_reported(self):
        """All unsupported sites in a single function are returned."""

        def dm():
            some_func(dfb.reserve())  # noqa: F821  violation 1
            some_other(dfb.wait())  # noqa: F821  violation 2

        violations = validate_thread_function(dm)
        assert len(violations) == 2

    # ---- ttl.copy() violations ----

    def test_copy_passed_to_function_is_violation(self):
        """ttl.copy() nested inside another function call is flagged."""

        def dm():
            group.add(ttl.copy(src, dst))  # noqa: F821

        violations = validate_thread_function(dm)
        assert len(violations) == 1
        assert "ttl.copy()" in violations[0].message

    def test_bare_copy_is_ok(self):
        """Bare ttl.copy(src, dst) is a supported pattern; no violation."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk)  # noqa: F821

        assert validate_thread_function(dm) == []

    def test_assigned_copy_is_ok(self):
        """tx = ttl.copy(src, dst) is a supported pattern; no violation."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            tx.wait()  # noqa: F821

        assert validate_thread_function(dm) == []

    def test_violation_contains_source_location(self):
        """PatternViolation has a valid source file and line number."""

        def dm():
            some_func(dfb.reserve())  # noqa: F821

        violations = validate_thread_function(dm)
        assert len(violations) == 1
        v = violations[0]
        assert v.source_file.endswith(".py")
        assert v.lineno > 0
        assert v.col > 0
        assert v.func_name == "dm"

    def test_func_name_in_violation(self):
        """PatternViolation.func_name matches the thread function name."""

        def my_dm_thread():
            some_func(dfb.reserve())  # noqa: F821

        violations = validate_thread_function(my_dm_thread)
        assert violations[0].func_name == "my_dm_thread"


class TestValidationRaisesAtRuntime:
    """Verify that kernels with unsupported patterns abort with diagnostics."""

    def test_unsupported_pattern_raises_runtime_error(self, reset_simulator_context):
        """A kernel with dfb.reserve() in an unsupported position raises RuntimeError."""

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb = ttl.make_dataflow_buffer_like(
                inp, shape=(1, 1), block_count=2
            )  # noqa: F841

            @ttl.compute()
            def compute():
                pass  # noqa: F821

            @ttl.datamovement()
            def dm_read():
                some_func(dfb.reserve())  # noqa: F821  unsupported

            @ttl.datamovement()
            def dm_write():
                pass  # noqa: F821

        inp = ttnn.from_torch(__import__("torch").ones(32, 32))
        out = ttnn.from_torch(__import__("torch").zeros(32, 32))

        with pytest.raises(RuntimeError, match="unsupported pattern"):
            op(inp, out)
