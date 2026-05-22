# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic copy-wait insertion (analysis.py).

Tests cover:
- AST analysis: InjectionPoint detection for reserve/wait without explicit release
- Explicit push/pop preserved (no double-insert)
- copy -> tx.wait() one-hop use detection
- Context-manager (with) acquires are skipped
- Runtime: kernel runs correctly without any explicit push/pop calls
- Runtime: explicit push/pop still works alongside auto-insertion
- Runtime: sequential reserve then wait on same DFB (the deadlock scenario)
- Runtime: complex control flow (nested loops, if-inside-for, issue #536 pattern)
"""

import pytest

from sim import ttl, ttnn
from sim.analysis import (
    InjectionPoint,
    PatternViolation,
    KernelAnalysis,
    analyze_kernel_function,
    collect_reachable_analyses,
    validate_kernel_function,
)
from sim.context import get_context, reset_context


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


class TestAnalyzeKernelFunction:
    """Verify InjectionPoint detection from function source.

    Push/pop injection is now handled directly by DataflowBuffer.reserve() and
    DataflowBuffer.wait() at runtime; AST analysis only generates 'wait'
    injection points for unwaited ttl.copy() calls.
    """

    def test_explicit_push_suppresses_injection(self):
        """When an explicit push() is present no injection point is generated."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(data, blk).wait()  # noqa: F821
            blk.push()

        ips = analyze_kernel_function(dm).injection_points
        assert ips == ()

    def test_explicit_pop_suppresses_injection(self):
        """When an explicit pop() is present no injection point is generated."""

        def compute():
            blk = dfb.wait()  # noqa: F821
            _ = blk + blk  # noqa: F821
            blk.pop()

        ips = analyze_kernel_function(compute).injection_points
        assert ips == ()

    def test_with_acquire_skipped(self):
        """Acquires inside with-statements are already handled by __exit__."""

        def dm():
            with dfb.reserve() as blk:  # noqa: F821
                ttl.copy(data, blk).wait()  # noqa: F821

        ips = analyze_kernel_function(dm).injection_points
        assert ips == ()


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
        """Producer reserves, writes, then consumer waits on the same DFB across kernels.

        This is the critical deadlock scenario: push must fire BEFORE the
        subsequent wait, not at end of scope.
        """
        inp = ttnn.rand((32, 32))
        out = ttnn.empty((32, 32))

        @ttl.operation(grid=(1, 1))
        def op(a, o):
            # Single shared DFB used by all three kernels.
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

    def test_multi_iteration_loop_copy_wait(self):
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
# Deadlock-resolution tests
# ---------------------------------------------------------------------------


class TestDeadlockResolution:
    """Verify that auto-injection resolves scenarios that would otherwise deadlock."""

    def test_sequential_reserves_same_dfb_single_pass(self, reset_simulator_context):
        """Two sequential reserve() calls on the same DFB in a single pass (no loop).

        dm_read reserves blk1 then blk2 from dfb_in sequentially without a loop.
        Without auto-injection, blk1 is never pushed before blk2 = dfb_in.reserve()
        is called, which blocks forever when block_count == 1.
        With auto-injection the scope boundary (blk2 = dfb_in.reserve()) triggers
        push(blk1) before the second reserve, allowing the pipeline to drain cleanly.
        """
        import torch

        inp = ttnn.rand((32, 32))
        out = ttnn.empty((2 * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb_in = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            dfb_out = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Consume two blocks in sequence.
                blk1 = dfb_in.wait()
                o1 = dfb_out.reserve()
                o1.store(blk1)
                blk2 = dfb_in.wait()
                o2 = dfb_out.reserve()
                o2.store(blk2)

            @ttl.datamovement()
            def dm_read():
                blk1 = dfb_in.reserve()  # first reserve
                ttl.copy(inp[0, 0], blk1).wait()
                # push(blk1) auto-injected at the blk2 = dfb_in.reserve() line below
                blk2 = dfb_in.reserve()  # second reserve on same DFB
                ttl.copy(inp[0, 0], blk2).wait()
                # push(blk2) auto-injected on return

            @ttl.datamovement()
            def dm_write():
                for i in range(2):
                    blk = dfb_out.wait()
                    ttl.copy(blk, out[i, 0]).wait()

        op(inp, out)
        inp_t = ttnn.to_torch(inp).float()
        out_t = ttnn.to_torch(out).float()
        # Both output rows should match the input (pass-through compute).
        assert torch.allclose(out_t[0:32], inp_t, atol=1e-2)
        assert torch.allclose(out_t[32:64], inp_t, atol=1e-2)


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

        ips = analyze_kernel_function(dm).injection_points
        assert ips == ()

    def test_wait_before_assignment_does_not_suppress_injection(self):
        """tx.wait() appearing before tx = ttl.copy(...) must not suppress injection.

        In loop bodies a wait at the top of the iteration releases the previous
        iteration's copy; the new copy at the bottom still needs auto-injection.
        The copy is the last statement in the flat list, so the trigger must be
        on function return (not on the preceding tx.wait() line).
        """

        def dm():
            for _i in range(2):  # noqa: F821
                tx.wait()  # noqa: F821 — releases previous iteration's copy
                blk = dfb.reserve()  # noqa: F821
                tx = ttl.copy(src, blk)  # noqa: F821 — new copy, must be injected

        ips = analyze_kernel_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "tx"
        # The copy is the last statement in source order; no later line exists to
        # trigger on, so the injection fires at function return.
        assert ips[0].trigger_on_return is True

    def test_var_name_reused_for_dfb_then_copy(self):
        """tx = dfb.reserve(); ...; tx = ttl.copy(...) — the DFB assignment must not
        suppress or confuse the copy-wait injection for the later copy assignment.

        The copy is the last statement so the trigger must be on function return,
        confirming the injection point corresponds to the ttl.copy() line and not
        to the earlier dfb.reserve() assignment.
        """

        def dm():
            tx = dfb.reserve()  # noqa: F821 — DFB block, not a copy
            tx.store(src)  # noqa: F821
            tx = ttl.copy(
                src, blk
            )  # noqa: F821 — name reused for a copy, needs injection

        ips = analyze_kernel_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "tx"
        assert ips[0].trigger_on_return is True

    def test_assigned_copy_without_wait_detected(self):
        """tx = ttl.copy(...) with no tx.wait() produces a wait injection on return."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            # no tx.wait()

        ips = analyze_kernel_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "tx"
        assert ips[0].trigger_on_return is True

    def test_copy_in_outer_loop_triggers_at_inner_for_not_inside_it(self):
        """A copy in the outer loop body must trigger at the inner for statement.

        The trigger must not bleed into the inner loop body — it fires at the
        line of the inner for statement (the next statement after the copy), not
        at any line inside the inner loop.
        """
        import inspect

        def dm():
            for _i in range(2):  # noqa: F821
                tx = ttl.copy(
                    src, blk
                )  # noqa: F821 -- outer copy; trigger should be the inner for
                for _j in range(3):  # noqa: F821
                    pass

        analysis = analyze_kernel_function(dm)
        assert len(analysis.injection_points) == 1
        ip = analysis.injection_points[0]
        assert ip.var_name == "tx"
        assert ip.trigger_on_return is False

        # The trigger line must be the inner `for` statement, not any line inside it.
        src_lines, start = inspect.getsourcelines(dm)
        # Find the line numbers of the inner for and the pass inside it.
        inner_for_lineno = next(
            start + i for i, ln in enumerate(src_lines) if "for _j" in ln
        )
        pass_lineno = next(
            start + i for i, ln in enumerate(src_lines) if ln.strip() == "pass"
        )
        assert ip.trigger_lineno == inner_for_lineno
        assert ip.trigger_lineno != pass_lineno

    def test_assigned_copy_wait_triggers_on_return_when_last_stmt(self):
        """If the copy is the last statement, trigger_on_return is True."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821

        ips = analyze_kernel_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "tx"
        assert ips[0].trigger_on_return is True

    def test_assigned_copy_wait_triggers_on_next_line(self):
        """If there is a statement after the copy, trigger is on that exact line."""
        import inspect

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            blk.push()  # next statement — trigger must land here

        ips = analyze_kernel_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "tx"
        assert ips[0].trigger_on_return is False

        src_lines, start = inspect.getsourcelines(dm)
        push_lineno = next(
            start + i for i, ln in enumerate(src_lines) if "blk.push()" in ln
        )
        assert ips[0].trigger_lineno == push_lineno

    def test_bare_copy_lineno_detected(self):
        """Bare ttl.copy(...) call (no assignment) records the absolute lineno."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk)  # noqa: F821  bare call — Case A

        result = analyze_kernel_function(dm)
        assert len(result.bare_copy_linenos) == 1

    def test_bare_copy_not_in_injection_points(self):
        """Bare copy is handled via auto_wait_copy_lines, not injection points."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk)  # noqa: F821

        assert analyze_kernel_function(dm).injection_points == ()

    def test_non_ttl_copy_not_detected(self):
        """copy() from a different namespace is not treated as ttl.copy."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = copy(src, blk)  # noqa: F821  plain 'copy', not ttl.copy
            tx.wait()

        result = analyze_kernel_function(dm)
        assert result.injection_points == ()
        assert result.bare_copy_linenos == frozenset()

    def test_chained_wait_produces_no_injection(self):
        """ttl.copy(...).wait() is already waited inline — no injection needed.

        The chained form is an ast.Expr whose value is the .wait() Call, not
        the ttl.copy() Call, so _is_ttl_copy_call returns False and the
        statement is not recorded as either a bare copy or an assigned copy.
        """

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk).wait()  # noqa: F821  chained wait

        result = analyze_kernel_function(dm)
        assert result.injection_points == ()
        assert result.bare_copy_linenos == frozenset()

    def test_copy_inside_if_triggers_at_post_if_statement(self):
        """A copy inside an if block triggers at the first statement after the if.

        _all_stmts_flat flattens the if body into the overall statement list, so
        the next statement in source order after the copy is the post-if statement.
        """
        import inspect

        def dm():
            if cond:  # noqa: F821
                tx = ttl.copy(src, blk)  # noqa: F821
            post_if_call()  # noqa: F821

        analysis = analyze_kernel_function(dm)
        assert len(analysis.injection_points) == 1
        ip = analysis.injection_points[0]
        assert ip.var_name == "tx"
        assert ip.trigger_on_return is False

        src_lines, start = inspect.getsourcelines(dm)
        post_if_lineno = next(
            start + i for i, ln in enumerate(src_lines) if "post_if_call" in ln
        )
        assert ip.trigger_lineno == post_if_lineno

    def test_same_variable_two_copies_both_get_injection(self):
        """tx reused for two successive copies: both copies get independent injection points.

        The first copy's trigger is the line of the second tx = ttl.copy(...);
        at that point tx still holds the first CopyTransaction so the wait fires
        on the right object.  The second copy's trigger is the following statement.
        """
        import inspect

        def dm():
            tx = ttl.copy(src, blk)  # noqa: F821  first copy
            tx = ttl.copy(src, blk)  # noqa: F821  second copy
            done()  # noqa: F821

        analysis = analyze_kernel_function(dm)
        assert len(analysis.injection_points) == 2

        src_lines, start = inspect.getsourcelines(dm)
        second_copy_lineno = next(
            start + i for i, ln in enumerate(src_lines) if "second copy" in ln
        )
        done_lineno = next(
            start + i for i, ln in enumerate(src_lines) if "done()" in ln
        )

        trigger_linenos = {ip.trigger_lineno for ip in analysis.injection_points}
        assert second_copy_lineno in trigger_linenos
        assert done_lineno in trigger_linenos


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

    def test_multiline_bare_copy_auto_waited(self, reset_simulator_context):
        """A bare ttl.copy() call spanning multiple lines is correctly auto-waited.

        The AST records stmt.lineno as the first line of the call expression
        (the 'ttl.copy(' line).  copy.py reads frame.f_lineno from the calling
        frame, which Python also sets to the first line of the call expression.
        If these diverge the auto-wait does not fire and the kernel deadlocks.
        """
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                o = out_dfb.reserve()
                o.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()
                ttl.copy(  # multi-line bare call — auto-waited on the first line
                    inp[0, 0],
                    blk,
                )

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        op(inp, out)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())

    def test_kernel_exit_cleanup_does_not_push_other_kernels_block(
        self, reset_simulator_context
    ):
        """dm_write's exit cleanup must not push a block that dm_read reserved.

        The scenario that exercises the greenlet-identity guard in
        DataflowBuffer.auto_push_block() (dfb.py line ~1335):

        1. dm_read reserves a block from inp_dfb (pending on dm_read's greenlet).
        2. dm_read yields by blocking on sync_dfb.wait() — dm_read holds the
           pending block while suspended.
        3. dm_write runs, produces a sync token to sync_dfb, and exits.
        4. dm_write's _tagged cleanup calls inp_dfb.auto_push_block().
           The guard ``if self._pending_reserved_greenlet is not getcurrent()``
           must detect the mismatch and skip the push.
        5. dm_read unblocks, fills the block, and pushes it explicitly.

        Without the guard, step 4 would push an unfilled block to compute,
        producing zeroes instead of the expected ones in the output.
        """
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))
        # Scratch tensor absorbs the dummy sync copy; its final value is irrelevant.
        scratch = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out, scratch):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            # sync_dfb carries one token from dm_write to dm_read, forcing
            # dm_read to yield while its inp_dfb block is still pending.
            sync_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()  # pending on dm_read's greenlet

                # Yield here — dm_write runs and EXITS while blk is still pending.
                # dm_write's cleanup calls inp_dfb.auto_push_block(); the greenlet
                # guard must prevent it from pushing blk.
                sig = sync_dfb.wait()

                # The block state machine requires copy-out before auto-pop; write the
                # token value to scratch (the content is irrelevant for this test).
                ttl.copy(sig, scratch[0, 0]).wait()

                # Fill and push blk now that the sync signal has been received.
                ttl.copy(inp[0, 0], blk).wait()
                blk.push()

                # Read compute's output so the pipeline drains cleanly.
                ob = out_dfb.wait()
                ttl.copy(ob, out[0, 0]).wait()

            @ttl.datamovement()
            def dm_write():
                # Produce a sync token to unblock dm_read, then exit immediately.
                # auto-push fires for the sync block at dm_write's return.
                #
                # The _tagged cleanup then calls inp_dfb.auto_push_block() while
                # dm_read's block is still pending — exactly the scenario under test.
                s = sync_dfb.reserve()
                ttl.copy(inp[0, 0], s).wait()

        op(inp, out, scratch)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())


# ---------------------------------------------------------------------------
# Unit tests: AST analysis — inline DFB acquires
# ---------------------------------------------------------------------------


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


class TestValidateKernelFunction:
    """Verify that validate_kernel_function catches unsupported ttl.copy() patterns."""

    def test_copy_passed_to_function_is_violation(self):
        """ttl.copy() nested inside another function call is flagged."""

        def dm():
            group.add(ttl.copy(src, dst))  # noqa: F821

        violations = validate_kernel_function(dm)
        assert len(violations) == 1
        assert "ttl.copy()" in violations[0].message

    def test_bare_copy_is_ok(self):
        """Bare ttl.copy(src, dst) is a supported pattern; no violation."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            ttl.copy(src, blk)  # noqa: F821

        assert validate_kernel_function(dm) == []

    def test_assigned_copy_is_ok(self):
        """tx = ttl.copy(src, dst) is a supported pattern; no violation."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            tx.wait()  # noqa: F821

        assert validate_kernel_function(dm) == []

    def test_violation_contains_source_location(self):
        """PatternViolation has a valid source file and line number."""

        def dm():
            group.add(ttl.copy(src, dst))  # noqa: F821

        violations = validate_kernel_function(dm)
        assert len(violations) == 1
        v = violations[0]
        assert v.source_file.endswith(".py")
        assert v.lineno > 0
        assert v.col > 0
        assert v.func_name == "dm"

    def test_func_name_in_violation(self):
        """PatternViolation.func_name matches the kernel function name."""

        def my_dm_kernel():
            group.add(ttl.copy(src, dst))  # noqa: F821

        violations = validate_kernel_function(my_dm_kernel)
        assert violations[0].func_name == "my_dm_kernel"

    def test_method_chain_wait_is_ok(self):
        """ttl.copy(src, dst).wait() is a supported pattern; no violation."""

        def dm():
            ttl.copy(src, dst).wait()  # noqa: F821

        assert validate_kernel_function(dm) == []

    def test_method_chain_non_wait_is_violation(self):
        """ttl.copy(src, dst).foo() is not a supported pattern and must be flagged.

        Only .wait() is a valid method chain on a copy result; any other
        attribute call (e.g. .foo()) must produce a PatternViolation.
        """

        def dm():
            ttl.copy(src, dst).foo()  # noqa: F821

        violations = validate_kernel_function(dm)
        assert len(violations) == 1
        assert "ttl.copy()" in violations[0].message


# ---------------------------------------------------------------------------
# Complex control-flow tests (issue #536 and related patterns)
# ---------------------------------------------------------------------------


class TestComplexControlFlow:
    """Auto-injection with nested loops, conditionals, and the #536 pop-hoisting pattern.

    Issue #536 describes a compiler bug where auto-inserted cb_pop_front calls
    are hoisted past subsequent cb_wait_front calls on the same DFB, causing
    the read pointer to never advance.  The tests here verify that the simulator
    correctly interleaves push/pop with the surrounding control flow.
    """

    # ------------------------------------------------------------------
    # Issue #536: two consecutive wait() calls on the same DFB (no loop)
    # ------------------------------------------------------------------

    def test_sequential_waits_same_dfb_runtime(self, reset_simulator_context):
        """Two consecutive wait() calls on the same DFB produce distinct values (#536).

        The producer fills two slots with distinct values (7.0 and 8.0).  The
        second ``out_cb.wait()`` auto-pops blk1 before acquiring blk2, so the
        read pointer advances and the consumer sees [7.0, 8.0] rather than
        stalling at slot 0.
        """
        import torch

        TILE = 32
        out = ttnn.empty((2 * TILE, TILE))

        @ttl.operation(grid=(1, 1))
        def op(out):
            out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                # Fill slot 0 with 7.0, slot 1 with 8.0.
                with out_cb.reserve() as v:
                    v.store(ttl.math.fill(v, 7.0))
                with out_cb.reserve() as v:
                    v.store(ttl.math.fill(v, 8.0))

            @ttl.datamovement()
            def dm_read():
                pass

            @ttl.datamovement()
            def dm_write():
                # Consume both slots; pop(blk1) must fire before second wait().
                blk1 = out_cb.wait()
                ttl.copy(blk1, out[0, 0]).wait()
                blk2 = out_cb.wait()
                ttl.copy(blk2, out[1, 0]).wait()

        op(out)
        out_t = ttnn.to_torch(out).float()
        assert torch.allclose(
            out_t[0:TILE], torch.full((TILE, TILE), 7.0), atol=1e-2
        ), f"Slot 0 expected 7.0, got {out_t[0, 0].item()}"
        assert torch.allclose(
            out_t[TILE:], torch.full((TILE, TILE), 8.0), atol=1e-2
        ), f"Slot 1 expected 8.0, got {out_t[TILE, 0].item()}"

    # ------------------------------------------------------------------
    # Nested for loops
    # ------------------------------------------------------------------

    def test_nested_for_loop_runtime(self, reset_simulator_context):
        """Auto push/pop fires correctly at each inner-loop iteration.

        The outer loop runs OUTER times; the inner loop runs INNER times per
        outer iteration, producing one block each.  The total block count is
        OUTER * INNER.  Every block must be pushed and popped in order.
        """
        import torch

        OUTER = 2
        INNER = 3
        TOTAL = OUTER * INNER
        inp = ttnn.rand((TOTAL * 32, 32))
        out = ttnn.empty((TOTAL * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for _ in range(TOTAL):
                    blk = dfb.wait()
                    ob = out_dfb.reserve()
                    ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                for _i in range(OUTER):
                    for _j in range(INNER):
                        blk = dfb.reserve()
                        idx = _i * INNER + _j
                        ttl.copy(inp[idx, 0], blk).wait()
                        # auto push before next inner iteration

            @ttl.datamovement()
            def dm_write():
                for i in range(TOTAL):
                    blk = out_dfb.wait()
                    ttl.copy(blk, out[i, 0]).wait()

        op(inp, out)
        assert torch.allclose(
            ttnn.to_torch(inp).float(), ttnn.to_torch(out).float(), atol=1e-2
        )

    def test_nested_for_loop_sequential_waits(self, reset_simulator_context):
        """Two nested loops each doing wait/copy on the same DFB advance the pointer.

        Outer produces N*M blocks; the inner consumer loop consumes each block
        immediately, so pops must interleave with the inner-loop waits.
        """
        import torch

        OUTER = 2
        INNER = 2
        TOTAL = OUTER * INNER
        inp = ttnn.rand((TOTAL * 32, 32))
        out = ttnn.empty((TOTAL * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for _ in range(TOTAL):
                    src = dfb.wait()
                    dst = out_dfb.reserve()
                    dst.store(src)

            @ttl.datamovement()
            def dm_read():
                for i in range(TOTAL):
                    blk = dfb.reserve()
                    ttl.copy(inp[i, 0], blk).wait()

            @ttl.datamovement()
            def dm_write():
                for _i in range(OUTER):
                    for _j in range(INNER):
                        idx = _i * INNER + _j
                        blk = out_dfb.wait()
                        ttl.copy(blk, out[idx, 0]).wait()
                        # auto pop before next inner-loop wait

        op(inp, out)
        assert torch.allclose(
            ttnn.to_torch(inp).float(), ttnn.to_torch(out).float(), atol=1e-2
        )

    # ------------------------------------------------------------------
    # if inside for
    # ------------------------------------------------------------------

    def test_if_inside_for_runtime(self, reset_simulator_context):
        """Conditional reserve inside a loop: push fires even when the if branch is not taken.

        dm_read iterates 2*ITERS times but only reserves inside an ``if i % 2 == 0``
        guard, producing ITERS blocks total.  On the odd iterations the LINE callback
        for the reserve line does not fire, so the auto-push for the block from the
        previous even iteration is deferred until the NEXT even iteration's reserve
        line (or function return for the last block).  The pipeline must drain cleanly.
        """
        import torch

        ITERS = 3  # blocks produced; loop runs 2*ITERS times
        inp = ttnn.rand((ITERS * 32, 32))
        out = ttnn.empty((ITERS * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for _ in range(ITERS):
                    blk = dfb.wait()
                    ob = out_dfb.reserve()
                    ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                for i in range(ITERS * 2):
                    if i % 2 == 0:
                        blk = dfb.reserve()
                        ttl.copy(inp[i // 2, 0], blk).wait()
                        # auto push deferred to next even iteration (or return)

            @ttl.datamovement()
            def dm_write():
                for i in range(ITERS):
                    blk = out_dfb.wait()
                    ttl.copy(blk, out[i, 0]).wait()

        op(inp, out)
        assert torch.allclose(
            ttnn.to_torch(inp).float(), ttnn.to_torch(out).float(), atol=1e-2
        )

    # ------------------------------------------------------------------
    # post-loop trigger (code after for loop)
    # ------------------------------------------------------------------

    def test_post_loop_trigger_fires_at_post_loop_code_runtime(
        self, reset_simulator_context
    ):
        """Kernel with a producing loop followed by unrelated code runs cleanly.

        The final-iteration push must fire at the post-loop statement, not at
        function return, so the consumer can drain before the producer returns.
        """
        import torch

        ITERS = 3
        inp = ttnn.rand((ITERS * 32, 32))
        out = ttnn.empty((ITERS * 32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                for _ in range(ITERS):
                    blk = dfb.wait()
                    ob = out_dfb.reserve()
                    ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                for i in range(ITERS):
                    blk = dfb.reserve()
                    ttl.copy(inp[i, 0], blk).wait()
                # Post-loop code: the final auto-push should have fired
                # at this point so the consumer is not blocked.
                _ = 0  # noqa: F841

            @ttl.datamovement()
            def dm_write():
                for i in range(ITERS):
                    blk = out_dfb.wait()
                    ttl.copy(blk, out[i, 0]).wait()

        op(inp, out)
        assert torch.allclose(
            ttnn.to_torch(inp).float(), ttnn.to_torch(out).float(), atol=1e-2
        )


# ---------------------------------------------------------------------------
# Unit tests: collect_reachable_analyses (nested defs + module-scope helpers)
# ---------------------------------------------------------------------------


def _module_scope_copy_helper(src, dst):  # noqa: F821
    """Module-scope helper containing a bare ttl.copy() call."""
    ttl.copy(src, dst)  # noqa: F821


def _module_scope_assigned_copy_helper(src, dst):  # noqa: F821
    """Module-scope helper with an assigned ttl.copy() (no explicit wait)."""
    tx = ttl.copy(src, dst)  # noqa: F821
    _ = tx  # noqa: F841


def _transitive_h1(src, dst):  # noqa: F821
    """Calls _transitive_h2; no direct ttl.copy()."""
    _transitive_h2(src, dst)  # noqa: F821


def _transitive_h2(src, dst):  # noqa: F821
    """Leaf helper with a bare ttl.copy(); discovered transitively via _transitive_h1."""
    ttl.copy(src, dst)  # noqa: F821


def _recursive_a(src, dst):  # noqa: F821
    """Part of a mutually-recursive pair; calls _recursive_b."""
    _recursive_b(src, dst)  # noqa: F821


def _recursive_b(src, dst):  # noqa: F821
    """Part of a mutually-recursive pair; calls _recursive_a back."""
    _recursive_a(src, dst)  # noqa: F821
    ttl.copy(src, dst)  # noqa: F821


class TestCollectReachableAnalyses:
    """collect_reachable_analyses discovers and analyses helper functions.

    Tests verify that bare / assigned ttl.copy() calls inside nested defs
    and module-scope helpers are picked up by the recursive analysis so that
    their code objects receive injection hooks.
    """

    def test_top_level_function_included(self):
        """The kernel function itself is always present in the result."""

        def dm():
            ttl.copy(src, dst)  # noqa: F821

        result = collect_reachable_analyses(dm)
        assert dm.__code__ in result

    def test_nested_def_bare_copy_detected(self):
        """A bare ttl.copy() inside a nested def gets an injection entry."""

        def dm():
            def helper():
                ttl.copy(src, dst)  # noqa: F821

            helper()  # noqa: F821

        result = collect_reachable_analyses(dm)
        nested_codes = [code for code in result if code.co_name == "helper"]
        assert len(nested_codes) == 1
        assert len(result[nested_codes[0]].bare_copy_linenos) == 1

    def test_nested_def_assigned_copy_detected(self):
        """An assigned tx = ttl.copy() inside a nested def gets an InjectionPoint."""

        def dm():
            def pipe_src(blk):
                tx = ttl.copy(src, blk)  # noqa: F821
                blk.push()  # noqa: F821

            pipe_src(blk)  # noqa: F821

        result = collect_reachable_analyses(dm)
        nested_codes = [code for code in result if code.co_name == "pipe_src"]
        assert len(nested_codes) == 1
        nested_analysis = result[nested_codes[0]]
        assert len(nested_analysis.injection_points) == 1
        assert nested_analysis.injection_points[0].var_name == "tx"

    def test_module_scope_helper_detected(self):
        """A bare ttl.copy() in a module-scope helper called by name is detected."""

        def dm():
            _module_scope_copy_helper(src, dst)  # noqa: F821

        result = collect_reachable_analyses(dm)
        helper_codes = [
            code for code in result if code.co_name == "_module_scope_copy_helper"
        ]
        assert len(helper_codes) == 1
        assert len(result[helper_codes[0]].bare_copy_linenos) == 1

    def test_shared_visited_prevents_duplicate_analysis(self):
        """Passing a shared visited set skips functions already analysed."""

        def dm():
            _module_scope_copy_helper(src, dst)  # noqa: F821

        visited: set[int] = set()
        first = collect_reachable_analyses(dm, visited)
        assert _module_scope_copy_helper.__code__ in first

        second = collect_reachable_analyses(dm, visited)
        assert second == {}

    def test_explicit_wait_in_nested_def_suppresses_injection(self):
        """A nested def with tx.wait() must not generate an injection point."""

        def dm():
            def helper():
                tx = ttl.copy(src, dst)  # noqa: F821
                tx.wait()  # noqa: F821

            helper()  # noqa: F821

        result = collect_reachable_analyses(dm)
        nested_codes = [code for code in result if code.co_name == "helper"]
        assert len(nested_codes) == 1
        assert len(result[nested_codes[0]].injection_points) == 0

    def test_deeply_nested_def_detected(self):
        """A bare ttl.copy() two levels deep (def inside def) is discovered."""

        def dm():
            def outer():
                def inner():
                    ttl.copy(src, dst)  # noqa: F821

                inner()  # noqa: F821

            outer()  # noqa: F821

        result = collect_reachable_analyses(dm)
        inner_codes = [code for code in result if code.co_name == "inner"]
        assert len(inner_codes) == 1
        assert len(result[inner_codes[0]].bare_copy_linenos) == 1

    def test_if_src_if_dst_both_detected(self):
        """Both pipe_src and pipe_dst nested defs are discovered and analysed.

        Mirrors the real-world if_src/if_dst pipe pattern where two nested
        functions each contain a ttl.copy() call.
        """

        def dm():
            def pipe_src(pipe_id):
                ttl.copy(src, pipe_id)  # noqa: F821

            def pipe_dst(pipe_id):
                ttl.copy(pipe_id, dst)  # noqa: F821

            pipe_src(x)  # noqa: F821
            pipe_dst(x)  # noqa: F821

        result = collect_reachable_analyses(dm)
        src_codes = [code for code in result if code.co_name == "pipe_src"]
        dst_codes = [code for code in result if code.co_name == "pipe_dst"]
        assert len(src_codes) == 1
        assert len(dst_codes) == 1
        assert len(result[src_codes[0]].bare_copy_linenos) == 1
        assert len(result[dst_codes[0]].bare_copy_linenos) == 1

    def test_transitive_module_scope_discovered(self):
        """A helper called by a helper (two hops) is discovered transitively."""

        def dm():
            _transitive_h1(src, dst)  # noqa: F821

        result = collect_reachable_analyses(dm)
        h2_codes = [code for code in result if code.co_name == "_transitive_h2"]
        assert len(h2_codes) == 1
        assert len(result[h2_codes[0]].bare_copy_linenos) == 1

    def test_module_scope_assigned_copy_detected(self):
        """An assigned tx = ttl.copy() in a module-scope helper gets an InjectionPoint."""

        def dm():
            _module_scope_assigned_copy_helper(src, dst)  # noqa: F821

        result = collect_reachable_analyses(dm)
        helper_codes = [
            code
            for code in result
            if code.co_name == "_module_scope_assigned_copy_helper"
        ]
        assert len(helper_codes) == 1
        assert len(result[helper_codes[0]].injection_points) == 1
        assert result[helper_codes[0]].injection_points[0].var_name == "tx"

    def test_mutually_recursive_helpers_terminate(self):
        """Mutually recursive module-scope helpers do not cause infinite recursion."""

        def dm():
            _recursive_a(src, dst)  # noqa: F821

        result = collect_reachable_analyses(dm)
        names = {code.co_name for code in result}
        assert "_recursive_a" in names
        assert "_recursive_b" in names


# ---------------------------------------------------------------------------
# Runtime tests: nested defs and module-scope helpers
# ---------------------------------------------------------------------------


class TestNestedDefCopyWaitRuntime:
    """End-to-end tests for copy-wait injection into nested def helpers."""

    def test_nested_def_bare_copy_runs_without_explicit_wait(
        self, reset_simulator_context
    ):
        """A bare ttl.copy() inside a nested def is auto-waited at runtime."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                def do_copy(src_tile, dst_block):
                    ttl.copy(src_tile, dst_block)  # bare, no wait

                blk = inp_dfb.reserve()
                do_copy(inp[0, 0], blk)

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        op(inp, out)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())

    def test_module_scope_helper_bare_copy_runs_without_explicit_wait(
        self, reset_simulator_context
    ):
        """A bare ttl.copy() in a module-scope helper is auto-waited at runtime."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()
                _module_scope_copy_helper(inp[0, 0], blk)

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        op(inp, out)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())

    def test_two_nested_defs_both_auto_waited(self, reset_simulator_context):
        """Two nested defs (if_src / if_dst style) each with a bare ttl.copy() run correctly.

        Both code objects must receive injection hooks; if either is missed the
        copy would not be waited and the kernel would deadlock or corrupt data.
        """
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                def do_reserve(target):
                    ttl.copy(inp[0, 0], target)  # bare — auto-waited

                def do_nothing():
                    pass  # second nested def with no copy; exercises multi-def discovery

                blk = inp_dfb.reserve()
                do_reserve(blk)
                do_nothing()

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        op(inp, out)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())

    def test_transitive_module_scope_helper_runtime(self, reset_simulator_context):
        """A bare ttl.copy() two hops away (dm -> h1 -> h2) is auto-waited at runtime."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()
                _transitive_h1(inp[0, 0], blk)

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                ttl.copy(blk, out[0, 0]).wait()

        op(inp, out)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())

    def test_shared_module_scope_helper_used_by_both_dm_kernels(
        self, reset_simulator_context
    ):
        """A module-scope helper shared by dm0 and dm1 is installed once and runs correctly."""
        import torch

        inp = ttnn.from_torch(torch.ones(32, 32))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(1, 1))
        def op(inp, out):
            inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                blk = inp_dfb.wait()
                ob = out_dfb.reserve()
                ob.store(blk)

            @ttl.datamovement()
            def dm_read():
                blk = inp_dfb.reserve()
                _module_scope_copy_helper(inp[0, 0], blk)

            @ttl.datamovement()
            def dm_write():
                blk = out_dfb.wait()
                _module_scope_copy_helper(blk, out[0, 0])

        op(inp, out)
        assert torch.allclose(ttnn.to_torch(out).float(), torch.ones(32, 32).float())
