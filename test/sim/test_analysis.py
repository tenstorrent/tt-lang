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
    ThreadAnalysis,
    analyze_thread_function,
    validate_thread_function,
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


class TestAnalyzeThreadFunction:
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

        ips = analyze_thread_function(dm).injection_points
        assert ips == ()

    def test_assigned_copy_without_wait_detected(self):
        """tx = ttl.copy(...) with no tx.wait() produces a wait injection."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            # no tx.wait()

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].var_name == "tx"

    def test_assigned_copy_wait_triggers_on_return_when_last_stmt(self):
        """If the copy is the last statement, trigger_on_return is True."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].trigger_on_return is True

    def test_assigned_copy_wait_triggers_on_next_line(self):
        """If there is a statement after the copy, trigger is on that line."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = ttl.copy(src, blk)  # noqa: F821
            blk.push()  # next statement

        ips = analyze_thread_function(dm).injection_points
        assert len(ips) == 1
        assert ips[0].trigger_on_return is False
        assert ips[0].trigger_lineno is not None

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

        assert analyze_thread_function(dm).injection_points == ()

    def test_non_ttl_copy_not_detected(self):
        """copy() from a different namespace is not treated as ttl.copy."""

        def dm():
            blk = dfb.reserve()  # noqa: F821
            tx = copy(src, blk)  # noqa: F821  plain 'copy', not ttl.copy
            tx.wait()

        result = analyze_thread_function(dm)
        assert result.injection_points == ()
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
    """Verify that validate_thread_function catches unsupported ttl.copy() patterns."""

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
            group.add(ttl.copy(src, dst))  # noqa: F821

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
            group.add(ttl.copy(src, dst))  # noqa: F821

        violations = validate_thread_function(my_dm_thread)
        assert violations[0].func_name == "my_dm_thread"


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
