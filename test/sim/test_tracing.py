# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the simulator tracing system.

Tests cover:
- Enabling tracing via SimulatorConfig
- Correct events emitted for operation, kernel, dfb, and copy categories
- Inclusive (--trace-events) and exclusive (--no-trace-events) filtering
- JSONL output via ttlang-sim CLI
- CLI validation (mutual exclusivity, requires --trace)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from python.sim import ttl, ttnn
from python.sim.context import get_context, reset_context
from python.sim.trace import ALL_CATEGORIES


def _run_simple_kernel_with_tracing(
    trace_set: frozenset[str] | None = None,
) -> list[dict]:
    """Run a minimal single-node kernel with tracing enabled.

    Args:
        trace_set: Categories to record. Defaults to ALL_CATEGORIES.

    Returns the collected trace events as a list of dicts (matching TraceEvent fields).
    """
    ctx = get_context()
    ctx.config.trace_set = trace_set if trace_set is not None else ALL_CATEGORIES

    inp = ttnn.rand((32, 32))
    out = ttnn.empty((32, 32))

    @ttl.operation(grid=(1, 1))
    def trace_kernel(a: ttnn.Tensor, o: ttnn.Tensor):
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                out_blk.store(blk + blk)

        @ttl.datamovement()
        def dm_read():
            with dfb.reserve() as blk:
                tx = ttl.copy(a[0, 0], blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, o[0, 0])
                tx.wait()

    trace_kernel(inp, out)

    return [
        {"event": ev.event, "tick": ev.tick, "kernel": ev.kernel, **ev.data}
        for ev in ctx.trace_events
    ]


def _run_pipe_kernel_with_tracing(
    trace_set: frozenset[str] | None = None,
) -> list[dict]:
    """Run a minimal 2-node kernel that transfers one tile via a pipe.

    Node (0,0) reads a tile from inp and sends it through a pipe to node (0,1).
    Node (0,1) receives the tile from the pipe and writes it to out.

    Args:
        trace_set: Categories to record. Defaults to ALL_CATEGORIES.

    Returns the collected trace events as a list of dicts.
    """
    ctx = get_context()
    ctx.config.trace_set = trace_set if trace_set is not None else ALL_CATEGORIES

    inp = ttnn.rand((32, 32))
    out = ttnn.empty((32, 32))

    @ttl.operation(grid=(1, 2))
    def pipe_kernel(a: ttnn.Tensor, o: ttnn.Tensor):
        pipe = ttl.Pipe((0, 0), (0, 1))
        pipe_net = ttl.PipeNet([pipe])
        dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
        out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

        @ttl.compute()
        def compute():
            if not pipe.has_current_node():
                return
            with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                out_blk.store(blk)

        @ttl.datamovement()
        def dm_read():
            if not pipe.has_current_node():
                return
            with dfb.reserve() as blk:

                def pipe_src(pipe_id):
                    tx = ttl.copy(a[0, 0], blk)
                    tx.wait()
                    tx2 = ttl.copy(blk, pipe_id)
                    tx2.wait()

                def pipe_dst(pipe_id):
                    tx = ttl.copy(pipe_id, blk)
                    tx.wait()

                pipe_net.if_src(pipe_src)
                pipe_net.if_dst(pipe_dst)

        @ttl.datamovement()
        def dm_write():
            if not pipe.has_current_node():
                return
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, o[0, 0])
                tx.wait()

    pipe_kernel(inp, out)

    return [
        {"event": ev.event, "tick": ev.tick, "kernel": ev.kernel, **ev.data}
        for ev in ctx.trace_events
    ]


class TestTraceEventTypes:
    """Verify that the correct events are emitted for each category."""

    def test_operation_events_emitted(self) -> None:
        """operation_start and operation_end are recorded."""
        events = _run_simple_kernel_with_tracing(frozenset({"operation"}))
        event_names = [e["event"] for e in events]
        assert "operation_start" in event_names
        assert "operation_end" in event_names

    def test_operation_events_carry_node(self) -> None:
        """operation_start / operation_end carry a node field."""
        events = _run_simple_kernel_with_tracing(frozenset({"operation"}))
        for ev in events:
            if ev["event"] in ("operation_start", "operation_end"):
                assert "node" in ev, f"Missing 'node' in {ev}"

    def test_kernel_lifecycle_events_emitted(self) -> None:
        """kernel_start and kernel_end are recorded for each kernel."""
        events = _run_simple_kernel_with_tracing(frozenset({"kernel"}))
        starts = [e for e in events if e["event"] == "kernel_start"]
        ends = [e for e in events if e["event"] == "kernel_end"]
        # There are 3 kernels: compute, dm_read, dm_write
        assert len(starts) == 3
        assert len(ends) == 3

    def test_kernel_block_unblock_pairs(self) -> None:
        """Every kernel_block is paired with a subsequent kernel_unblock for the same kernel."""
        events = _run_simple_kernel_with_tracing(frozenset({"kernel"}))
        blocks = [e for e in events if e["event"] == "kernel_block"]
        unblocks = [e for e in events if e["event"] == "kernel_unblock"]
        assert len(blocks) > 0, "Expected at least one blocking event"
        assert len(blocks) == len(unblocks)

    def test_dfb_events_emitted(self) -> None:
        """DFB reserve/push/wait/pop events are emitted."""
        events = _run_simple_kernel_with_tracing(frozenset({"dfb"}))
        event_names = {e["event"] for e in events}
        assert "dfb_reserve_begin" in event_names
        assert "dfb_reserve_end" in event_names
        assert "dfb_push" in event_names
        assert "dfb_wait_begin" in event_names
        assert "dfb_wait_end" in event_names
        assert "dfb_pop" in event_names

    def test_dfb_events_carry_dfb_name(self) -> None:
        """Every DFB event carries a 'dfb' field."""
        events = _run_simple_kernel_with_tracing(frozenset({"dfb"}))
        dfb_event_names = {
            "dfb_reserve_begin",
            "dfb_reserve_end",
            "dfb_push",
            "dfb_wait_begin",
            "dfb_wait_end",
            "dfb_pop",
        }
        for ev in events:
            if ev["event"] in dfb_event_names:
                assert "dfb" in ev, f"Missing 'dfb' field in {ev}"

    def test_copy_events_emitted(self) -> None:
        """copy_start and copy_end events are emitted."""
        events = _run_simple_kernel_with_tracing(frozenset({"copy"}))
        event_names = {e["event"] for e in events}
        assert "copy_start" in event_names
        assert "copy_end" in event_names

    def test_copy_events_carry_src_dst_types(self) -> None:
        """copy_start / copy_end carry src and dst type names."""
        events = _run_simple_kernel_with_tracing(frozenset({"copy"}))
        for ev in events:
            if ev["event"] in ("copy_start", "copy_end"):
                assert "src" in ev, f"Missing 'src' in {ev}"
                assert "dst" in ev, f"Missing 'dst' in {ev}"

    def test_ticks_are_non_negative_integers(self) -> None:
        """Every event has a tick >= 0."""
        events = _run_simple_kernel_with_tracing()
        for ev in events:
            assert isinstance(ev["tick"], int)
            assert ev["tick"] >= 0

    def test_pipe_events_emitted(self) -> None:
        """pipe_send and pipe_recv are emitted when a pipe transfers data."""
        events = _run_pipe_kernel_with_tracing(frozenset({"pipe"}))
        event_names = {e["event"] for e in events}
        assert "pipe_send" in event_names
        assert "pipe_recv" in event_names

    def test_pipe_events_carry_pipe_and_tiles(self) -> None:
        """pipe_send and pipe_recv carry 'pipe' and 'tiles' fields with tile count > 0."""
        events = _run_pipe_kernel_with_tracing(frozenset({"pipe"}))
        for ev in events:
            if ev["event"] in ("pipe_send", "pipe_recv"):
                assert "pipe" in ev, f"Missing 'pipe' field in {ev}"
                assert "tiles" in ev, f"Missing 'tiles' field in {ev}"
                assert ev["tiles"] > 0, f"Expected tiles > 0 in {ev}"

    def test_no_events_when_tracing_disabled(self) -> None:
        """No trace events are recorded when trace_set is empty."""
        ctx = get_context()
        ctx.config.trace_set = frozenset()

        inp = ttnn.rand((32, 32))
        out = ttnn.empty((32, 32))

        @ttl.operation(grid=(1, 1))
        def noop_kernel(a: ttnn.Tensor, o: ttnn.Tensor):
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                with dfb.wait() as blk, out_dfb.reserve() as out_blk:
                    out_blk.store(blk + blk)

            @ttl.datamovement()
            def dm_read():
                with dfb.reserve() as blk:
                    tx = ttl.copy(a[0, 0], blk)
                    tx.wait()

            @ttl.datamovement()
            def dm_write():
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, o[0, 0])
                    tx.wait()

        noop_kernel(inp, out)
        assert len(ctx.trace_events) == 0


class TestFiltering:
    """Verify inclusive and exclusive category filtering."""

    def test_inclusive_filter_only_returns_requested_categories(self) -> None:
        """With trace_set={'dfb'}, only dfb events appear."""
        events = _run_simple_kernel_with_tracing(frozenset({"dfb"}))
        dfb_events = {
            "dfb_reserve_begin",
            "dfb_reserve_end",
            "dfb_push",
            "dfb_wait_begin",
            "dfb_wait_end",
            "dfb_pop",
        }
        for ev in events:
            assert ev["event"] in dfb_events, f"Unexpected event: {ev['event']}"

    def test_exclusive_filter_suppresses_requested_categories(self) -> None:
        """With ALL_CATEGORIES minus 'dfb', no dfb events appear."""
        events = _run_simple_kernel_with_tracing(ALL_CATEGORIES - frozenset({"dfb"}))
        dfb_events = {
            "dfb_reserve_begin",
            "dfb_reserve_end",
            "dfb_push",
            "dfb_wait_begin",
            "dfb_wait_end",
            "dfb_pop",
        }
        for ev in events:
            assert ev["event"] not in dfb_events, f"Unexpected dfb event: {ev['event']}"

    def test_no_filter_returns_all_categories(self) -> None:
        """With no filter, all event categories appear."""
        events = _run_simple_kernel_with_tracing()
        names = {e["event"] for e in events}
        assert names & {"operation_start", "operation_end"}
        assert names & {"kernel_start", "kernel_end"}
        assert names & {"dfb_reserve_begin", "dfb_push"}
        assert names & {"copy_start", "copy_end"}


class TestCLITracing:
    """Tests for --trace, --trace-events, and --no-trace-events CLI flags."""

    def _make_script(self) -> Path:
        """Create a minimal runnable kernel script."""
        content = """\
import sys
sys.path.insert(0, "python")
import torch
import ttl
import ttnn

@ttl.operation(grid=(1, 1))
def kernel(a: ttnn.Tensor, o: ttnn.Tensor):
    dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with dfb.wait() as blk, out_dfb.reserve() as out_blk:
            out_blk.store(blk + blk)

    @ttl.datamovement()
    def dm_read():
        with dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, o[0, 0])
            tx.wait()

a = ttnn.rand((32, 32))
o = ttnn.empty((32, 32))
kernel(a, o)
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def _run_sim(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        script = self._make_script()
        try:
            return subprocess.run(
                [sys.executable, "-m", "sim.ttlang_sim", *extra_args, str(script)],
                cwd=Path(__file__).parent.parent.parent,
                env={**os.environ, "PYTHONPATH": "python"},
                capture_output=True,
                text=True,
            )
        finally:
            script.unlink()

    def test_trace_writes_jsonl_file(self) -> None:
        """--trace FILE writes a JSON Lines file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            trace_path = Path(tmp.name)
        try:
            result = self._run_sim("--trace", str(trace_path))
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert trace_path.exists()
            lines = trace_path.read_text().splitlines()
            assert len(lines) > 0
            for line in lines:
                record = json.loads(line)
                assert "event" in record
                assert "tick" in record
                assert "kernel" in record
        finally:
            trace_path.unlink(missing_ok=True)

    def test_trace_events_inclusive_filter(self) -> None:
        """--trace-events dfb writes only dfb events."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            trace_path = Path(tmp.name)
        try:
            result = self._run_sim("--trace", str(trace_path), "--trace-events", "dfb")
            assert result.returncode == 0, f"stderr: {result.stderr}"
            lines = trace_path.read_text().splitlines()
            assert len(lines) > 0
            dfb_events = {
                "dfb_reserve_begin",
                "dfb_reserve_end",
                "dfb_push",
                "dfb_wait_begin",
                "dfb_wait_end",
                "dfb_pop",
            }
            for line in lines:
                record = json.loads(line)
                assert record["event"] in dfb_events, f"Unexpected: {record['event']}"
        finally:
            trace_path.unlink(missing_ok=True)

    def test_no_trace_events_exclusive_filter(self) -> None:
        """--no-trace-events dfb suppresses dfb events."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            trace_path = Path(tmp.name)
        try:
            result = self._run_sim(
                "--trace", str(trace_path), "--no-trace-events", "dfb"
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            lines = trace_path.read_text().splitlines()
            assert len(lines) > 0
            dfb_events = {
                "dfb_reserve_begin",
                "dfb_reserve_end",
                "dfb_push",
                "dfb_wait_begin",
                "dfb_wait_end",
                "dfb_pop",
            }
            for line in lines:
                record = json.loads(line)
                assert (
                    record["event"] not in dfb_events
                ), f"DFB event should be suppressed: {record['event']}"
        finally:
            trace_path.unlink(missing_ok=True)

    def test_trace_events_without_trace_errors(self) -> None:
        """--trace-events without --trace exits with error."""
        result = self._run_sim("--trace-events", "dfb")
        assert result.returncode != 0
        assert "--trace-events requires --trace" in result.stderr

    def test_no_trace_events_without_trace_errors(self) -> None:
        """--no-trace-events without --trace exits with error."""
        result = self._run_sim("--no-trace-events", "dfb")
        assert result.returncode != 0
        assert "--no-trace-events requires --trace" in result.stderr

    def test_mutual_exclusivity_errors(self) -> None:
        """Specifying both --trace-events and --no-trace-events exits with error."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            trace_path = Path(tmp.name)
        try:
            result = self._run_sim(
                "--trace",
                str(trace_path),
                "--trace-events",
                "dfb",
                "--no-trace-events",
                "copy",
            )
            assert result.returncode != 0
            assert "mutually exclusive" in result.stderr
        finally:
            trace_path.unlink(missing_ok=True)

    def test_unknown_category_errors(self) -> None:
        """An unknown category name exits with error."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            trace_path = Path(tmp.name)
        try:
            result = self._run_sim(
                "--trace", str(trace_path), "--trace-events", "invalid_cat"
            )
            assert result.returncode != 0
            assert "Unknown trace categories" in result.stderr
        finally:
            trace_path.unlink(missing_ok=True)
