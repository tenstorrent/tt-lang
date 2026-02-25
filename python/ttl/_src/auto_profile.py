# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-profiling infrastructure for tt-lang kernels.

Enabled via TTLANG_AUTO_PROFILE=1 environment variable.
Automatically instruments every operation with signposts and generates
a visual profile report showing cycle counts per source line.
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Background colors for CB visualization (8 pastel colors, avoiding red/yellow)
    CB_BACKGROUNDS = [
        "\033[48;5;153m",  # Light steel blue
        "\033[48;5;158m",  # Pale turquoise
        "\033[48;5;183m",  # Lavender
        "\033[48;5;151m",  # Pale green
        "\033[48;5;181m",  # Light pink
        "\033[48;5;152m",  # Pale cyan
        "\033[48;5;187m",  # Wheat/tan
        "\033[48;5;146m",  # Light periwinkle
    ]

    @classmethod
    def cb_bg(cls, cb_index: int) -> str:
        """Get background color for a CB index, or empty if out of range."""
        if 0 <= cb_index < len(cls.CB_BACKGROUNDS):
            return cls.CB_BACKGROUNDS[cb_index]
        return ""


def is_auto_profile_enabled() -> bool:
    """Check if auto-profiling is enabled via environment variable."""
    return os.environ.get("TTLANG_AUTO_PROFILE", "0") == "1"


class SourceLineMapper:
    """Maps signpost markers back to source code lines."""

    def __init__(self):
        self.signpost_to_line: Dict[str, Tuple[int, str]] = {}
        self.source_lines: List[str] = []
        self.line_offset: int = 0

    def register_signpost(self, signpost_name: str, lineno: int, source: str):
        """Register a signpost with its source line information."""
        self.signpost_to_line[signpost_name] = (lineno, source)

    def set_source(self, source_lines: List[str]):
        """Set the source code lines for display."""
        self.source_lines = source_lines

    def get_line_info(self, signpost_name: str) -> Optional[Tuple[int, str]]:
        """Get line number and source for a signpost."""
        return self.signpost_to_line.get(signpost_name)


def parse_signpost_name(signpost: str) -> Tuple[Optional[str], bool]:
    """
    Parse op name and implicit flag from signpost name.

    Returns (op_name, is_implicit) where op_name is None for line-only signposts.
    Format: "<kernel>_L<lineno>[_[implicit_]<op>]"
    Examples:
      "compute_L52" -> (None, False)
      "dm_read_L52_cb_wait" -> ("cb_wait", False)
      "dm_write_L52_implicit_cb_pop" -> ("cb_pop", True)
    """
    import re

    m = re.search(r"_L\d+_", signpost)
    if m is None:
        # Line-only signpost: "<kernel>_L<num>"
        return None, False

    rest = signpost[m.end() :]  # e.g., "cb_wait" or "implicit_cb_pop"
    if rest.startswith("implicit_"):
        return rest[len("implicit_") :], True
    return rest, False


class ProfileResult:
    """Represents profiling results for a single signpost."""

    def __init__(
        self, signpost: str, thread: str, cycles: int, lineno: int, source: str
    ):
        self.signpost = signpost
        self.thread = thread
        self.cycles = cycles
        self.lineno = lineno
        self.source = source.strip()
        self.op_name, self.implicit = parse_signpost_name(signpost)


def generate_signpost_name(operation: str, lineno: int, col: int) -> Tuple[str, str]:
    """
    Generate before/after signpost names for an operation.

    Returns:
        Tuple of (before_name, after_name)
    """
    base = f"{operation}_L{lineno}_C{col}"
    return (f"{base}_before", f"{base}_after")


def parse_device_profile_csv(
    csv_path: Path, line_mapper: SourceLineMapper
) -> List[ProfileResult]:
    """
    Parse the device profile CSV and extract signpost timing data.

    Each scoped signpost in the CSV has a base name (e.g. "compute_L52")
    with ZONE_START/ZONE_END timestamps spanning the actual work.

    Args:
        csv_path: Path to profile_log_device.csv
        line_mapper: Mapper to correlate signposts to source lines

    Returns:
        List of ProfileResult objects sorted by line number
    """
    results = []
    signpost_starts = {}

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) < 13:
                continue

            try:
                timestamp = int(row[5])
            except (ValueError, IndexError):
                continue

            thread = row[3]
            signpost = row[10]
            zone_type = row[11]

            if not signpost:
                continue

            key = f"{thread}_{signpost}"

            if zone_type == "ZONE_START":
                signpost_starts[key] = timestamp
            elif zone_type == "ZONE_END" and key in signpost_starts:
                start_ts = signpost_starts[key]
                duration = timestamp - start_ts

                line_info = line_mapper.get_line_info(signpost)
                if line_info:
                    lineno, source = line_info
                    results.append(
                        ProfileResult(signpost, thread, duration, lineno, source)
                    )

                del signpost_starts[key]

    results.sort(key=lambda r: r.lineno)
    return results


def print_profile_report(
    results: List[ProfileResult],
    all_source_lines: Dict[str, List[str]],
    thread_to_kernel: Dict[str, str],
    line_mapper: Optional[SourceLineMapper] = None,
    cb_wait_to_dma: Optional[Dict[Tuple[str, int], Tuple[str, int, int, str]]] = None,
    dma_producer_to_cb: Optional[Dict[Tuple[str, int], int]] = None,
    kernel_line_offsets: Optional[Dict[str, int]] = None,
):
    """
    Print a profile report organized by thread.

    Shows full source context with cycle annotations where available.
    Each thread displays its corresponding kernel's source code.

    Args:
        results: List of ProfileResult from CSV parsing
        all_source_lines: Dict mapping kernel name to source lines
        thread_to_kernel: Dict mapping RISC thread name to kernel name
        line_mapper: Optional SourceLineMapper with line offset info
        cb_wait_to_dma: Optional mapping from (kernel, line) -> (dma_kernel, dma_line, cb_index)
        dma_producer_to_cb: Optional mapping from (kernel, line) -> cb_index for DMA producers
        kernel_line_offsets: Optional mapping from kernel name to line offset
    """
    if cb_wait_to_dma is None:
        cb_wait_to_dma = {}
    if dma_producer_to_cb is None:
        dma_producer_to_cb = {}
    if kernel_line_offsets is None:
        kernel_line_offsets = {}

    print()
    print("=" * 100)
    print("TTLANG AUTO-PROFILE REPORT")
    print("=" * 100)
    print()

    # Print CB color key
    active_cbs = set(dma_producer_to_cb.values()) | {
        info[2] for info in cb_wait_to_dma.values()
    }
    if active_cbs:
        print("CB Colors: ", end="")
        for cb_idx in sorted(active_cbs):
            bg = Colors.cb_bg(cb_idx)
            if bg:
                print(f"{bg} CB[{cb_idx}] {Colors.RESET} ", end="")
        print()
        print()

    thread_cycles = defaultdict(int)
    thread_ops = defaultdict(int)
    for r in results:
        thread_cycles[r.thread] += r.cycles
        thread_ops[r.thread] += 1

    total_cycles = max(thread_cycles.values()) if thread_cycles else 1

    print(f"Total operations: {len(results)}")
    print(f"Longest thread: {total_cycles:,} cycles")
    print()

    thread_to_results = defaultdict(list)
    for result in results:
        thread_to_results[result.thread].append(result)

    thread_order = ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
    sorted_threads = sorted(
        thread_to_results.keys(),
        key=lambda t: thread_order.index(t) if t in thread_order else 999,
    )

    for thread in sorted_threads:
        thread_results = sorted(thread_to_results[thread], key=lambda r: r.lineno)

        # Get the kernel name and source for this thread
        kernel_name = thread_to_kernel.get(thread, "")
        source_lines = all_source_lines.get(kernel_name, [])

        print("=" * 100)
        kernel_info = f" [{kernel_name}]" if kernel_name else ""
        print(
            f"THREAD: {thread:<10}{kernel_info} ({thread_ops[thread]} ops, "
            f"{thread_cycles[thread]:,} cycles, "
            f"{100.0 * thread_cycles[thread] / total_cycles:.1f}% of total)"
        )
        print("=" * 100)
        print()
        print(f"{'LINE':<6} {'%TIME':<7} {'CYCLES':<10} SOURCE")
        print(f"{'-'*6} {'-'*7} {'-'*10} {'-'*70}")

        source_groups = defaultdict(list)
        for result in thread_results:
            source_groups[result.source.strip()].append(result)

        all_cycle_counts = []
        for line_results in source_groups.values():
            total_for_line = sum(r.cycles for r in line_results)
            all_cycle_counts.append(total_for_line)

        all_cycle_counts.sort(reverse=True)
        hottest = all_cycle_counts[0] if len(all_cycle_counts) > 0 else 0
        second_hottest = all_cycle_counts[1] if len(all_cycle_counts) > 1 else 0

        if source_lines:
            line_offset = kernel_line_offsets.get(kernel_name, 0)

            for lineno in range(1, len(source_lines) + 1):
                file_lineno = lineno + line_offset
                source_line = source_lines[lineno - 1].rstrip()
                source_stripped = source_line.strip()

                if source_stripped in source_groups:
                    line_results = source_groups[source_stripped]
                    total_line_cycles = sum(r.cycles for r in line_results)

                    color = ""
                    if total_line_cycles >= hottest and hottest > 0:
                        color = Colors.RED
                    elif total_line_cycles >= second_hottest and second_hottest > 0:
                        color = Colors.YELLOW

                    # Check if this line is a DMA producer/consumer
                    original_lineno = line_results[0].lineno if line_results else -1
                    producer_cb_idx = dma_producer_to_cb.get(
                        (kernel_name, original_lineno)
                    )
                    consumer_dma_info = cb_wait_to_dma.get(
                        (kernel_name, original_lineno)
                    )
                    # Only producers get source line background (consumers only highlight remark)
                    cb_bg = (
                        Colors.cb_bg(producer_cb_idx)
                        if producer_cb_idx is not None
                        else ""
                    )

                    # Group by op_name to show breakdown
                    op_groups = defaultdict(list)
                    for r in line_results:
                        key = (r.op_name, r.implicit)
                        op_groups[key].append(r)

                    # Check if we have multiple distinct ops
                    has_named_ops = any(r.op_name for r in line_results)

                    if len(line_results) == 1 and not has_named_ops:
                        r = line_results[0]
                        pct = 100.0 * r.cycles / thread_cycles[thread]
                        source_colored = (
                            f"{cb_bg}{source_line}{Colors.RESET}"
                            if cb_bg
                            else source_line
                        )
                        print(
                            f"{color}{file_lineno:<6} {pct:>5.1f}%  "
                            f"{r.cycles:<10,} {source_colored}{Colors.RESET if color else ''}"
                        )
                        # Show remarks for consumer/producer lines
                        if consumer_dma_info or producer_cb_idx is not None:
                            indent = 27 + len(source_line)
                            if consumer_dma_info:
                                barrier_kernel, barrier_line, cb_idx, label = (
                                    consumer_dma_info
                                )
                                dma_cb_bg = Colors.cb_bg(cb_idx)
                                remark = f"waiting for {label} @ line {barrier_line} ({barrier_kernel})"
                                if dma_cb_bg:
                                    remark = f"{dma_cb_bg}{remark}{Colors.RESET}"
                                is_last = producer_cb_idx is None
                                arrow = "╰─" if is_last else "├─"
                                print(
                                    f"{Colors.DIM}{' ' * indent}"
                                    f"{arrow} {remark}{Colors.RESET}"
                                )
                            if producer_cb_idx is not None:
                                producer_bg = Colors.cb_bg(producer_cb_idx)
                                remark = f"DMA barrier for CB[{producer_cb_idx}]"
                                if producer_bg:
                                    remark = f"{producer_bg}{remark}{Colors.RESET}"
                                print(
                                    f"{Colors.DIM}{' ' * indent}"
                                    f"╰─ {remark}{Colors.RESET}"
                                )
                    elif has_named_ops:
                        # Show line with total, then breakdown per op
                        pct = 100.0 * total_line_cycles / thread_cycles[thread]
                        source_colored = (
                            f"{cb_bg}{source_line}{Colors.RESET}"
                            if cb_bg
                            else source_line
                        )
                        print(
                            f"{color}{file_lineno:<6} {pct:>5.1f}%  "
                            f"{total_line_cycles:<10,} {source_colored}{Colors.RESET if color else ''}"
                        )
                        # Calculate indent to align arrows at end of source line
                        # Format: "%-6s %-7s %-10s %s" = 6 + 1 + 7 + 2 + 10 + 1 = 27 + source
                        indent = 27 + len(source_line)
                        # Sort: explicit ops first (implicit=False), then implicit
                        sorted_ops = sorted(
                            op_groups.items(), key=lambda x: (x[0][1], x[0][0] or "")
                        )
                        op_list = list(sorted_ops)

                        for i, ((op_name, implicit), ops) in enumerate(op_list):
                            op_cycles = sum(r.cycles for r in ops)
                            op_label = op_name or "line"
                            if implicit:
                                op_label = f"{op_label} (implicit)"
                            if len(ops) > 1:
                                op_label = f"{op_label} (x{len(ops)})"
                            is_last = (
                                i == len(op_list) - 1
                                and consumer_dma_info is None
                                and producer_cb_idx is None
                            )
                            arrow = "╰─" if is_last else "├─"
                            print(
                                f"{Colors.DIM}{' ' * indent}"
                                f"{arrow} {op_cycles:,} {op_label}{Colors.RESET}"
                            )

                        # Show DMA attribution for consumer (cb_wait) with CB background color
                        if consumer_dma_info:
                            barrier_kernel, barrier_line, cb_idx, label = (
                                consumer_dma_info
                            )
                            dma_cb_bg = Colors.cb_bg(cb_idx)
                            remark = f"waiting for {label} @ line {barrier_line} ({barrier_kernel})"
                            if dma_cb_bg:
                                remark = f"{dma_cb_bg}{remark}{Colors.RESET}"
                            is_last = producer_cb_idx is None
                            arrow = "╰─" if is_last else "├─"
                            print(
                                f"{Colors.DIM}{' ' * indent}"
                                f"{arrow} {remark}{Colors.RESET}"
                            )

                        # Show DMA barrier remark for producer lines
                        if producer_cb_idx is not None:
                            producer_bg = Colors.cb_bg(producer_cb_idx)
                            remark = f"DMA barrier for CB[{producer_cb_idx}]"
                            if producer_bg:
                                remark = f"{producer_bg}{remark}{Colors.RESET}"
                            print(
                                f"{Colors.DIM}{' ' * indent}"
                                f"╰─ {remark}{Colors.RESET}"
                            )
                    else:
                        cycles_list = [r.cycles for r in line_results]
                        avg_cycles = sum(cycles_list) / len(cycles_list)
                        min_cycles = min(cycles_list)
                        max_cycles = max(cycles_list)
                        sum_cycles = sum(cycles_list)
                        pct = 100.0 * sum_cycles / thread_cycles[thread]
                        source_colored = (
                            f"{cb_bg}{source_line}{Colors.RESET}"
                            if cb_bg
                            else source_line
                        )

                        if min_cycles == max_cycles:
                            stats = f"(x{len(line_results)} = {sum_cycles:,} cycles)"
                            print(
                                f"{color}{file_lineno:<6} {pct:>5.1f}%  "
                                f"{min_cycles:<10,} {source_colored}  "
                                f"{Colors.RESET if color else ''}{Colors.DIM}{stats}{Colors.RESET}"
                            )
                        else:
                            range_str = f"{min_cycles:,}-{max_cycles:,}"
                            stats = f"(x{len(line_results)}, avg={avg_cycles:.1f}, total={sum_cycles:,})"
                            print(
                                f"{color}{file_lineno:<6} {pct:>5.1f}%  "
                                f"{range_str:<10} {source_colored}  "
                                f"{Colors.RESET if color else ''}{Colors.DIM}{stats}{Colors.RESET}"
                            )

                        # Show remarks for consumer/producer lines
                        if consumer_dma_info or producer_cb_idx is not None:
                            indent = 27 + len(source_line)
                            if consumer_dma_info:
                                barrier_kernel, barrier_line, cb_idx, label = (
                                    consumer_dma_info
                                )
                                dma_cb_bg = Colors.cb_bg(cb_idx)
                                remark = f"waiting for {label} @ line {barrier_line} ({barrier_kernel})"
                                if dma_cb_bg:
                                    remark = f"{dma_cb_bg}{remark}{Colors.RESET}"
                                is_last = producer_cb_idx is None
                                arrow = "╰─" if is_last else "├─"
                                print(
                                    f"{Colors.DIM}{' ' * indent}"
                                    f"{arrow} {remark}{Colors.RESET}"
                                )
                            if producer_cb_idx is not None:
                                producer_bg = Colors.cb_bg(producer_cb_idx)
                                remark = f"DMA barrier for CB[{producer_cb_idx}]"
                                if producer_bg:
                                    remark = f"{producer_bg}{remark}{Colors.RESET}"
                                print(
                                    f"{Colors.DIM}{' ' * indent}"
                                    f"╰─ {remark}{Colors.RESET}"
                                )
                else:
                    if source_line.strip():
                        print(f"{file_lineno:<6} {'':7} {'':10} {source_line}")
        else:
            # No source available for this thread
            print(f"       (no source available for kernel '{kernel_name}')")

        print()

    print("=" * 100)
    print("THREAD SUMMARY")
    print("=" * 100)
    for thread in sorted_threads:
        kernel_name = thread_to_kernel.get(thread, "")
        kernel_info = f" [{kernel_name}]" if kernel_name else ""
        print(
            f"  {thread:<12}{kernel_info:>20} {thread_cycles[thread]:>10,} cycles "
            f"({thread_ops[thread]:>3} ops) "
            f"[{100.0 * thread_cycles[thread] / total_cycles:>5.1f}%]"
        )
    print()

    # Roofline: subtract sync waits (cb_wait, cb_reserve) from all threads
    # to isolate actual work cycles.
    _SYNC_OPS = {"cb_wait", "cb_reserve"}
    thread_sync_cycles = defaultdict(int)
    for r in results:
        if r.op_name in _SYNC_OPS:
            thread_sync_cycles[r.thread] += r.cycles

    dm_threads = ["NCRISC", "BRISC"]
    compute_threads = ["TRISC_0", "TRISC_1", "TRISC_2"]
    active_threads = [t for t in sorted_threads if thread_cycles.get(t, 0) > 0]

    if active_threads:
        print("ROOFLINE ANALYSIS")
        print("=" * 100)
        print(f"  {'Thread':<12} {'Total':>10}   - {'Sync Waits':>10}   = {'Work':>10}")
        print(f"  {'-'*12} {'-'*10}   - {'-'*10}   = {'-'*10}")

        thread_work = {}
        for thread in active_threads:
            total = thread_cycles.get(thread, 0)
            sync = thread_sync_cycles.get(thread, 0)
            work = total - sync
            thread_work[thread] = work
            print(f"  {thread:<12} {total:>10,}   - {sync:>10,}   = {work:>10,}")
        print()

        active_dm = [t for t in dm_threads if t in thread_work]
        active_compute = [t for t in compute_threads if t in thread_work]

        memory_best = (
            max(active_dm, key=lambda t: thread_work[t]) if active_dm else None
        )
        compute_best = (
            max(active_compute, key=lambda t: thread_work[t])
            if active_compute
            else None
        )

        memory_cycles = thread_work.get(memory_best, 0) if memory_best else 0
        compute_cycles = thread_work.get(compute_best, 0) if compute_best else 0

        if memory_cycles > 0 or compute_cycles > 0:
            if memory_best:
                print(f"  Memory:  {memory_cycles:>10,} cycles  ({memory_best})")
            if compute_best:
                print(f"  Compute: {compute_cycles:>10,} cycles  ({compute_best})")
            print()

            total_bottleneck = memory_cycles + compute_cycles
            memory_ratio = (
                memory_cycles / total_bottleneck if total_bottleneck > 0 else 0.5
            )

            if memory_cycles > compute_cycles:
                bound_type = "memory"
                bound_pct = 100 * (memory_cycles - compute_cycles) / memory_cycles
            elif compute_cycles > memory_cycles:
                bound_type = "compute"
                bound_pct = 100 * (compute_cycles - memory_cycles) / compute_cycles
            else:
                bound_type = "balanced"
                bound_pct = 0

            roof_width = 40
            marker_pos = int(memory_ratio * (roof_width - 1))
            roof_line = "─" * marker_pos + "●" + "─" * (roof_width - 1 - marker_pos)

            if bound_type == "balanced":
                print(f"  Perfectly balanced!")
            else:
                print(f"  {bound_pct:.0f}% {bound_type} bound")
            print(f"  Compute ├{roof_line}┤ Memory")
            print(
                f"          {compute_cycles:,} cycles"
                f"{' ' * (roof_width - 12)}"
                f"{memory_cycles:,} cycles"
            )
            print()

    print("=" * 100)
    print()


# =============================================================================
# CB Flow Graph Integration
# =============================================================================


def load_cb_flow_graph(csv_path: Path) -> Optional[Dict]:
    """Load CB flow graph JSON from same directory as CSV."""
    json_path = csv_path.parent / "cb_flow_graph.json"
    if not json_path.exists():
        return None

    try:
        with open(json_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_cb_wait_to_dma_map(
    cb_flow: Optional[Dict],
) -> Dict[Tuple[str, int], Tuple[str, int, int, str]]:
    """Build mapping from cb_wait locations to their barrier sources.

    For read-direction CBs (DMA reads into CB, compute consumes):
      compute's cb_wait -> DMA read barrier with label "DMA"
    For write-direction CBs (compute produces, DMA writes from CB):
      DM write's cb_wait -> compute producer with label "compute"

    Returns:
        Dict mapping (kernel, line) -> (source_kernel, source_line, cb_index, label)
    """
    if not cb_flow:
        return {}

    result = {}
    for cb_info in cb_flow.get("circular_buffers", []):
        cb_index = cb_info.get("cb_index", -1)

        # Read direction: DMA reads into CB, compute consumes via cb_wait
        read_barriers = [
            op for op in cb_info.get("wait_ops", []) if op.get("direction") == "read"
        ]
        if read_barriers:
            barrier_op = read_barriers[0]
            barrier_kernel = barrier_op.get("kernel", "")
            barrier_line = barrier_op.get("line", -1)

            for consumer in cb_info.get("consumers", []):
                consumer_kernel = consumer.get("kernel", "")
                consumer_line = consumer.get("line", -1)
                if consumer_line > 0:
                    result[(consumer_kernel, consumer_line)] = (
                        barrier_kernel,
                        barrier_line,
                        cb_index,
                        "DMA",
                    )
            continue

        # Write direction: compute produces, DM writes from CB via cb_wait
        write_dma = [
            op for op in cb_info.get("dma_ops", []) if op.get("direction") == "write"
        ]
        if write_dma:
            producers = cb_info.get("producers", [])
            if producers:
                producer = producers[0]
                producer_kernel = producer.get("kernel", "")
                producer_line = producer.get("line", -1)
                for consumer in cb_info.get("consumers", []):
                    consumer_kernel = consumer.get("kernel", "")
                    consumer_line = consumer.get("line", -1)
                    if consumer_line > 0:
                        result[(consumer_kernel, consumer_line)] = (
                            producer_kernel,
                            producer_line,
                            cb_index,
                            "compute",
                        )

    return result


def build_dma_producer_to_cb_map(
    cb_flow: Optional[Dict],
) -> Dict[Tuple[str, int], int]:
    """Build mapping from DMA barrier locations to CB index.

    Returns:
        Dict mapping (kernel, line) of DMA read barrier -> cb_index
    """
    if not cb_flow:
        return {}

    result = {}
    for cb_info in cb_flow.get("circular_buffers", []):
        cb_index = cb_info.get("cb_index", -1)
        if cb_index < 0 or cb_index >= len(Colors.CB_BACKGROUNDS):
            continue

        for wait_op in cb_info.get("wait_ops", []):
            if wait_op.get("direction") == "read":
                barrier_kernel = wait_op.get("kernel", "")
                barrier_line = wait_op.get("line", -1)
                if barrier_line > 0:
                    result[(barrier_kernel, barrier_line)] = cb_index

    return result


# Global line mapper instance
_global_line_mapper = SourceLineMapper()


def get_line_mapper() -> SourceLineMapper:
    """Get the global line mapper instance."""
    return _global_line_mapper
