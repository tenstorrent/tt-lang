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
    Examples:
      "line_52_before" -> (None, False)
      "line_52_cb_wait_before" -> ("cb_wait", False)
      "line_52_implicit_cb_pop_before" -> ("cb_pop", True)
    """
    parts = signpost.rsplit("_", 1)  # Split off before/after
    if len(parts) != 2 or parts[1] not in ("before", "after"):
        return None, False

    middle = parts[
        0
    ]  # e.g., "line_52" or "line_52_cb_wait" or "line_52_implicit_cb_pop"
    line_parts = middle.split("_", 2)  # Split "line", "52", rest
    if len(line_parts) <= 2:
        return None, False

    rest = line_parts[2]  # e.g., "cb_wait" or "implicit_cb_pop"
    if rest.startswith("implicit_"):
        return rest[9:], True
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
    cb_wait_to_dma: Optional[Dict[Tuple[str, int], Tuple[str, int, int]]] = None,
    dma_producer_to_cb: Optional[Dict[Tuple[str, int], int]] = None,
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
    """
    if cb_wait_to_dma is None:
        cb_wait_to_dma = {}
    if dma_producer_to_cb is None:
        dma_producer_to_cb = {}

    print()
    print("=" * 100)
    print("TTLANG AUTO-PROFILE REPORT")
    print("=" * 100)
    print()

    # DEBUG: Test CB background colors with different text colors - REMOVE LATER
    print("CB Color Test (black/yellow/red text on each background):")
    black_text = "\033[30m"
    for i, bg in enumerate(Colors.CB_BACKGROUNDS):
        print(f"  {bg}{black_text}CB[{i}] black{Colors.RESET}  "
              f"{bg}{Colors.YELLOW}CB[{i}] yellow{Colors.RESET}  "
              f"{bg}{Colors.RED}CB[{i}] red{Colors.RESET}")
    print()

    # Print CB color key
    active_cbs = set(dma_producer_to_cb.values()) | {info[2] for info in cb_wait_to_dma.values()}
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
            line_offset = getattr(line_mapper, "line_offset", 0)

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

                    # Check if this line is a DMA producer and get CB background
                    original_lineno = line_results[0].lineno if line_results else -1
                    producer_cb_idx = dma_producer_to_cb.get((kernel_name, original_lineno))
                    cb_bg = Colors.cb_bg(producer_cb_idx) if producer_cb_idx is not None else ""

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
                        source_colored = f"{cb_bg}{source_line}{Colors.RESET}" if cb_bg else source_line
                        print(
                            f"{color}{file_lineno:<6} {pct:>5.1f}%  "
                            f"{r.cycles:<10,} {source_colored}{Colors.RESET if color else ''}"
                        )
                    elif has_named_ops:
                        # Show line with total, then breakdown per op
                        pct = 100.0 * total_line_cycles / thread_cycles[thread]
                        source_colored = f"{cb_bg}{source_line}{Colors.RESET}" if cb_bg else source_line
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

                        # Check if any op is cb_wait and get DMA attribution
                        has_cb_wait = any(op_name == "cb_wait" for (op_name, _), _ in op_list)
                        dma_info = None
                        if has_cb_wait and line_results:
                            dma_info = cb_wait_to_dma.get((kernel_name, original_lineno))

                        for i, ((op_name, implicit), ops) in enumerate(op_list):
                            op_cycles = sum(r.cycles for r in ops)
                            op_label = op_name or "line"
                            if implicit:
                                op_label = f"{op_label} (implicit)"
                            if len(ops) > 1:
                                op_label = f"{op_label} (x{len(ops)})"
                            is_last = i == len(op_list) - 1 and dma_info is None
                            arrow = "╰─" if is_last else "├─"
                            print(
                                f"{Colors.DIM}{' ' * indent}"
                                f"{arrow} {op_cycles:,} {op_label}{Colors.RESET}"
                            )

                        # Show DMA attribution for cb_wait with CB background color
                        if dma_info:
                            dma_kernel, dma_line, cb_idx = dma_info
                            dma_cb_bg = Colors.cb_bg(cb_idx)
                            # Get cb_wait cycles for the remark
                            cb_wait_cycles = sum(
                                r.cycles for r in line_results if r.op_name == "cb_wait"
                            )
                            remark = f"{cb_wait_cycles:,} cycles waiting for DMA @ line {dma_line} ({dma_kernel})"
                            if dma_cb_bg:
                                remark = f"{dma_cb_bg}{remark}{Colors.RESET}"
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
                        source_colored = f"{cb_bg}{source_line}{Colors.RESET}" if cb_bg else source_line

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
) -> Dict[Tuple[str, int], Tuple[str, int, int]]:
    """Build mapping from cb_wait locations to DMA producer locations.

    Only maps consumers waiting for DMA reads (data flowing into CB).
    cb_wait ops waiting for compute output (where DMA is a write) are not mapped.

    Returns:
        Dict mapping (kernel, line) of cb_wait -> (dma_kernel, dma_line, cb_index)
    """
    if not cb_flow:
        return {}

    result = {}
    for cb_info in cb_flow.get("circular_buffers", []):
        cb_index = cb_info.get("cb_index", -1)

        # Only consider DMA reads (data flowing INTO CB)
        dma_reads = [op for op in cb_info.get("dma_ops", []) if op.get("direction") == "read"]
        if not dma_reads:
            continue

        dma_op = dma_reads[0]
        dma_kernel = dma_op.get("kernel", "")
        dma_line = dma_op.get("line", -1)

        for consumer in cb_info.get("consumers", []):
            consumer_kernel = consumer.get("kernel", "")
            consumer_line = consumer.get("line", -1)
            if consumer_line > 0:
                result[(consumer_kernel, consumer_line)] = (
                    dma_kernel,
                    dma_line,
                    cb_index,
                )

    return result


def build_dma_producer_to_cb_map(
    cb_flow: Optional[Dict],
) -> Dict[Tuple[str, int], int]:
    """Build mapping from DMA producer locations to CB index.

    Returns:
        Dict mapping (kernel, line) of DMA read -> cb_index
    """
    if not cb_flow:
        return {}

    result = {}
    for cb_info in cb_flow.get("circular_buffers", []):
        cb_index = cb_info.get("cb_index", -1)
        if cb_index < 0 or cb_index >= len(Colors.CB_BACKGROUNDS):
            continue

        for dma_op in cb_info.get("dma_ops", []):
            if dma_op.get("direction") == "read":
                dma_kernel = dma_op.get("kernel", "")
                dma_line = dma_op.get("line", -1)
                if dma_line > 0:
                    result[(dma_kernel, dma_line)] = cb_index

    return result


# Global line mapper instance
_global_line_mapper = SourceLineMapper()


def get_line_mapper() -> SourceLineMapper:
    """Get the global line mapper instance."""
    return _global_line_mapper
