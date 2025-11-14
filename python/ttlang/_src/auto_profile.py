# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-profiling infrastructure for tt-lang kernels.

Enabled via TTLANG_AUTO_PROFILE=1 environment variable.
Automatically instruments every operation with signposts and generates
a visual profile report showing cycle counts per source line.
"""

import os
import ast
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def is_auto_profile_enabled() -> bool:
    """Check if auto-profiling is enabled via environment variable."""
    return os.environ.get("TTLANG_AUTO_PROFILE", "0") == "1"


class SourceLineMapper:
    """Maps signpost markers back to source code lines."""

    def __init__(self):
        self.signpost_to_line: Dict[str, Tuple[int, str]] = {}  # signpost -> (lineno, source)
        self.source_lines: List[str] = []

    def register_signpost(self, signpost_name: str, lineno: int, source: str):
        """Register a signpost with its source line information."""
        self.signpost_to_line[signpost_name] = (lineno, source)

    def set_source(self, source_lines: List[str]):
        """Set the source code lines for display."""
        self.source_lines = source_lines

    def get_line_info(self, signpost_name: str) -> Optional[Tuple[int, str]]:
        """Get line number and source for a signpost."""
        return self.signpost_to_line.get(signpost_name)


class ProfileResult:
    """Represents profiling results for a single signpost."""

    def __init__(self, signpost: str, thread: str, cycles: int, lineno: int, source: str):
        self.signpost = signpost
        self.thread = thread
        self.cycles = cycles
        self.lineno = lineno
        self.source = source.strip()


def generate_signpost_name(operation: str, lineno: int, col: int) -> Tuple[str, str]:
    """
    Generate before/after signpost names for an operation.

    Returns:
        Tuple of (before_name, after_name)
    """
    # Use format: op_L<line>_C<col>_before/after
    base = f"{operation}_L{lineno}_C{col}"
    return (f"{base}_before", f"{base}_after")


def parse_device_profile_csv(csv_path: Path, line_mapper: SourceLineMapper) -> List[ProfileResult]:
    """
    Parse the device profile CSV and extract signpost timing data.

    Args:
        csv_path: Path to profile_log_device.csv
        line_mapper: Mapper to correlate signposts to source lines

    Returns:
        List of ProfileResult objects sorted by line number
    """
    results = []
    signpost_starts = {}  # Track ZONE_START timestamps

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) < 13:
                continue

            # Try to parse timestamp - skip if it's not a valid integer
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

                # Get source line info
                line_info = line_mapper.get_line_info(signpost)
                if line_info:
                    lineno, source = line_info
                    results.append(ProfileResult(signpost, thread, duration, lineno, source))

                del signpost_starts[key]

    # Sort by line number
    results.sort(key=lambda r: r.lineno)
    return results


def print_profile_report(results: List[ProfileResult], source_lines: List[str]):
    """
    Print a beautiful side-by-side profile report organized by thread.

    Shows full source context with function decorators and organized by thread/core.
    """
    print()
    print("=" * 100)
    print("TTLANG AUTO-PROFILE REPORT")
    print("=" * 100)
    print()

    # Calculate total cycles per thread (not sum of all results since there are duplicates)
    thread_cycles = defaultdict(int)
    thread_ops = defaultdict(int)
    for r in results:
        thread_cycles[r.thread] += r.cycles
        thread_ops[r.thread] += 1

    # Total is the max of any single thread (they run in parallel)
    total_cycles = max(thread_cycles.values()) if thread_cycles else 1

    print(f"Total operations: {len(results)}")
    print(f"Longest thread: {total_cycles:,} cycles")
    print()

    # Group results by thread
    thread_to_results = defaultdict(list)
    for result in results:
        thread_to_results[result.thread].append(result)

    # Sort threads (NCRISC, BRISC, TRISC_0, TRISC_1, TRISC_2)
    thread_order = ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
    sorted_threads = sorted(thread_to_results.keys(),
                           key=lambda t: thread_order.index(t) if t in thread_order else 999)

    # Print each thread's execution
    for thread in sorted_threads:
        thread_results = sorted(thread_to_results[thread], key=lambda r: r.lineno)

        print("=" * 100)
        print(f"THREAD: {thread:<10} ({thread_ops[thread]} ops, {thread_cycles[thread]:,} cycles, "
              f"{100.0 * thread_cycles[thread] / total_cycles:.1f}% of total)")
        print("=" * 100)
        print()
        print(f"{'LINE':<6} {'CYCLES':<10} SOURCE")
        print(f"{'-'*6} {'-'*10} {'-'*70}")

        # Group results by source text (since line numbers are relative to inner functions)
        source_groups = defaultdict(list)
        for result in thread_results:
            # Use stripped source as key to match against full file
            source_groups[result.source.strip()].append(result)

        # Print the entire source with cycle annotations where available
        if source_lines:
            for lineno in range(1, len(source_lines) + 1):
                source_line = source_lines[lineno - 1].rstrip()
                source_stripped = source_line.strip()

                if source_stripped in source_groups:
                    # This line has profiling data
                    line_results = source_groups[source_stripped]

                    if len(line_results) == 1:
                        # Single execution
                        r = line_results[0]
                        print(f"{lineno:<6} {r.cycles:<10,} {source_line}")
                    else:
                        # Multiple executions - show summary
                        cycles_list = [r.cycles for r in line_results]
                        avg_cycles = sum(cycles_list) / len(cycles_list)
                        min_cycles = min(cycles_list)
                        max_cycles = max(cycles_list)
                        total_cycles = sum(cycles_list)

                        if min_cycles == max_cycles:
                            # All executions took same time
                            print(f"{lineno:<6} {min_cycles:<10,} {source_line}  (×{len(line_results)} = {total_cycles:,} cycles)")
                        else:
                            # Variable execution times - format range to fit in same column width
                            range_str = f"{min_cycles:,}-{max_cycles:,}"
                            print(f"{lineno:<6} {range_str:<10} {source_line}  (×{len(line_results)}, avg={avg_cycles:.1f}, total={total_cycles:,})")
                else:
                    # Context line - no profiling data
                    if source_line.strip():  # Only show non-empty lines
                        print(f"{'':6} {'':10} {source_line}")

        print()

    # Thread summary
    print("=" * 100)
    print("THREAD SUMMARY")
    print("=" * 100)
    for thread in sorted_threads:
        print(f"  {thread:<12} {thread_cycles[thread]:>10,} cycles ({thread_ops[thread]:>3} ops) "
              f"[{100.0 * thread_cycles[thread] / total_cycles:>5.1f}%]")
    print()
    print("=" * 100)
    print()


def run_profiling_pipeline(flatbuffer_path: Path, source_lines: List[str],
                          line_mapper: SourceLineMapper) -> Optional[List[ProfileResult]]:
    """
    Run the full profiling pipeline: ttrt perf → parse CSV → return results.

    Args:
        flatbuffer_path: Path to the compiled flatbuffer
        source_lines: Source code lines for display
        line_mapper: Mapper to correlate signposts to source

    Returns:
        List of ProfileResult objects, or None on error
    """
    print(f"\n🔍 Running profiler on {flatbuffer_path.name}...")

    # Run ttrt perf
    try:
        result = subprocess.run(
            ["ttrt", "perf", str(flatbuffer_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"❌ ttrt perf failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("❌ ttrt perf timed out")
        return None
    except FileNotFoundError:
        print("❌ ttrt command not found - is it in your PATH?")
        return None

    # Find the device profile CSV
    # ttrt perf creates artifacts in current working directory
    artifact_dir = Path.cwd() / "ttrt-artifacts" / flatbuffer_path.name / "perf"
    device_csv = artifact_dir / "profile_log_device.csv"

    if not device_csv.exists():
        print(f"❌ Device profile CSV not found at {device_csv}")
        print(f"   (Expected at: {artifact_dir})")
        return None

    # Parse the CSV
    print(f"📊 Parsing profile data from {device_csv}...")
    results = parse_device_profile_csv(device_csv, line_mapper)

    if not results:
        print("⚠️  No signpost data found in profile")
        print(f"   Line mapper has {len(line_mapper.signpost_to_line)} registered signposts")
        print(f"   Check if signposts are in the CSV with: grep 'ZONE' {device_csv}")
        return None

    print(f"✅ Found {len(results)} profiled operations")
    print(f"   Registered signposts: {len(line_mapper.signpost_to_line)}")

    # Debug: show which operations were found
    unique_sources = set(r.source for r in results)
    print(f"   Unique operations: {len(unique_sources)}")

    return results


# Global line mapper instance
_global_line_mapper = SourceLineMapper()


def get_line_mapper() -> SourceLineMapper:
    """Get the global line mapper instance."""
    return _global_line_mapper
