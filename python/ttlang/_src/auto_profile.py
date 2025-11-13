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
        for row in reader:
            if len(row) < 13:
                continue

            thread = row[3]
            timestamp = int(row[5])
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
    Print a beautiful side-by-side profile report.

    Format:
        LINE THREAD    CYCLES  SOURCE CODE
        ---- ------    ------  -----------
          42 BRISC        23   lhs = lhs_cb.pop()
          43 NCRISC       18   result = lhs + rhs
    """
    print()
    print("=" * 100)
    print("TTLANG AUTO-PROFILE REPORT")
    print("=" * 100)
    print()
    print(f"{'LINE':<6} {'THREAD':<12} {'CYCLES':<8}  SOURCE")
    print(f"{'-'*6} {'-'*12} {'-'*8}  {'-'*60}")

    # Group by line number
    line_to_results = defaultdict(list)
    for result in results:
        line_to_results[result.lineno].append(result)

    # Print results line by line
    prev_lineno = 0
    for lineno in sorted(line_to_results.keys()):
        # Add spacing for line gaps
        if lineno > prev_lineno + 1:
            print()

        line_results = line_to_results[lineno]

        # Print first result with source code
        first = line_results[0]
        print(f"{first.lineno:<6} {first.thread:<12} {first.cycles:<8}  {first.source}")

        # Print additional results for same line (different threads)
        for result in line_results[1:]:
            print(f"{'':6} {result.thread:<12} {result.cycles:<8}")

        prev_lineno = lineno

    print()
    print("=" * 100)
    print(f"Total operations profiled: {len(results)}")
    print(f"Total cycles: {sum(r.cycles for r in results)}")
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
    artifact_dir = flatbuffer_path.parent / "ttrt-artifacts" / flatbuffer_path.name / "perf"
    device_csv = artifact_dir / "profile_log_device.csv"

    if not device_csv.exists():
        print(f"❌ Device profile CSV not found at {device_csv}")
        return None

    # Parse the CSV
    print(f"📊 Parsing profile data from {device_csv}...")
    results = parse_device_profile_csv(device_csv, line_mapper)

    if not results:
        print("⚠️  No signpost data found in profile")
        return None

    return results


# Global line mapper instance
_global_line_mapper = SourceLineMapper()


def get_line_mapper() -> SourceLineMapper:
    """Get the global line mapper instance."""
    return _global_line_mapper
