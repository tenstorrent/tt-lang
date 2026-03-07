# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic utilities for the simulator.

Provides shared utilities for error reporting and source location tracking,
including lazy import of the compiler's diagnostic module.
"""

import inspect
from typing import Any, Optional, Tuple


def lazy_import_diagnostics() -> Any:
    """Lazy import of ttl.diagnostics module to avoid circular dependency.

    Returns:
        The ttl.diagnostics module

    Raises:
        ImportError: If the diagnostics module cannot be loaded
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Direct import of diagnostics module without going through ttl package
    # This avoids importing the full compiler infrastructure
    diagnostics_path = Path(__file__).parent.parent / "ttl" / "diagnostics.py"
    spec = importlib.util.spec_from_file_location("ttl.diagnostics", diagnostics_path)
    if spec and spec.loader:
        diagnostics = importlib.util.module_from_spec(spec)
        sys.modules["ttl.diagnostics"] = diagnostics
        spec.loader.exec_module(diagnostics)
        return diagnostics
    raise ImportError("Could not load ttl.diagnostics")


def find_user_code_location(
    skip_frames: int = 1,
) -> Tuple[Optional[str], Optional[int]]:
    """Walk up the call stack to find user code location.

    Skips simulator internal frames (anything in /python/sim/ or greenlet).

    Args:
        skip_frames: Number of frames to skip before starting the search.
                    Use 1 to skip the immediate caller, 2 to skip caller + one more, etc.

    Returns:
        Tuple of (filename, line_number) or (None, None) if no user code found
    """
    frame = inspect.currentframe()
    if not frame:
        return None, None

    # Skip the requested number of frames
    caller_frame = frame.f_back
    for _ in range(skip_frames):
        if caller_frame and caller_frame.f_back:
            caller_frame = caller_frame.f_back
        else:
            return None, None

    # Walk up the stack to find user code
    while caller_frame:
        filename = caller_frame.f_code.co_filename
        # Skip simulator internals
        if "/python/sim/" not in filename and "greenlet" not in filename:
            return filename, caller_frame.f_lineno
        caller_frame = caller_frame.f_back

    return None, None


def format_core_ranges(core_numbers: list[int]) -> str:
    """Format a list of core numbers as ranges.

    Args:
        core_numbers: Sorted list of core numbers (e.g., [0, 1, 2, 3, 8, 9, 10, 11])

    Returns:
        Formatted string with ranges (e.g., "0-3, 8-11")
    """
    if not core_numbers:
        return ""

    # Sort to ensure consecutive numbers are adjacent
    sorted_cores = sorted(core_numbers)
    ranges: list[str] = []
    start = sorted_cores[0]
    end = sorted_cores[0]

    for i in range(1, len(sorted_cores)):
        if sorted_cores[i] == end + 1:
            # Consecutive, extend the range
            end = sorted_cores[i]
        else:
            # Gap found, save the current range and start a new one
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = sorted_cores[i]
            end = sorted_cores[i]

    # Add the final range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def extract_core_id_from_thread_name(thread_name: Optional[str]) -> str:
    """Extract core ID from a thread name.

    Thread names follow the pattern "coreN-type" where N is the core number
    and type is the thread type (e.g., "dm", "compute").

    Args:
        thread_name: Thread name like "core0-dm" or "core0-compute"

    Returns:
        Core ID like "core0", or "unknown" if extraction fails

    Examples:
        >>> extract_core_id_from_thread_name("core0-dm")
        'core0'
        >>> extract_core_id_from_thread_name("core15-compute")
        'core15'
        >>> extract_core_id_from_thread_name(None)
        'unknown'
    """
    if not thread_name:
        return "unknown"

    # Extract core ID from thread name (e.g., "core0-dm" -> "core0")
    if "-" in thread_name:
        return thread_name.split("-")[0]  # Take the part before first dash

    return thread_name
