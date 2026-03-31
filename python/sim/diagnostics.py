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


def is_simulator_frame(filename: str) -> bool:
    """Check if a filename is from simulator internal code.

    Args:
        filename: Path to source file

    Returns:
        True if this is a simulator internal frame that should be skipped
    """
    return "/python/sim/" in filename or "/greenlet/" in filename


def find_user_code_location() -> Tuple[str, int]:
    """Walk up the call stack to find user code location.

    Skips simulator internal frames (anything in /python/sim/ or /greenlet/)
    and returns the first user code location found.

    Returns:
        Tuple of (filename, line_number)

    Raises:
        RuntimeError: If no user code found in stack (should never happen)
    """
    frame = inspect.currentframe()
    if not frame:
        raise RuntimeError(
            "inspect.currentframe() returned None - introspection not supported"
        )

    # Start from caller and walk up to find user code
    caller_frame = frame.f_back
    while caller_frame:
        filename = caller_frame.f_code.co_filename
        # Skip simulator internals - return first non-sim frame
        if not is_simulator_frame(filename):
            return filename, caller_frame.f_lineno
        caller_frame = caller_frame.f_back

    raise RuntimeError(
        "No user code found in call stack - all frames are simulator code"
    )


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


def print_diagnostic_warning(
    message: str,
    source_file: str,
    source_line: int,
    cores_label: str,
    flush: bool = True,
) -> None:
    """Print a warning with diagnostic formatting.

    Args:
        message: Warning message to display
        source_file: Path to source file where warning occurred
        source_line: Line number in source file
        cores_label: Label identifying affected cores (e.g., "core0" or "cores: 0-3")
        flush: Whether to flush output immediately (default: True)
    """
    import builtins

    diagnostics = lazy_import_diagnostics()
    SourceDiagnostic = diagnostics.SourceDiagnostic

    # Read source lines
    with open(source_file, "r") as f:
        source_lines = f.read().splitlines()

    # Format warning using diagnostics module
    diag = SourceDiagnostic(source_lines, source_file)
    warning_msg = diag.format_error(
        line=source_line,
        col=1,
        message=f"{message} ({cores_label})",
        label="warning",
    )
    builtins.print(warning_msg, flush=flush)


def print_diagnostic_error(
    name: str,
    message: str,
    source_file: str,
    source_line: int,
    source_col: int = 1,
) -> None:
    """Print an error with diagnostic formatting.

    Args:
        name: Name/label for the error context (e.g., thread name)
        message: Error message to display
        source_file: Path to source file where error occurred
        source_line: Line number in source file
        source_col: Column number in source file (default: 1)
    """
    diagnostics = lazy_import_diagnostics()
    TTLangCompileError = diagnostics.TTLangCompileError
    compile_error = TTLangCompileError(
        message,
        source_file=source_file,
        line=source_line,
        col=source_col,
    )
    print(f"\n❌ Error in {name}:")
    print(compile_error.format())
    print("-" * 50)


def warn_once_per_location(
    warnings_dict: dict[tuple[str, int], set[str]],
    message: str,
    core_id: str,
) -> None:
    """Issue a warning once per source location, tracking which cores hit it.

    This is a common pattern for simulator warnings: we want to warn about an issue
    once per source location, but show which cores encountered it.

    Args:
        warnings_dict: Dictionary tracking {(filename, line): set(core_ids)}
        message: Warning message to display
        core_id: ID of the current core (from get_current_core_id())
    """
    # Find user code location
    source_file, source_line = find_user_code_location()

    # Track this core hitting this location
    location_key = (source_file, source_line)
    first_occurrence = location_key not in warnings_dict
    if first_occurrence:
        warnings_dict[location_key] = set()

    warnings_dict[location_key].add(core_id)

    # Only print on first occurrence for this location
    if first_occurrence:
        cores = warnings_dict[location_key]

        # Format the core label
        if len(cores) == 1 and core_id != "unknown":
            cores_label = core_id
        else:
            # Extract numeric core IDs and format as ranges
            unique_cores = sorted(cores, key=lambda x: (len(x), x))
            try:
                core_numbers = [
                    int(core[4:])
                    for core in unique_cores
                    if core.startswith("core") and core[4:].isdigit()
                ]
                if core_numbers:
                    cores_label = f"cores: {format_core_ranges(core_numbers)}"
                else:
                    cores_label = f"cores: {', '.join(unique_cores)}"
            except (ValueError, IndexError):
                cores_label = f"cores: {', '.join(unique_cores)}"

        # Print warning with diagnostic formatting
        print_diagnostic_warning(message, source_file, source_line, cores_label)
