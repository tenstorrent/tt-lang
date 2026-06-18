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

    Uses file-based import rather than ``from ttl import diagnostics`` because
    tt-lang-sim shadows the ``ttl`` module with a simulator shim.

    Returns:
        The ttl.diagnostics module

    Raises:
        ImportError: If the diagnostics module cannot be loaded
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Resolve the real path of this file (follows symlinks) to find the
    # source tree, then locate diagnostics.py relative to it.
    sim_dir = Path(__file__).resolve().parent  # python/sim/
    diagnostics_path = sim_dir.parent / "ttl" / "diagnostics.py"
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
    return (
        "/python/sim/" in filename
        or "/ttl/sim/" in filename
        or "/greenlet/" in filename
    )


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


def format_node_ranges(node_numbers: list[int]) -> str:
    """Format a list of node numbers as ranges.

    Args:
        node_numbers: Sorted list of node numbers (e.g., [0, 1, 2, 3, 8, 9, 10, 11])

    Returns:
        Formatted string with ranges (e.g., "0-3, 8-11")
    """
    if not node_numbers:
        return ""

    # Sort to ensure consecutive numbers are adjacent
    sorted_nodes = sorted(node_numbers)
    ranges: list[str] = []
    start = sorted_nodes[0]
    end = sorted_nodes[0]

    for i in range(1, len(sorted_nodes)):
        if sorted_nodes[i] == end + 1:
            # Consecutive, extend the range
            end = sorted_nodes[i]
        else:
            # Gap found, save the current range and start a new one
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = sorted_nodes[i]
            end = sorted_nodes[i]

    # Add the final range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def extract_node_id_from_kernel_name(kernel_name: Optional[str]) -> str:
    """Extract node ID from a scheduled kernel name.

    Names follow the pattern ``nodeN-<func_name>`` where N is the node number
    and ``<func_name>`` is the kernel function's ``__name__``
    (see :func:`sim.greenlet_scheduler.kernel_display_name`).

    Args:
        kernel_name: Scheduled kernel display name like ``node0-mm_reader`` or
            ``node15-mm_compute``.

    Returns:
        Node ID like "node0", or "unknown" if extraction fails

    Examples:
        >>> extract_node_id_from_kernel_name("node0-mm_reader")
        'node0'
        >>> extract_node_id_from_kernel_name("node15-mm_compute")
        'node15'
        >>> extract_node_id_from_kernel_name(None)
        'unknown'
    """
    if not kernel_name:
        return "unknown"

    # Extract node ID from scheduled kernel name (e.g., "node0-dm" -> "node0")
    if "-" in kernel_name:
        return kernel_name.split("-")[0]  # Take the part before first dash

    return kernel_name


def print_diagnostic_warning(
    message: str,
    source_file: str,
    source_line: int,
    nodes_label: str,
    flush: bool = True,
) -> None:
    """Print a warning with diagnostic formatting.

    Args:
        message: Warning message to display
        source_file: Path to source file where warning occurred
        source_line: Line number in source file
        nodes_label: Label identifying affected nodes (e.g., "node0" or "nodes: 0-3")
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
        message=f"{message} ({nodes_label})",
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
        name: Name/label for the error context (e.g., scheduled kernel name or "deadlock")
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
    if name == "deadlock":
        print("\n❌ Error during deadlock detection:")
    else:
        print(f"\n❌ Error in kernel {name}:")
    print(compile_error.format())
    print("-" * 50)


def warn_once_per_location(
    warnings_dict: dict[tuple[str, int], set[str]],
    message: str,
    node_id: str,
) -> None:
    """Issue a warning once per source location, tracking which nodes hit it.

    This is a common pattern for simulator warnings: we want to warn about an issue
    once per source location, but show which nodes encountered it.

    Args:
        warnings_dict: Dictionary tracking {(filename, line): set(node_ids)}
        message: Warning message to display
        node_id: ID of the current node (from get_current_node_id())
    """
    # Find user code location
    source_file, source_line = find_user_code_location()

    # Track this node hitting this location
    location_key = (source_file, source_line)
    first_occurrence = location_key not in warnings_dict
    if first_occurrence:
        warnings_dict[location_key] = set()

    warnings_dict[location_key].add(node_id)

    # Only print on first occurrence for this location
    if first_occurrence:
        nodes = warnings_dict[location_key]

        # Format the node label
        if len(nodes) == 1 and node_id != "unknown":
            nodes_label = node_id
        else:
            # Extract numeric node IDs and format as ranges
            unique_nodes = sorted(nodes, key=lambda x: (len(x), x))
            try:
                node_numbers = [
                    int(n[4:])
                    for n in unique_nodes
                    if n.startswith("node") and n[4:].isdigit()
                ]
                if node_numbers:
                    nodes_label = f"nodes: {format_node_ranges(node_numbers)}"
                else:
                    nodes_label = f"nodes: {', '.join(unique_nodes)}"
            except (ValueError, IndexError):
                nodes_label = f"nodes: {', '.join(unique_nodes)}"

        # Print warning with diagnostic formatting
        print_diagnostic_warning(message, source_file, source_line, nodes_label)
