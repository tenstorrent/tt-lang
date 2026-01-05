# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic utilities for formatting compiler errors with source context.

This module provides Rust/Swift-style error formatting that displays
source code snippets with ASCII arrows pointing to the error location.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple


def find_variable_assignment(source_lines: List[str], var_name: str, before_line: int) -> Optional[int]:
    """Find the line where a variable was assigned, searching backwards.

    Args:
        source_lines: List of source lines (0-indexed)
        var_name: Variable name to search for
        before_line: Search backwards from this 1-based line number

    Returns:
        1-based line number where assignment was found, or None
    """
    pattern = re.compile(rf'^\s*{re.escape(var_name)}\s*=')

    for i in range(min(before_line - 1, len(source_lines) - 1), -1, -1):
        if pattern.match(source_lines[i]):
            return i + 1
    return None


def _verbose_errors_enabled() -> bool:
    """Check if verbose MLIR error output is enabled."""
    return os.environ.get("TTLANG_VERBOSE_ERRORS", "0") == "1"


class SourceDiagnostic:
    """Format errors with source context and ASCII arrows.

    Produces error messages in the style of modern compilers (Rust, Swift):

        error: type mismatch in add operation
          --> kernel.py:43:16
           |
        43 |         result = l + r
           |                  ^^^ expected bf16, got f32
           |
    """

    def __init__(self, source_lines: List[str], filename: str):
        """Initialize with source code and filename.

        Args:
            source_lines: List of source code lines (0-indexed internally)
            filename: Source file path for display
        """
        self.source_lines = source_lines
        self.filename = filename

    def format_error(
        self,
        line: int,
        col: int,
        message: str,
        label: str = "error",
        span_length: int = 1,
        note: Optional[str] = None,
    ) -> str:
        """Format an error with source context.

        Args:
            line: 1-based line number
            col: 1-based column number
            message: Main error message
            label: Error label (e.g., "error", "warning")
            span_length: Length of the underline (^^^)
            note: Optional additional note

        Returns:
            Formatted error string with source context
        """
        # Build header
        result = [f"{label}: {message}"]
        result.append(f"  --> {self.filename}:{line}:{col}")

        # Get line number width for alignment
        line_num_width = len(str(line))
        gutter = " " * line_num_width

        result.append(f"{gutter} |")

        # Show source line if available
        if 0 < line <= len(self.source_lines):
            source_line = self.source_lines[line - 1].rstrip()
            result.append(f"{line:>{line_num_width}} | {source_line}")

            # Build underline with carets
            underline_padding = " " * (col - 1)
            underline = "^" * max(1, span_length)
            result.append(f"{gutter} | {underline_padding}{underline}")

        result.append(f"{gutter} |")

        if note:
            result.append(f"{gutter} = note: {note}")

        return "\n".join(result)

    def format_error_chain(
        self, errors: List[Tuple[int, int, str, Optional[str]]]
    ) -> str:
        """Format multiple related errors.

        Args:
            errors: List of (line, col, message, note) tuples

        Returns:
            Formatted error chain
        """
        results = []
        for i, (line, col, message, note) in enumerate(errors):
            label = "error" if i == 0 else "note"
            results.append(
                self.format_error(line, col, message, label=label, note=note)
            )
        return "\n\n".join(results)


def parse_mlir_location(loc_str: str) -> Optional[Tuple[str, int, int]]:
    """Parse an MLIR location string to extract file, line, and column.

    MLIR locations can appear in several formats:
    - loc("filename":line:col)
    - loc("filename":line:col to :line:col)
    - loc(#loc1) with #loc1 = loc("filename":line:col)

    Args:
        loc_str: MLIR location string

    Returns:
        Tuple of (filename, line, col) or None if not parseable
    """
    # Match loc("filename":line:col)
    match = re.search(r'loc\("([^"]+)":(\d+):(\d+)', loc_str)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    # Match standalone "filename":line:col pattern
    match = re.search(r'"([^"]+)":(\d+):(\d+)', loc_str)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))

    return None


def extract_location_from_mlir_error(error_msg: str) -> Optional[Tuple[str, int, int]]:
    """Extract source location from an MLIR error message.

    MLIR errors often include location information like:
        error: 'op' op some error message
        note: see current operation: %0 = "op"(...) loc("file.py":42:10)

    Args:
        error_msg: Full MLIR error message

    Returns:
        Tuple of (filename, line, col) or None if no location found
    """
    # Look for location patterns in the error message
    loc_info = parse_mlir_location(error_msg)
    if loc_info:
        return loc_info

    # Try to find location in "note: see current operation" lines
    for line in error_msg.split("\n"):
        if "loc(" in line:
            loc_info = parse_mlir_location(line)
            if loc_info:
                return loc_info

    return None


def _read_file_lines(filepath: str) -> Optional[List[str]]:
    """Read source lines from a file if it exists."""
    try:
        with open(filepath, "r") as f:
            return f.read().splitlines()
    except (IOError, OSError):
        return None


def format_mlir_error(
    error_msg: str,
    source_lines: Optional[List[str]] = None,
    source_file: Optional[str] = None,
) -> str:
    """Format an MLIR error with source context if location is available.

    Args:
        error_msg: The MLIR error message
        source_lines: Original Python source lines (optional, will read from file if needed)
        source_file: Source filename (optional, extracted from error if not provided)

    Returns:
        Formatted error message, with source context if available
    """
    loc_info = extract_location_from_mlir_error(error_msg)

    if loc_info is None:
        return error_msg

    filename, line, col = loc_info
    display_file = source_file if source_file else filename

    # Read source from file if not provided or if line number exceeds provided lines
    if source_lines is None or line > len(source_lines):
        source_lines = _read_file_lines(filename)

    if source_lines is None:
        return error_msg

    diag = SourceDiagnostic(source_lines, display_file)
    core_msg = _extract_core_message(error_msg)

    formatted = diag.format_error(line=line, col=col, message=core_msg)

    if _verbose_errors_enabled():
        formatted += f"\n\nMLIR diagnostic:\n{error_msg}"

    return formatted


def _extract_core_message(error_msg: str) -> str:
    """Extract the core error message from MLIR diagnostic output.

    MLIR errors often look like:
        error: 'ttl.copy' op expects transfer handle to be synchronized with ttl.wait

    This extracts: "expects transfer handle to be synchronized with ttl.wait"
    """
    # Look for the pattern: 'op_name' op <message>
    match = re.search(r"'[^']+' op (.+?)(?:\n|$)", error_msg)
    if match:
        return match.group(1).strip()

    # Look for error: <message> pattern
    match = re.search(r"error: (.+?)(?:\n|$)", error_msg)
    if match:
        return match.group(1).strip()

    # Fall back to first line
    return error_msg.split("\n")[0].strip()


def _extract_note(error_msg: str) -> Optional[str]:
    """Extract any note from the MLIR error message."""
    match = re.search(r"note: (.+?)(?:\n|$)", error_msg)
    if match:
        return match.group(1).strip()
    return None


def format_python_error(
    error: Exception,
    source_file: str,
    line: int,
    source_lines: Optional[List[str]] = None,
) -> str:
    """Format a Python error with source context.

    Args:
        error: The Python exception
        source_file: Source file path
        line: Line number in source file
        source_lines: Source lines (will read from file if not provided)

    Returns:
        Formatted error message with source context
    """
    if source_lines is None:
        source_lines = _read_file_lines(source_file)

    if source_lines is None:
        return f"{type(error).__name__}: {error}"

    diag = SourceDiagnostic(source_lines, source_file)
    return diag.format_error(
        line=line,
        col=1,
        message=str(error),
        label=type(error).__name__,
    )


class TTLangCompileError(Exception):
    """Exception for tt-lang compilation errors with source context.

    This exception carries enough information to produce pretty error messages
    pointing to the exact source location where the error occurred.
    """

    def __init__(
        self,
        message: str,
        source_file: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        source_lines: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.source_file = source_file
        self.line = line
        self.col = col
        self.source_lines = source_lines

    def format(self) -> str:
        """Format error with source context if available."""
        if self.source_file is None or self.line is None:
            return str(self)

        # Read source lines if not provided
        lines = self.source_lines
        if lines is None:
            lines = _read_file_lines(self.source_file)

        if lines is None:
            return (
                f"error: {self}\n  --> {self.source_file}:{self.line}:{self.col or 1}"
            )

        diag = SourceDiagnostic(lines, self.source_file)
        return diag.format_error(
            line=self.line,
            col=self.col or 1,
            message=str(self),
        )
