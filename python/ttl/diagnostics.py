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


def find_variable_assignment(
    source_lines: List[str], var_name: str, before_line: int
) -> Optional[int]:
    """Find the line where a variable was assigned, searching backwards.

    Args:
        source_lines: List of source lines (0-indexed)
        var_name: Variable name to search for
        before_line: Search backwards from this 1-based line number

    Returns:
        1-based line number where assignment was found, or None
    """
    pattern = re.compile(rf"^\s*{re.escape(var_name)}\s*=")

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
        indent: int = 0,
    ) -> str:
        """Format an error with source context.

        Args:
            line: 1-based line number
            col: 1-based column number
            message: Main error message
            label: Error label (e.g., "error", "warning")
            span_length: Length of the underline (^^^)
            note: Optional additional note
            indent: Number of leading spaces to prepend to every line
                of the rendered block, used to visually subordinate
                child notes under their parent error.

        Returns:
            Formatted error string with source context
        """
        prefix = " " * indent
        # Build header
        result = [f"{prefix}{label}: {message}"]
        result.append(f"{prefix}  --> {self.filename}:{line}:{col}")

        # Get line number width for alignment
        line_num_width = len(str(line))
        gutter = " " * line_num_width

        result.append(f"{prefix}{gutter} |")

        # Show source line if available
        if 0 < line <= len(self.source_lines):
            source_line = self.source_lines[line - 1].rstrip()
            result.append(f"{prefix}{line:>{line_num_width}} | {source_line}")

            # Build underline with carets
            underline_padding = " " * (col - 1)
            underline = "^" * max(1, span_length)
            result.append(f"{prefix}{gutter} | {underline_padding}{underline}")

        result.append(f"{prefix}{gutter} |")

        if note:
            result.append(f"{prefix}{gutter} = note: {note}")

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

    Splits the diagnostic stream into one group per error (each error
    plus the notes attached to it) and renders each group as its own
    `error:` + `note:` blocks. Multiple unrelated violations are
    rendered as multiple errors, not a single primary with the rest
    folded into notes.

    Args:
        error_msg: The MLIR error message
        source_lines: Original Python source lines (optional, will read from file if needed)
        source_file: Source filename (optional, extracted from error if not provided)

    Returns:
        Formatted error message, with source context if available
    """
    groups = _extract_diagnostic_groups(error_msg)

    if not groups:
        return error_msg

    blocks: List[str] = []
    for primary, notes in groups:
        primary_loc, primary_msg = primary
        block = _render_diagnostic_block(
            primary_loc, primary_msg, "error", source_lines, source_file
        )
        if block is not None:
            blocks.append(block)
        seen: List[Tuple[Optional[Tuple[str, int, int]], str]] = []
        for note_loc, note_msg in notes:
            # Drop the MLIR-internal `see current operation` note,
            # which echoes the failing op's IR rather than adding user
            # context.
            if note_msg.startswith("see current operation"):
                continue
            # Drop duplicate notes (some passes emit the same note
            # at the same location more than once).
            key = (note_loc, note_msg)
            if key in seen:
                continue
            seen.append(key)
            # Indent notes 2 spaces so they read as visually subordinate
            # to their parent error.
            note_block = _render_diagnostic_block(
                note_loc, note_msg, "note", source_lines, source_file, indent=2
            )
            if note_block is not None:
                blocks.append(note_block)

    formatted = "\n\n".join(blocks)

    if _verbose_errors_enabled():
        formatted += f"\n\nMLIR diagnostic:\n{error_msg}"

    return formatted


def _render_diagnostic_block(
    loc_info: Optional[Tuple[str, int, int]],
    message: str,
    label: str,
    source_lines: Optional[List[str]],
    source_file: Optional[str],
    indent: int = 0,
) -> Optional[str]:
    """Render one diagnostic (primary or note) with a source block.

    Falls back to a `<label>: <message>` line if location resolution
    fails — better than dropping the note entirely.
    """
    prefix = " " * indent
    if loc_info is None:
        return f"{prefix}{label}: {message}"

    filename, line, col = loc_info
    display_file = source_file if source_file else filename

    block_lines = source_lines
    if block_lines is None or line > len(block_lines):
        block_lines = _read_file_lines(filename)

    if block_lines is None:
        return f"{prefix}{label}: {message}\n{prefix}  --> {filename}:{line}:{col}"

    diag = SourceDiagnostic(block_lines, display_file)
    return diag.format_error(
        line=line, col=col, message=message, label=label, indent=indent
    )


_Diagnostic = Tuple[Optional[Tuple[str, int, int]], str]
_DiagnosticGroup = Tuple[_Diagnostic, List[_Diagnostic]]

# Strip the leading `'op_name' op ` prefix MLIR adds for op-level
# diagnostics so the user sees the message we wrote.
_OP_PREFIX = re.compile(r"'[^']+'\s+op\s+(.*)$")


def _extract_diagnostic_groups(error_msg: str) -> List[_DiagnosticGroup]:
    """Split an MLIR error string into groups of (error, [notes]).

    MLIR emits one diagnostic per line in the form
        <file>:<line>:<col>: <kind>: <message>
    with `<kind>` in {error, note, warning}. Notes are attached to the
    most recent error/warning that preceded them, so a stream like
        error: A
        note: about A
        error: B
        note: about B
    parses as two groups: (A, [note about A]) and (B, [note about B]).
    Continuation lines (e.g. the IR dump under `see current operation`)
    don't start with a recognized kind keyword and are skipped.
    """
    groups: List[_DiagnosticGroup] = []

    # MLIR diagnostics from the Python pass manager are emitted as
    #     <kind>: "path":line:col: <message>
    # (notes sometimes carry a leading space). Raw `ttlang-opt` runs
    # emit
    #     path:line:col: <kind>: <message>
    # Both forms are accepted. `loc_first` is tried before `kind_first`
    # because its location segment is mandatory (most-specific match
    # wins); reordering would silently change behavior on the
    # path-first variant.
    kind_first = re.compile(
        r"^\s*(?P<kind>error|note|warning):\s*"
        r"(?:\"?(?P<file>[^\"\n]+?)\"?:(?P<line>\d+):(?P<col>\d+):\s*)?"
        r"(?P<msg>.*)$"
    )
    loc_first = re.compile(
        r"^(?:\"?(?P<file>[^\"\n]+?)\"?:(?P<line>\d+):(?P<col>\d+):\s*)"
        r"(?P<kind>error|note|warning):\s*(?P<msg>.*)$"
    )

    for raw in error_msg.splitlines():
        match = loc_first.match(raw) or kind_first.match(raw)
        if not match:
            continue
        kind = match.group("kind")
        msg = match.group("msg").strip()
        op_match = _OP_PREFIX.match(msg)
        if op_match:
            msg = op_match.group(1).strip()

        if match.group("file"):
            loc = (
                match.group("file"),
                int(match.group("line")),
                int(match.group("col")),
            )
        else:
            loc = None

        if kind in ("error", "warning"):
            current_notes: List[_Diagnostic] = []
            groups.append(((loc, msg), current_notes))
        else:  # note
            # Notes before any error are dropped — there's nothing for
            # them to attach to. In practice MLIR doesn't emit those.
            if groups:
                current_notes.append((loc, msg))

    return groups


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
