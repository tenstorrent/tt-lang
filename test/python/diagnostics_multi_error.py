# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s

"""FileCheck test for `format_mlir_error` multi-error rendering.

Feeds a synthetic MLIR diagnostic stream with two unrelated errors
(each with attached notes) through the formatter and verifies the
rendered output. Line numbers embedded in the diagnostic are printed
in a preamble so CHECKs can capture them with `[[VAR]]` instead of
hardcoding.
"""

import textwrap

from ttl.diagnostics import format_mlir_error

ANCHOR_FIRST = "FIRST_ERROR_ANCHOR"
ANCHOR_FIRST_DECLARED = "FIRST_DECLARED_ANCHOR"
ANCHOR_SECOND = "SECOND_ERROR_ANCHOR"
ANCHOR_SECOND_DECLARED = "SECOND_DECLARED_ANCHOR"


def _line_of(anchor: str) -> int:
    with open(__file__, "r") as fd:
        for idx, raw in enumerate(fd, start=1):
            # Skip the constant definitions; we want the callsites below.
            if raw.lstrip().startswith("ANCHOR_"):
                continue
            if anchor in raw:
                return idx
    raise RuntimeError(f"anchor {anchor!r} not found in {__file__}")


def main() -> None:
    first_line = _line_of(ANCHOR_FIRST)  # FIRST_ERROR_ANCHOR
    first_decl_line = _line_of(ANCHOR_FIRST_DECLARED)  # FIRST_DECLARED_ANCHOR
    second_line = _line_of(ANCHOR_SECOND)  # SECOND_ERROR_ANCHOR
    second_decl_line = _line_of(ANCHOR_SECOND_DECLARED)  # SECOND_DECLARED_ANCHOR

    print(f"FIRST_LINE={first_line}")
    print(f"FIRST_DECL_LINE={first_decl_line}")
    print(f"SECOND_LINE={second_line}")
    print(f"SECOND_DECL_LINE={second_decl_line}")

    diagnostic_stream = textwrap.dedent(
        f"""\
        Failure while executing pass pipeline:
        error: "{__file__}":{first_line}:1: 'fake.op' op first violation
         note: "{__file__}":{first_line}:1: see current operation: %0 = "fake.op"() : () -> ()
         note: "{__file__}":{first_line}:1: first witness here
         note: "{__file__}":{first_decl_line}:1: first thing declared here
         note: "{__file__}":{first_decl_line}:1: first thing declared here
        error: "{__file__}":{second_line}:1: 'fake.op' op second violation
         note: "{__file__}":{second_line}:1: second witness here
         note: "{__file__}":{second_decl_line}:1: second thing declared here
        """
    )
    print(format_mlir_error(diagnostic_stream))

    # ===== single error, no notes =====
    print("=== SINGLE ===")
    print(
        format_mlir_error(
            f'error: "{__file__}":{first_line}:1: lone error with no notes'
        )
    )

    # ===== error with no location prefix =====
    print("=== NOLOC ===")
    print(format_mlir_error("error: bare error message with no location"))

    # ===== leading note (before any error) is dropped =====
    print("=== LEADING_NOTE ===")
    print(
        format_mlir_error(
            f'note: "{__file__}":{first_line}:1: orphan note with no parent\n'
            f'error: "{__file__}":{first_line}:1: real error after orphan note'
        )
    )


if __name__ == "__main__":
    main()


# Capture the line numbers the script embedded.
# CHECK: FIRST_LINE=[[FIRST_LINE:[0-9]+]]
# CHECK: FIRST_DECL_LINE=[[FIRST_DECL_LINE:[0-9]+]]
# CHECK: SECOND_LINE=[[SECOND_LINE:[0-9]+]]
# CHECK: SECOND_DECL_LINE=[[SECOND_DECL_LINE:[0-9]+]]

# First error: op-prefix stripped, source context shown.
# CHECK: error: first violation
# CHECK-NEXT: --> {{.*}}diagnostics_multi_error.py:[[FIRST_LINE]]:1
# CHECK: FIRST_ERROR_ANCHOR

# Notes indent two spaces; `see current operation` dropped; duplicate
# `first thing declared here` rendered once.
# CHECK: {{^  note: first witness here}}
# CHECK-NOT: see current operation
# CHECK: {{^  note: first thing declared here}}
# CHECK: --> {{.*}}diagnostics_multi_error.py:[[FIRST_DECL_LINE]]:1
# CHECK: FIRST_DECLARED_ANCHOR
# CHECK-NOT: first thing declared here

# Second error renders as its own `error:`, not as a note.
# CHECK: error: second violation
# CHECK-NEXT: --> {{.*}}diagnostics_multi_error.py:[[SECOND_LINE]]:1
# CHECK: SECOND_ERROR_ANCHOR
# CHECK: {{^  note: second witness here}}
# CHECK: {{^  note: second thing declared here}}
# CHECK: --> {{.*}}diagnostics_multi_error.py:[[SECOND_DECL_LINE]]:1
# CHECK: SECOND_DECLARED_ANCHOR

# Single error, no notes — base case.
# CHECK: === SINGLE ===
# CHECK: error: lone error with no notes
# CHECK-NEXT: --> {{.*}}diagnostics_multi_error.py:[[FIRST_LINE]]:1

# Error with no location prefix falls back to the bare label-and-message line.
# CHECK: === NOLOC ===
# CHECK-NEXT: error: bare error message with no location

# A note before any error is dropped (no parent to attach to); the
# subsequent error still renders.
# CHECK: === LEADING_NOTE ===
# CHECK-NOT: orphan note with no parent
# CHECK: error: real error after orphan note
