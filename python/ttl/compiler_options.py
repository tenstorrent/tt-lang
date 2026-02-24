# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compiler pipeline options for kernel compilation.

To add a new boolean option:
  1. Add a field with a default to the CompilerOptions dataclass.
  2. Add an ``add_argument()`` call in ``_make_parser()``.
     ``BooleanOptionalAction`` generates ``--flag``/``--no-flag`` automatically.
Parsing, argv extraction, and merge logic require no changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Optional


def _make_parser() -> argparse.ArgumentParser:
    """Build the compiler options parser."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--maximize-dst",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable DST maximization via subblock compute.",
    )
    p.add_argument(
        "--fpu-binary-ops",
        default=True,
        dest="enable_fpu_binary_ops",
        action=argparse.BooleanOptionalAction,
        help="Use FPU for binary add/sub/mul.",
    )
    return p


_PARSER = _make_parser()


@dataclasses.dataclass(frozen=True)
class CompilerOptions:
    """Compiler pipeline options for kernel compilation.

    Frozen so it's hashable and usable directly as a cache key component.
    Does NOT include TTNN compute config (fp32_dest_acc_en, dst_full_sync_en).
    """

    maximize_dst: bool = True
    enable_fpu_binary_ops: bool = True

    @staticmethod
    def from_string(options: Optional[str] = None) -> CompilerOptions:
        """Parse an option string (e.g., "--no-maximize-dst").

        Later tokens override earlier ones. Returns defaults when
        *options* is `None` or empty.
        """
        tokens = options.split() if options else []
        ns, unknown = _PARSER.parse_known_args(tokens)
        if unknown:
            raise ValueError(f"Unknown kernel option(s): {unknown}")
        return CompilerOptions(
            maximize_dst=ns.maximize_dst,
            enable_fpu_binary_ops=ns.enable_fpu_binary_ops,
        )

    @staticmethod
    def from_argv() -> CompilerOptions:
        """Extract compiler options from `sys.argv`, ignoring
        unrecognised arguments (test runner flags, file paths, etc.)."""
        ns, _ = _PARSER.parse_known_args(sys.argv[1:])
        return CompilerOptions(
            maximize_dst=ns.maximize_dst,
            enable_fpu_binary_ops=ns.enable_fpu_binary_ops,
        )

    @staticmethod
    def usage() -> str:
        """Return a help string describing all available compiler options."""
        return _PARSER.format_help()

    def merge(self, overrides: CompilerOptions) -> CompilerOptions:
        """Return a new CompilerOptions where `overrides` takes priority
        for any field that differs from its default."""
        defaults = CompilerOptions()
        kwargs = {}
        for f in dataclasses.fields(self):
            base = getattr(self, f.name)
            over = getattr(overrides, f.name)
            # If the override is non-default, use it; otherwise keep base.
            kwargs[f.name] = over if over != f.default else base
        return CompilerOptions(**kwargs)
