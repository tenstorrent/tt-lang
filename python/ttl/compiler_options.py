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
import os
import sys
from typing import Optional, Sequence


def _make_parser() -> argparse.ArgumentParser:
    """Build the compiler options parser.

    Defaults are ``None`` so callers can distinguish "not specified" from
    "explicitly set to the dataclass default".
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--ttl-maximize-dst",
        default=None,
        dest="maximize_dst",
        action=argparse.BooleanOptionalAction,
        help="Enable DST maximization via subblock compute and scheduling (default: enabled).",
    )
    p.add_argument(
        "--ttl-fpu-binary-ops",
        default=None,
        dest="enable_fpu_binary_ops",
        action=argparse.BooleanOptionalAction,
        help="Use FPU for binary add/sub/mul (default: enabled).",
    )
    p.add_argument(
        "--ttl-block-matmul",
        default=None,
        dest="use_block_matmul",
        action=argparse.BooleanOptionalAction,
        help="Lower matmul to block-level hardware calls (default: enabled).",
    )
    p.add_argument(
        "--ttl-combine-pack-tiles",
        default=None,
        dest="combine_pack_tiles",
        action=argparse.BooleanOptionalAction,
        help="Combine consecutive pack_tile ops into pack_tile_block (default: enabled).",
    )
    return p


_PARSER = _make_parser()

# Cached result from the first from_argv() call.  We assume sys.argv does
# not change after process startup, so a single parse is sufficient.
_argv_result: Optional[CompilerOptions] = None


def _parse_explicit(tokens: Sequence[str], *, reject_unknown: bool = False) -> dict:
    """Parse *tokens* and return only the fields that were explicitly set."""
    if reject_unknown:
        ns, unknown = _PARSER.parse_known_args(tokens)
        if unknown:
            raise ValueError(f"Unknown kernel option(s): {unknown}")
    else:
        ns, _ = _PARSER.parse_known_args(tokens)
    return {k: v for k, v in vars(ns).items() if v is not None}


@dataclasses.dataclass(frozen=True)
class CompilerOptions:
    """Compiler pipeline options for kernel compilation.

    Frozen so it's hashable and usable directly as a cache key component.
    Does NOT include TTNN compute config (fp32_dest_acc_en, dst_full_sync_en).

    Priority ordering (highest wins)::

        sys.argv  >  TTLANG_COMPILER_OPTIONS env var  >  decorator ``options=``

    The call site in ``ttl_api.py`` builds a base from the decorator string
    and env var, then merges ``from_argv()`` on top.  ``merge()`` only
    applies fields that were explicitly set in the override, so unmentioned
    flags fall through from the base.
    """

    maximize_dst: bool = True
    enable_fpu_binary_ops: bool = True
    use_block_matmul: bool = True
    combine_pack_tiles: bool = True

    # Fields that were explicitly provided (not defaulted). Excluded from
    # equality and hashing so two instances with the same bool values are
    # interchangeable for caching regardless of how they were constructed.
    _explicit: frozenset = dataclasses.field(
        default=frozenset(), compare=False, hash=False, repr=False
    )

    @staticmethod
    def from_string(options: Optional[str] = None) -> CompilerOptions:
        """Parse an option string (e.g., "--no-ttl-maximize-dst").

        Later tokens override earlier ones. Returns defaults when
        *options* is `None` or empty.  Raises ``ValueError`` on
        unrecognised tokens.
        """
        tokens = options.split() if options else []
        explicit = _parse_explicit(tokens, reject_unknown=True)
        return CompilerOptions(**explicit, _explicit=frozenset(explicit))

    @staticmethod
    def from_argv() -> CompilerOptions:
        """Extract compiler options from `sys.argv`, ignoring
        unrecognised arguments (test runner flags, file paths, etc.).

        The result is cached on first call and reused for subsequent calls
        (argv is assumed stable for the lifetime of the process).

        If ``--ttl-help`` is present, prints available compiler options and
        exits.
        """
        global _argv_result
        if _argv_result is not None:
            return _argv_result

        if "--ttl-help" in sys.argv[1:]:
            print("TTL compiler options:\n")
            print(CompilerOptions.usage())
            sys.exit(0)
        explicit = _parse_explicit(sys.argv[1:])
        _argv_result = CompilerOptions(**explicit, _explicit=frozenset(explicit))
        return _argv_result

    @staticmethod
    def usage() -> str:
        """Return a help string describing all available compiler options."""
        return _PARSER.format_help()

    def merge(self, overrides: CompilerOptions) -> CompilerOptions:
        """Return a new CompilerOptions where explicitly-set fields in
        *overrides* take priority over *self*."""
        kwargs = {}
        explicit = set(self._explicit)
        for f in dataclasses.fields(self):
            if f.name.startswith("_"):
                continue
            if f.name in overrides._explicit:
                kwargs[f.name] = getattr(overrides, f.name)
                explicit.add(f.name)
            else:
                kwargs[f.name] = getattr(self, f.name)
        return CompilerOptions(**kwargs, _explicit=frozenset(explicit))
