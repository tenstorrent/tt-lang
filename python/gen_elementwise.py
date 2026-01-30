#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generator script for elementwise operation wrappers.

Parses TTLElementwiseOps.def and generates Python wrapper functions
that call the corresponding TTL dialect operations.

Usage:
    python gen_elementwise.py <def_file> -o <output_file>
"""

import argparse
import re
from pathlib import Path

HEADER = '''\
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-generated elementwise operation wrappers.

DO NOT EDIT - Generated from TTLElementwiseOps.def by gen_elementwise.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .dialects import ttl
from ._src.ttl_ast import syntax

if TYPE_CHECKING:
    from .operators import TensorBlock

'''

BINARY_OP_TEMPLATE = '''\
@syntax("{name}")
def {name}(lhs: "TensorBlock", rhs: "TensorBlock") -> "TensorBlock":
    """Element-wise {name} operation."""
    return ttl.{name}(lhs.type, lhs, rhs)

'''

UNARY_OP_TEMPLATE = '''\
@syntax("{name}")
def {name}(input: "TensorBlock") -> "TensorBlock":
    """Element-wise {name} operation."""
    return ttl.{name}(input.type, input)

'''

ALL_TEMPLATE = """\
__all__ = [
{entries}
]
"""


def parse_def_file(def_path: Path) -> tuple[list[str], list[str]]:
    """Parse TTLElementwiseOps.def and extract operation names.

    Handles both standard and special binary ops:
    - TTL_BINARY_TILE_OP(Name, TileOp, TTKInit, TTKCompute)
    - TTL_BINARY_TILE_OP_SPECIAL(Name, TileOp, TTKInit, TTKCompute)
    - TTL_UNARY_TILE_OP(Name, TileOp, TTKInit, TTKCompute)
    """
    content = def_path.read_text()

    binary_ops = []
    unary_ops = []

    # Match TTL_BINARY_TILE_OP(Name, ...), TTL_BINARY_TILE_OP_SPECIAL(Name, ...),
    # and TTL_BINARY_TILE_OP_MINMAX(Name, ...) but skip #define lines
    for match in re.finditer(
        r"^TTL_BINARY_TILE_OP(?:_SPECIAL|_MINMAX)?\((\w+),", content, re.MULTILINE
    ):
        name = match.group(1).lower()
        # Skip macro parameter names (lowercase indicates it's a parameter)
        if name[0].isupper() or name not in (
            "ttl_op",
            "tile_op",
            "ttk_init",
            "ttk_compute",
        ):
            binary_ops.append(name)

    # Match TTL_UNARY_TILE_OP(Name, ...) but skip #define lines
    for match in re.finditer(r"^TTL_UNARY_TILE_OP\((\w+),", content, re.MULTILINE):
        name = match.group(1).lower()
        # Skip macro parameter names
        if name[0].isupper() or name not in (
            "ttl_op",
            "tile_op",
            "ttk_init",
            "ttk_compute",
        ):
            unary_ops.append(name)

    return binary_ops, unary_ops


def generate_python(binary_ops: list[str], unary_ops: list[str]) -> str:
    """Generate Python source code for the operations."""
    lines = [HEADER]

    # Generate binary operations
    if binary_ops:
        lines.append("# Binary elementwise operations\n")
        for name in binary_ops:
            lines.append(BINARY_OP_TEMPLATE.format(name=name))

    # Generate unary operations
    if unary_ops:
        lines.append("# Unary elementwise operations\n")
        for name in unary_ops:
            lines.append(UNARY_OP_TEMPLATE.format(name=name))

    # Generate __all__
    all_ops = binary_ops + unary_ops
    entries = ",\n".join(f'    "{op}"' for op in all_ops)
    lines.append(ALL_TEMPLATE.format(entries=entries))

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python elementwise op wrappers from TTLElementwiseOps.def"
    )
    parser.add_argument("def_file", type=Path, help="Path to TTLElementwiseOps.def")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output Python file"
    )

    args = parser.parse_args()

    if not args.def_file.exists():
        raise FileNotFoundError(f"Definition file not found: {args.def_file}")

    binary_ops, unary_ops = parse_def_file(args.def_file)

    print(f"Found {len(binary_ops)} binary ops: {binary_ops}")
    print(f"Found {len(unary_ops)} unary ops: {unary_ops}")

    python_code = generate_python(binary_ops, unary_ops)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(python_code)

    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
