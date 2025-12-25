# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-generate op specifications from TTLElementwiseOps.def.

Parses the .def file to extract operation names and builds the OP_TORCH_MAP.
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import torch


def parse_elementwise_ops_def() -> Dict[str, Tuple[str, int]]:
    """
    Parse TTLElementwiseOps.def to get op name -> (tile_op, arity).

    Returns:
        Dict mapping op name (lowercase) to (TileOp name, arity).
    """
    def_path = (
        Path(__file__).parent.parent.parent
        / "include/ttlang/Dialect/TTL/TTLElementwiseOps.def"
    )

    if not def_path.exists():
        return {}

    ops = {}
    with open(def_path) as f:
        for line in f:
            # Match TTL_BINARY_TILE_OP(Add, AddTileOp)
            if match := re.match(r"TTL_BINARY_TILE_OP\((\w+),\s*(\w+)\)", line):
                op_name = match.group(1).lower()
                tile_op = match.group(2)
                ops[op_name] = (tile_op, 2)
            # Match TTL_UNARY_TILE_OP(Exp, ExpTileOp)
            elif match := re.match(r"TTL_UNARY_TILE_OP\((\w+),\s*(\w+)\)", line):
                op_name = match.group(1).lower()
                tile_op = match.group(2)
                ops[op_name] = (tile_op, 1)

    return ops


# Map op names to torch reference functions
OP_TORCH_MAP = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "max": torch.maximum,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "neg": torch.neg,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}

# Parse ops from .def file
ELEMENTWISE_OPS = parse_elementwise_ops_def()
