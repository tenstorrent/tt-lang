# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation specifications for middle-end tests.

Each spec defines an operation to test, including its TTL op name,
arity, golden reference function, and reader pattern type.

This module is the single source of truth for operation definitions.

Supports both single operations (ComputeOpSpec) and fused multi-op
kernels (FusedOpSpec) where operations are applied in sequence.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Union, Optional

import torch


@dataclass(frozen=True)
class ComputeOpSpec:
    """Specification for a single compute operation test."""

    name: str  # Human-readable name: "add", "exp", etc.
    ttl_op: str  # TTL operation name: "tile_add", "tile_exp", etc.
    arity: int  # Number of inputs: 1=unary, 2=binary
    golden: Callable[..., torch.Tensor]  # Torch reference function
    reader_type: str  # Reader pattern: "unary", "binary", "ternary"
    # Input constraints: (min, max) range for generating test inputs.
    # Useful for ops like sqrt/rsqrt that require positive inputs.
    input_range: Optional[Tuple[float, float]] = None

    @property
    def ttl_ops(self) -> List[str]:
        """Return ops as a list for uniform handling with FusedOpSpec."""
        return [self.ttl_op]

    @property
    def is_fused(self) -> bool:
        return False

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class FusedOpSpec:
    """
    Specification for a fused multi-op compute kernel.

    Operations are applied in sequence: output of op[i] becomes input to op[i+1].
    The first operation determines how many tensor inputs are consumed.

    Example:
        FusedOpSpec(
            name="add_exp",
            ttl_ops=["tile_add", "tile_exp"],
            arity=2,
            golden=lambda a, b: torch.exp(a + b),
            reader_type="binary",
        )
    """

    name: str  # Human-readable name: "add_exp", "gelu_approx", etc.
    ttl_ops: List[str]  # Sequence of TTL operations: ["tile_add", "tile_exp"]
    arity: int  # Number of tensor inputs (from first op)
    golden: Callable[..., torch.Tensor]  # Composed torch reference function
    reader_type: str  # Reader pattern: "unary", "binary", "ternary"
    # Input constraints: (min, max) range for generating test inputs.
    input_range: Optional[Tuple[float, float]] = None

    @property
    def ttl_op(self) -> str:
        """Return first op for compatibility."""
        return self.ttl_ops[0]

    @property
    def is_fused(self) -> bool:
        return True

    def __str__(self) -> str:
        return self.name


# Type alias for any compute spec.
AnyOpSpec = Union[ComputeOpSpec, FusedOpSpec]


# Binary operations - two inputs, one output.
BINARY_OPS = [
    ComputeOpSpec("add", "tile_add", 2, lambda a, b: a + b, "binary"),
    ComputeOpSpec("sub", "tile_sub", 2, lambda a, b: a - b, "binary"),
    ComputeOpSpec("mul", "tile_mul", 2, lambda a, b: a * b, "binary"),
    ComputeOpSpec("max", "tile_max", 2, torch.maximum, "binary"),
]

# Unary operations - one input, one output.
UNARY_OPS = [
    ComputeOpSpec("exp", "tile_exp", 1, torch.exp, "unary"),
    # sqrt requires positive inputs to avoid NaN.
    ComputeOpSpec(
        "sqrt", "tile_sqrt", 1, torch.sqrt, "unary", input_range=(0.01, 10.0)
    ),
    ComputeOpSpec("relu", "tile_relu", 1, torch.relu, "unary"),
    ComputeOpSpec("neg", "tile_neg", 1, torch.neg, "unary"),
    ComputeOpSpec("abs", "tile_abs", 1, torch.abs, "unary"),
    # rsqrt requires positive inputs to avoid NaN/Inf.
    ComputeOpSpec(
        "rsqrt", "tile_rsqrt", 1, torch.rsqrt, "unary", input_range=(0.01, 10.0)
    ),
    ComputeOpSpec("tanh", "tile_tanh", 1, torch.tanh, "unary"),
    ComputeOpSpec("sigmoid", "tile_sigmoid", 1, torch.sigmoid, "unary"),
]

# Fused operations - multiple ops applied in sequence.
FUSED_OPS: List[FusedOpSpec] = [
    # Binary -> Unary fusions.
    FusedOpSpec(
        name="add_exp",
        ttl_ops=["tile_add", "tile_exp"],
        arity=2,
        golden=lambda a, b: torch.exp(a + b),
        reader_type="binary",
    ),
    FusedOpSpec(
        name="mul_relu",
        ttl_ops=["tile_mul", "tile_relu"],
        arity=2,
        golden=lambda a, b: torch.relu(a * b),
        reader_type="binary",
    ),
    FusedOpSpec(
        name="add_sigmoid",
        ttl_ops=["tile_add", "tile_sigmoid"],
        arity=2,
        golden=lambda a, b: torch.sigmoid(a + b),
        reader_type="binary",
    ),
    FusedOpSpec(
        name="sub_abs",
        ttl_ops=["tile_sub", "tile_abs"],
        arity=2,
        golden=lambda a, b: torch.abs(a - b),
        reader_type="binary",
    ),
    # Unary -> Unary fusions.
    FusedOpSpec(
        name="exp_sqrt",
        ttl_ops=["tile_exp", "tile_sqrt"],
        arity=1,
        golden=lambda x: torch.sqrt(torch.exp(x)),
        reader_type="unary",
    ),
    FusedOpSpec(
        name="relu_neg",
        ttl_ops=["tile_relu", "tile_neg"],
        arity=1,
        golden=lambda x: torch.neg(torch.relu(x)),
        reader_type="unary",
    ),
    # Longer chains.
    FusedOpSpec(
        name="add_relu_neg",
        ttl_ops=["tile_add", "tile_relu", "tile_neg"],
        arity=2,
        golden=lambda a, b: torch.neg(torch.relu(a + b)),
        reader_type="binary",
    ),
    FusedOpSpec(
        name="sigmoid_tanh",
        ttl_ops=["tile_sigmoid", "tile_tanh"],
        arity=1,
        golden=lambda x: torch.tanh(torch.sigmoid(x)),
        reader_type="unary",
    ),
]

# All compute operations (single + fused).
COMPUTE_OPS: List[AnyOpSpec] = list(BINARY_OPS) + list(UNARY_OPS)
ALL_OPS: List[AnyOpSpec] = list(COMPUTE_OPS) + list(FUSED_OPS)


def get_op_by_name(name: str) -> AnyOpSpec:
    """Look up an operation spec by name (searches all ops including fused)."""
    for op in ALL_OPS:
        if op.name == name:
            return op
    raise ValueError(f"Unknown operation: {name}")


def get_ttl_tile_op_func(ttl_module: Any, ttl_op_name: str) -> Callable:
    """
    Get the TTL tile operation function from the ttl module.

    Args:
        ttl_module: The ttlang.dialects.ttl module.
        ttl_op_name: The operation name (e.g., "tile_add", "tile_exp").

    Returns:
        The TTL operation function.

    Raises:
        ValueError: If the operation is not found.
    """
    if not hasattr(ttl_module, ttl_op_name):
        raise ValueError(
            f"TTL operation '{ttl_op_name}' not found in ttl module. "
            f"Available ops: {[op.ttl_op for op in COMPUTE_OPS]}"
        )
    return getattr(ttl_module, ttl_op_name)


# Map of ttl_op names to their arities for validation.
TTL_OP_ARITY: Dict[str, int] = {op.ttl_op: op.arity for op in COMPUTE_OPS}
