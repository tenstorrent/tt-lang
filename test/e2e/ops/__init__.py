# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Op test base classes for E2E tests.

Provides class-based test infrastructure for elementwise operations.
Test classes are auto-generated from TTLElementwiseOps.def.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pytest
import torch
from torch import Tensor

from ..base import E2ETestBase
from ..config import E2EConfig


# Map op names to torch reference functions.
OP_TORCH_MAP: Dict[str, Callable[..., Tensor]] = {
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

# Domain constraints for ops that require specific input ranges.
OP_INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    "log": (0.01, 10.0),  # log requires positive inputs
    "sqrt": (0.01, 10.0),  # sqrt requires positive inputs
    "rsqrt": (0.01, 10.0),  # rsqrt requires positive inputs
}


def _parse_elementwise_ops_def() -> Dict[str, int]:
    """
    Parse TTLElementwiseOps.def to get op name -> arity.

    Returns:
        Dict mapping op name (lowercase) to arity (1 or 2).
    """
    def_path = (
        Path(__file__).parent.parent.parent.parent
        / "include/ttlang/Dialect/TTL/TTLElementwiseOps.def"
    )

    if not def_path.exists():
        return {}

    ops: Dict[str, int] = {}
    with open(def_path) as f:
        for line in f:
            # Match TTL_BINARY_TILE_OP(Add, AddTileOp)
            if match := re.match(r"TTL_BINARY_TILE_OP\((\w+),\s*\w+\)", line):
                ops[match.group(1).lower()] = 2
            # Match TTL_UNARY_TILE_OP(Exp, ExpTileOp)
            elif match := re.match(r"TTL_UNARY_TILE_OP\((\w+),\s*\w+\)", line):
                ops[match.group(1).lower()] = 1

    return ops


# Parse ops from .def file at module load time.
ELEMENTWISE_OPS: Dict[str, int] = _parse_elementwise_ops_def()


class OpTestBase(E2ETestBase):
    """
    Base for auto-generated op tests.

    Subclasses define a single class attribute OP_STR to specify the operation.
    All other behavior is inherited and can be overridden as needed.
    """

    # Class attributes - override in subclasses
    OP_STR: str  # "add", "exp", etc.
    ARITY: int  # 1 or 2 (set by UnaryOpTestBase/BinaryOpTestBase)
    INPUT_SHAPE = (2, 2)  # Grid shape in tiles
    INPUT_DTYPE = torch.bfloat16

    # Comparison tolerance (auto-computed from dtype if None)
    ULP_THRESHOLD: Optional[float] = None

    # Input value range
    MIN_VALUE = -1.0
    MAX_VALUE = 1.0

    # Override for ops with domain constraints (e.g., sqrt requires positive inputs)
    INPUT_RANGE: Optional[Tuple[float, float]] = None

    @pytest.fixture(scope="class")
    def torch_op(self) -> Callable[..., Tensor]:
        """Get torch reference function from OP_STR."""
        if self.OP_STR not in OP_TORCH_MAP:
            pytest.skip(f"No torch reference for {self.OP_STR}")
        return OP_TORCH_MAP[self.OP_STR]

    @pytest.fixture(scope="class")
    def config(self) -> E2EConfig:
        """Get test configuration."""
        return E2EConfig(
            grid_shape=self.INPUT_SHAPE,
            dtype=self.INPUT_DTYPE,
        )

    @pytest.fixture(scope="class")
    def input_range(self) -> Tuple[float, float]:
        """Get input value range."""
        return self.INPUT_RANGE or (self.MIN_VALUE, self.MAX_VALUE)

    @pytest.mark.order(1)
    def test_build_module(
        self, config: E2EConfig, input_range: Tuple[float, float]
    ) -> None:
        """Build TTL module from OP_STR."""
        from ..ttl_builder import build_ttl_module

        # Generate random inputs.
        lo, hi = input_range
        torch_inputs: List[Tensor] = []
        for _ in range(self.ARITY):
            t = torch.rand(config.tensor_shape, dtype=config.dtype) * (hi - lo) + lo
            torch_inputs.append(t)

        # Build module.
        module = build_ttl_module(self.OP_STR, self.ARITY, config, torch_inputs)
        assert module is not None

        # Save module to file for subsequent stages.
        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(str(module))

        # Save inputs for golden comparison.
        torch.save(torch_inputs, self.output_file("inputs.pt"))

        # Verify MLIR contains expected operation.
        mlir_str = str(module)
        assert (
            f"ttl.{self.OP_STR}" in mlir_str
            or f"func.func @compute_{self.OP_STR}" in mlir_str
        )


class UnaryOpTestBase(OpTestBase):
    """Base for unary operations (1 input, 1 output)."""

    ARITY = 1


class BinaryOpTestBase(OpTestBase):
    """Base for binary operations (2 inputs, 1 output)."""

    ARITY = 2


def generate_op_test_classes() -> Dict[str, Type[OpTestBase]]:
    """
    Auto-generate test classes from TTLElementwiseOps.def.

    Returns:
        Dict mapping class name (e.g., "TestAdd") to the generated class.
    """
    generated: Dict[str, Type[OpTestBase]] = {}

    for op_name, arity in ELEMENTWISE_OPS.items():
        # Determine base class from arity.
        base: Type[OpTestBase] = UnaryOpTestBase if arity == 1 else BinaryOpTestBase

        # Build class attributes.
        attrs: Dict[str, Any] = {"OP_STR": op_name}
        if op_name in OP_INPUT_RANGES:
            attrs["INPUT_RANGE"] = OP_INPUT_RANGES[op_name]

        # Create class dynamically.
        class_name = f"Test{op_name.capitalize()}"
        generated[class_name] = type(class_name, (base,), attrs)  # type: ignore[assignment]

    return generated


# Auto-generated test classes from .def file.
GENERATED_OP_TESTS: Dict[str, Type[OpTestBase]] = generate_op_test_classes()
