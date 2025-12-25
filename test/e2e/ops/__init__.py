# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Op test base classes for E2E tests.

Provides class-based test infrastructure for elementwise operations.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pytest
import torch

from ..base import E2ETestBase
from ..config import E2EConfig, SMOKE_CONFIGS
from ..ops_gen import OP_TORCH_MAP


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
    def torch_op(self) -> Callable:
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
    def test_build_module(self, config, input_range):
        """Build TTL module from OP_STR."""
        from ..ttl_builder import build_ttl_module

        # Generate random inputs
        lo, hi = input_range
        torch_inputs = []
        for _ in range(self.ARITY):
            t = torch.rand(config.tensor_shape, dtype=config.dtype) * (hi - lo) + lo
            torch_inputs.append(t)

        # Build module
        module = build_ttl_module(self.OP_STR, self.ARITY, config, torch_inputs)
        assert module is not None

        # Save module to file for subsequent stages
        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(str(module))

        # Save inputs for golden comparison
        torch.save(torch_inputs, self.output_file("inputs.pt"))

        # Verify MLIR contains expected operation
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

