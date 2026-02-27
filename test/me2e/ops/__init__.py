# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Op test base classes for ME2E tests.

Provides class-based test infrastructure for elementwise operations.
Test classes are auto-generated from TTLElementwiseOps.def.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

import pytest
import torch
from torch import Tensor

from ..base import ME2ETestBase
from ..config import E2EConfig


# Map op names to torch reference functions.
OP_TORCH_MAP: Dict[str, Callable[..., Tensor]] = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "max": torch.maximum,
    "min": torch.minimum,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "neg": torch.neg,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "floor": torch.floor,
    "recip": torch.reciprocal,
}

# Domain constraints for ops that require specific input ranges.
OP_INPUT_RANGES: Dict[str, Tuple[float, float]] = {
    "log": (0.01, 10.0),  # log requires positive inputs
    "sqrt": (0.01, 10.0),  # sqrt requires positive inputs
    "rsqrt": (0.01, 10.0),  # rsqrt requires positive inputs
    "recip": (0.01, 10.0),  # recip requires non-zero inputs
    "div": (0.01, 10.0),  # div requires non-zero divisor
}

# Per-op ULP threshold overrides keyed by dtype.
# Used for ops where the SFPU implementation has known precision limitations.
OP_ULP_THRESHOLD_OVERRIDES: Dict[str, Dict[torch.dtype, int]] = {
    # tt-metal SFPU log f32 uses a polynomial approximation with insufficient
    # precision for f32. tt-metal's own tests skip log f32 on WH/BH due to
    # "very high abs and relative diff". Measured ULP ~2^21.
    "log": {torch.float32: 2**22},
    # FPU binary f32: hardware computes at reduced precision.  PCC > 0.99999
    # and allclose(rtol=1e-3, atol=1e-3) passes, but near-zero values inflate
    # ULP distance.  Measured max ULP: add ~2^23.5, sub ~2^23.5, mul ~2^15.2.
    "add": {torch.float32: 2**24},
    "sub": {torch.float32: 2**24},
    "mul": {torch.float32: 2**16},
}

# Per-op PCC threshold overrides keyed by dtype.
# Default is 0.9999 (consistent with tt-metal). Override for ops where
# hardware precision limitations reduce correlation.
# Values below are aligned with tt-metal test thresholds.
OP_PCC_THRESHOLD_OVERRIDES: Dict[str, Dict[torch.dtype, float]] = {
    "exp": {torch.float32: 0.9998},
    "tanh": {torch.float32: 0.993},
    "recip": {torch.bfloat16: 0.999},
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
            # Match TTL_BINARY_TILE_OP(Add, ...) or TTL_BINARY_TILE_OP_MINMAX(Max, ...)
            if match := re.match(r"TTL_BINARY_TILE_OP(?:_MINMAX)?\((\w+),\s*\w+", line):
                ops[match.group(1).lower()] = 2
            # Match TTL_UNARY_TILE_OP(Exp, ExpTileOp, ...)
            elif match := re.match(r"TTL_UNARY_TILE_OP\((\w+),\s*\w+", line):
                ops[match.group(1).lower()] = 1

    return ops


# Parse ops from .def file at module load time.
ELEMENTWISE_OPS: Dict[str, int] = _parse_elementwise_ops_def()


class OpTestBase(ME2ETestBase):
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

    # Comparison tolerances (auto-computed from dtype if None)
    ULP_THRESHOLD: Optional[float] = None
    PCC_THRESHOLD: Optional[float] = None

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
        self,
        config: E2EConfig,
        input_range: Tuple[float, float],
        torch_op: Callable[..., Tensor],
    ) -> None:
        """Build full ME2E TTL module (reader, compute, writer) from OP_STR."""
        import os
        from ..builder.ttl_builder import build_e2e_module_mlir

        # Set seed for reproducible test inputs.
        seed = int(os.environ.get("TTLANG_TEST_SEED", "42"))
        torch.manual_seed(seed)

        # Generate random inputs.
        lo, hi = input_range
        torch_inputs: List[Tensor] = []
        for _ in range(self.ARITY):
            t = torch.rand(config.tensor_shape, dtype=config.dtype) * (hi - lo) + lo
            torch_inputs.append(t)

        # Compute golden using torch.
        if self.ARITY == 1:
            golden = torch_op(torch_inputs[0])
        else:
            golden = torch_op(torch_inputs[0], torch_inputs[1])

        # Build full ME2E module with reader, compute, and writer threads.
        mlir_str = build_e2e_module_mlir(self.OP_STR, self.ARITY, config)
        assert mlir_str is not None

        # Save module to file for subsequent stages.
        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(mlir_str)

        # Save inputs and golden for execution and validation.
        torch.save(torch_inputs, self.output_file("inputs.pt"))
        torch.save(golden, self.output_file("golden.pt"))

        # Verify MLIR contains expected threads.
        assert f"@compute_{self.OP_STR}" in mlir_str
        assert "@reader" in mlir_str
        assert "@writer" in mlir_str
        assert "ttl.kernel_thread = #ttkernel.thread<compute>" in mlir_str
        assert "ttl.kernel_thread = #ttkernel.thread<noc>" in mlir_str


class UnaryOpTestBase(OpTestBase):
    """Base for unary operations (1 input, 1 output)."""

    ARITY = 1


class BinaryOpTestBase(OpTestBase):
    """Base for binary operations (2 inputs, 1 output)."""

    ARITY = 2


def generate_op_test_classes() -> Dict[str, Type[OpTestBase]]:
    """
    Auto-generate test classes from TTLElementwiseOps.def.

    For each operation, creates test classes for each dtype (bfloat16, float32).

    Returns:
        Dict mapping class name (e.g., "TestAddBfloat16", "TestAddFloat32") to the generated class.
    """
    generated: Dict[str, Type[OpTestBase]] = {}

    # Test dtypes.
    test_dtypes = [
        (torch.bfloat16, "Bfloat16"),
        (torch.float32, "Float32"),
    ]

    for op_name, arity in ELEMENTWISE_OPS.items():
        # Determine base class from arity.
        base: Type[OpTestBase] = UnaryOpTestBase if arity == 1 else BinaryOpTestBase

        # Generate a test class for each dtype.
        for dtype, dtype_suffix in test_dtypes:
            # Build class attributes.
            attrs: Dict[str, Any] = {
                "OP_STR": op_name,
                "INPUT_DTYPE": dtype,
            }
            if op_name in OP_INPUT_RANGES:
                attrs["INPUT_RANGE"] = OP_INPUT_RANGES[op_name]
            if op_name in OP_ULP_THRESHOLD_OVERRIDES:
                overrides = OP_ULP_THRESHOLD_OVERRIDES[op_name]
                if dtype in overrides:
                    attrs["ULP_THRESHOLD"] = overrides[dtype]
            if op_name in OP_PCC_THRESHOLD_OVERRIDES:
                overrides = OP_PCC_THRESHOLD_OVERRIDES[op_name]
                if dtype in overrides:
                    attrs["PCC_THRESHOLD"] = overrides[dtype]

            # Create class dynamically with dtype suffix.
            class_name = f"Test{op_name.capitalize()}{dtype_suffix}"
            test_class = type(class_name, (base,), attrs)

            generated[class_name] = cast(Type[OpTestBase], test_class)

    return generated


# Auto-generated test classes from .def file.
GENERATED_OP_TESTS: Dict[str, Type[OpTestBase]] = generate_op_test_classes()
