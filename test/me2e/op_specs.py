# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation specifications for declarative E2E tests.

Defines ComputeOpSpec dataclass and COMPUTE_OPS registry for all elementwise operations.
COMPUTE_OPS is auto-generated from TTLElementwiseOps.def to keep tests in sync with the dialect.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

from .ops import ELEMENTWISE_OPS, OP_INPUT_RANGES, OP_TORCH_MAP


@dataclass(frozen=True)
class ComputeOpSpec:
    """
    Specification for a compute operation test.

    This dataclass encapsulates all information needed to test a single elementwise
    operation in the declarative E2E test framework. It enables operations to be
    specified as data rather than code, allowing automatic test generation through
    pytest parametrization.

    The framework uses ComputeOpSpec instances to:
    1. Generate MLIR modules with appropriate compute kernels
    2. Create test inputs within specified value ranges
    3. Execute operations on hardware via ttnn.generic_op
    4. Validate results using ULP comparison against golden reference

    All specs in COMPUTE_OPS are automatically parametrized in test_compute_ops.py,
    creating a test case for each (op, config) combination.

    Attributes:
        name: Operation name used for identification and MLIR function naming.
            Examples: "add", "exp", "sqrt". This appears in generated function
            names like `@compute_add` and test case identifiers.

        ttl_op: TTL tile operation name used when generating MLIR code.
            Examples: "tile_add", "tile_exp", "tile_sqrt". This must match
            the actual TTL dialect operation name (without the "ttl." prefix).
            Used in MLIR generation: `ttl.tile_add %a, %b`.

        arity: Number of input operands for the operation.
            - 1: Unary operation (e.g., exp, sqrt, neg)
            - 2: Binary operation (e.g., add, sub, mul)
            Determines function signatures, reader kernel selection, and
            golden reference function invocation.

        golden: PyTorch reference function for computing expected results.
            Must be callable with the same arity as the operation. Used for
            validation by comparing hardware results against PyTorch output.
            Examples:
            - `torch.add` or `lambda a, b: a + b` for binary ops
            - `torch.exp` or `torch.sqrt` for unary ops
            - Custom lambdas for operations without direct PyTorch equivalents

        reader_type: Data movement kernel type for reading inputs from DRAM.
            Must be one of: "unary", "binary", "ternary". Typically matches
            the arity (1->"unary", 2->"binary"), but can differ for special
            cases like fused operations that read multiple inputs.

        input_range: Optional input value range constraint as (min, max) tuple.
            If provided, test inputs are generated within this range using
            `torch.rand(...) * (max - min) + min`. Required for operations
            with domain constraints:
            - `log`, `sqrt`, `rsqrt`: Require positive inputs -> (0.01, 10.0)
            - `acos`, `asin`: Require inputs in [-1, 1] -> (-1.0, 1.0)
            If None, defaults to (-1.0, 1.0) for all operations.
    """

    name: str
    ttl_op: str
    arity: int
    golden: Callable[..., Any]
    reader_type: str
    input_range: Optional[Tuple[float, float]] = None


# Special cases for ops that need custom golden functions (not in OP_TORCH_MAP or need different implementation).
SPECIAL_GOLDEN_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "rsqrt": lambda x: 1.0 / torch.sqrt(x),  # type: ignore[misc]  # Use lambda instead of torch.rsqrt for consistency
}


def _generate_compute_ops() -> list[ComputeOpSpec]:
    """
    Auto-generate COMPUTE_OPS from TTLElementwiseOps.def.

    Operations are parsed from the .def file, and golden functions/input ranges
    are looked up from OP_TORCH_MAP and OP_INPUT_RANGES. Special cases can be
    handled via SPECIAL_GOLDEN_FUNCTIONS.

    Returns:
        List of ComputeOpSpec instances for all elementwise operations.
    """
    compute_ops: list[ComputeOpSpec] = []

    for op_name, arity in ELEMENTWISE_OPS.items():
        # Get golden function from special cases, OP_TORCH_MAP, or skip if not found.
        if op_name in SPECIAL_GOLDEN_FUNCTIONS:
            golden = SPECIAL_GOLDEN_FUNCTIONS[op_name]
        elif op_name in OP_TORCH_MAP:
            golden = OP_TORCH_MAP[op_name]
        else:
            # Skip ops without torch reference (shouldn't happen for elementwise ops).
            continue

        # Derive ttl_op name from op name.
        ttl_op = f"tile_{op_name}"

        # Derive reader_type from arity.
        reader_type = "unary" if arity == 1 else "binary"

        # Get input range if specified.
        input_range = OP_INPUT_RANGES.get(op_name)

        compute_ops.append(
            ComputeOpSpec(
                name=op_name,
                ttl_op=ttl_op,
                arity=arity,
                golden=golden,
                reader_type=reader_type,
                input_range=input_range,
            )
        )

    return compute_ops


# Auto-generated from TTLElementwiseOps.def.
# Adding a new op to the .def file automatically creates a test here.
COMPUTE_OPS = _generate_compute_ops()

# Validate that ops were generated.
if not COMPUTE_OPS:
    import warnings

    warnings.warn(
        "COMPUTE_OPS is empty. Check that TTLElementwiseOps.def exists and contains operations, "
        "and that OP_TORCH_MAP in ops/__init__.py has entries for all operations.",
        UserWarning,
    )
