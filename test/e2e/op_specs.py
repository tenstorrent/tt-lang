# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation specifications for declarative E2E tests.

Defines ComputeOpSpec dataclass and COMPUTE_OPS registry for all elementwise operations.
This enables declarative testing where operations are specified as data rather than code.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


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
            the arity (1→"unary", 2→"binary"), but can differ for special
            cases like fused operations that read multiple inputs.

        input_range: Optional input value range constraint as (min, max) tuple.
            If provided, test inputs are generated within this range using
            `torch.rand(...) * (max - min) + min`. Required for operations
            with domain constraints:
            - `log`, `sqrt`, `rsqrt`: Require positive inputs → (0.01, 10.0)
            - `acos`, `asin`: Require inputs in [-1, 1] → (-1.0, 1.0)
            If None, defaults to (-1.0, 1.0) for all operations.
    """

    name: str
    ttl_op: str
    arity: int
    golden: Callable
    reader_type: str
    input_range: Optional[Tuple[float, float]] = None


# All ops - add one line to test a new op
COMPUTE_OPS = [
    # Binary ops
    ComputeOpSpec("add", "tile_add", 2, lambda a, b: a + b, "binary"),
    ComputeOpSpec("sub", "tile_sub", 2, lambda a, b: a - b, "binary"),
    ComputeOpSpec("mul", "tile_mul", 2, lambda a, b: a * b, "binary"),
    ComputeOpSpec("max", "tile_max", 2, torch.maximum, "binary"),
    # Unary ops
    ComputeOpSpec("exp", "tile_exp", 1, torch.exp, "unary"),
    ComputeOpSpec("log", "tile_log", 1, torch.log, "unary", input_range=(0.01, 10.0)),
    ComputeOpSpec(
        "sqrt", "tile_sqrt", 1, torch.sqrt, "unary", input_range=(0.01, 10.0)
    ),
    ComputeOpSpec(
        "rsqrt",
        "tile_rsqrt",
        1,
        lambda x: 1.0 / torch.sqrt(x),
        "unary",
        input_range=(0.01, 10.0),
    ),
    ComputeOpSpec("tanh", "tile_tanh", 1, torch.tanh, "unary"),
    ComputeOpSpec("abs", "tile_abs", 1, torch.abs, "unary"),
    ComputeOpSpec("neg", "tile_neg", 1, torch.neg, "unary"),
    ComputeOpSpec("relu", "tile_relu", 1, torch.relu, "unary"),
    ComputeOpSpec("sigmoid", "tile_sigmoid", 1, torch.sigmoid, "unary"),
]
