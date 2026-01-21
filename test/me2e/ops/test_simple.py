# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple op tests that just validate MLIR module generation.

These tests don't require hardware - they just check that TTL modules
can be built correctly for each operation.
"""

import pytest
import torch

from ..config import DTYPE_TO_MLIR, E2EConfig, get_dtype_ids, get_test_dtypes
from ..builder.ttl_builder import build_ttl_module


class TestMLIRGeneration:
    """Test MLIR module generation for ops."""

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
    @pytest.mark.parametrize("dtype", get_test_dtypes(), ids=get_dtype_ids())
    def test_binary_ops(self, op_name, dtype):
        """Test binary op MLIR generation."""
        config = E2EConfig(grid_shape=(1, 1), dtype=dtype)
        inputs = [torch.rand(config.tensor_shape, dtype=config.dtype) for _ in range(2)]

        module = build_ttl_module(op_name, 2, config, inputs)
        assert module is not None

        mlir_str = str(module)
        # Verify function name and op presence.
        assert f"@compute_{op_name}" in mlir_str
        assert f"ttl.{op_name}" in mlir_str
        # Verify CB binding and attachment.
        assert "ttl.bind_cb" in mlir_str
        assert "ttl.attach_cb" in mlir_str
        # Verify two inputs.
        assert "%arg0" in mlir_str
        assert "%arg1" in mlir_str
        # Verify dtype appears in tile type.
        assert f"tile<32x32, {DTYPE_TO_MLIR[dtype]}>" in mlir_str

    @pytest.mark.parametrize("op_name", ["exp", "sqrt", "neg"])
    @pytest.mark.parametrize("dtype", get_test_dtypes(), ids=get_dtype_ids())
    def test_unary_ops(self, op_name, dtype):
        """Test unary op MLIR generation."""
        config = E2EConfig(grid_shape=(1, 1), dtype=dtype)
        inputs = [torch.rand(config.tensor_shape, dtype=config.dtype)]

        module = build_ttl_module(op_name, 1, config, inputs)
        assert module is not None

        mlir_str = str(module)
        # Verify function name and op presence.
        assert f"@compute_{op_name}" in mlir_str
        assert f"ttl.{op_name}" in mlir_str
        # Verify CB binding and attachment.
        assert "ttl.bind_cb" in mlir_str
        assert "ttl.attach_cb" in mlir_str
        # Verify one input.
        assert "%arg0" in mlir_str
        # Verify dtype appears in tile type.
        assert f"tile<32x32, {DTYPE_TO_MLIR[dtype]}>" in mlir_str

    def test_different_grid_shapes(self):
        """Test different grid shapes."""
        for grid_shape in [(1, 1), (2, 2), (4, 4)]:
            config = E2EConfig(grid_shape=grid_shape, dtype=torch.bfloat16)
            inputs = [
                torch.rand(config.tensor_shape, dtype=config.dtype) for _ in range(2)
            ]

            module = build_ttl_module("add", 2, config, inputs)
            assert module is not None

            mlir_str = str(module)
            # Verify grid shape appears in CB type.
            assert f"[{grid_shape[0]}, {grid_shape[1]}]" in mlir_str

    def test_different_dtypes(self):
        """Test different data types."""
        for dtype, mlir_dtype in DTYPE_TO_MLIR.items():
            config = E2EConfig(grid_shape=(1, 1), dtype=dtype)
            inputs = [torch.rand(config.tensor_shape, dtype=dtype) for _ in range(2)]

            module = build_ttl_module("add", 2, config, inputs)
            assert module is not None

            mlir_str = str(module)
            # Verify dtype appears in tile type.
            assert f"tile<32x32, {mlir_dtype}>" in mlir_str
