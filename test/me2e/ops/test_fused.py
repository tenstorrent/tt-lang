# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Custom fused operation tests.

Demonstrates how to test operations that are NOT auto-generated from
TTLElementwiseOps.def. Use this pattern for:
- Fused ops (e.g., exp(a + b), relu(a * b + c))
- Custom sequences of tile ops
- Ops with non-standard semantics

These tests use MLIR string templates to build ttl.compute regions with
multiple tile operations fused together.
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor
from ttmlir.ir import Context, Module

from ..base import E2ETestBase
from ..config import E2EConfig
from ..builder.dtype_utils import torch_dtype_to_mlir_str
from ..builder.dm_threads import (
    generate_binary_reader_mlir,
    generate_unary_reader_mlir,
    generate_writer_mlir,
    generate_layout_attrs,
)

import ttl.dialects.ttl as ttl


class FusedOpTestBase(E2ETestBase):
    """
    Base class for custom fused operation tests.

    Unlike auto-generated tests, fused ops require:
    1. Custom MLIR template (get_mlir_template)
    2. Custom torch reference function (torch_reference)
    """

    # Override these in subclasses.
    OP_NAME: str  # Descriptive name for the fused op
    ARITY: int  # Number of inputs
    INPUT_SHAPE: Tuple[int, int] = (2, 2)
    INPUT_DTYPE: torch.dtype = torch.bfloat16
    INPUT_RANGE: Tuple[float, float] = (-1.0, 1.0)

    @pytest.fixture(scope="class")
    def config(self) -> E2EConfig:
        """Get test configuration."""
        return E2EConfig(grid_shape=self.INPUT_SHAPE, dtype=self.INPUT_DTYPE)

    def torch_reference(self, *inputs: Tensor) -> Tensor:
        """
        Compute golden output using torch.

        Override this in subclasses for custom fused operations.
        """
        raise NotImplementedError("Subclasses must implement torch_reference()")

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Return MLIR string template for the fused operation.

        Override this to define the ttl.compute region with fused tile ops.
        """
        raise NotImplementedError("Subclasses must implement get_mlir_template()")

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        """Translate fused op TTKernel to C++ kernels."""
        super().test_translate_to_cpp()

    @pytest.mark.order(4)
    @pytest.mark.requires_device
    def test_execute(self, device) -> None:
        """Execute fused op kernels on device."""
        super().test_execute(device)

    @pytest.mark.order(5)
    def test_validate_golden(self) -> None:
        """Validate fused op result against golden."""
        super().test_validate_golden()

    @pytest.mark.order(1)
    def test_build_module(self, config: E2EConfig) -> None:
        """Build TTL module for fused op from MLIR template."""
        # Generate random inputs.
        lo, hi = self.INPUT_RANGE
        torch_inputs = []
        for _ in range(self.ARITY):
            t = torch.rand(config.tensor_shape, dtype=config.dtype) * (hi - lo) + lo
            torch_inputs.append(t)

        # Parse MLIR template.
        mlir_str = self.get_mlir_template(config)

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)
        with ctx:
            module = Module.parse(mlir_str, ctx)
            module.operation.verify()

        # Save module to file for subsequent stages.
        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(str(module))

        # Save inputs for golden comparison.
        torch.save(torch_inputs, self.output_file("inputs.pt"))

        # Compute and save golden output.
        golden = self.torch_reference(*torch_inputs)
        torch.save(golden, self.output_file("golden.pt"))


class TestExpAddFused(FusedOpTestBase):
    """
    Test exp(a + b) fused operation.

    This is a common pattern where add and exp are fused into a single
    ttl.compute region for better performance.
    """

    OP_NAME = "exp_add"
    ARITY = 2
    INPUT_RANGE = (-2.0, 2.0)  # Limit range to avoid exp overflow

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute exp(a + b)."""
        return torch.exp(a + b)

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Build MLIR with ttl.compute region containing tile_add + tile_exp.

        Pattern: reader → CB0, CB1 → compute(add, exp) → CB2 → writer
        Uses helper functions from dm_threads to ensure correct attributes.
        """
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        # Generate layout attributes (includes #dram and #layout)
        layout_attrs = generate_layout_attrs(config.grid_shape, config.dtype)

        # Generate reader and writer using helper functions
        reader_mlir = generate_binary_reader_mlir(config.grid_shape, config.dtype, bf)
        writer_mlir = generate_writer_mlir(
            config.grid_shape, config.dtype, bf, output_cb_index=2
        )

        # Manually write the compute function with custom fused operations
        compute_mlir = f"""
// Compute thread for exp(a + b) fused operation.
func.func @compute_exp_add(%arg0: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                            %arg1: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
    -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %output = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>

  // Wait for data from reader.
  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %b = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %output_cb = ttl.attach_cb %output, %cb_out : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  // Fused: add then exp
  %sum = ttl.add %a, %b : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %result = ttl.exp %sum : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  func.return %result : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""


class TestReluMulFused(FusedOpTestBase):
    """
    Test relu(a * b) fused operation.

    Demonstrates another common fusion pattern.
    """

    OP_NAME = "relu_mul"
    ARITY = 2

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute relu(a * b)."""
        return torch.relu(a * b)

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Build MLIR with ttl.compute region containing tile_mul + tile_relu.

        Pattern: reader → CB0, CB1 → compute(mul, relu) → CB2 → writer
        Uses helper functions from dm_threads to ensure correct attributes.
        """
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        # Generate layout attributes (includes #dram and #layout)
        layout_attrs = generate_layout_attrs(config.grid_shape, config.dtype)

        # Generate reader and writer using helper functions
        reader_mlir = generate_binary_reader_mlir(config.grid_shape, config.dtype, bf)
        writer_mlir = generate_writer_mlir(
            config.grid_shape, config.dtype, bf, output_cb_index=2
        )

        # Manually write the compute function with custom fused operations
        compute_mlir = f"""
// Compute thread for relu(a * b) fused operation.
func.func @compute_relu_mul(%arg0: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                             %arg1: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
    -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %output = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>

  // Wait for data from reader.
  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %b = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %output_cb = ttl.attach_cb %output, %cb_out : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  // Fused: mul then relu
  %prod = ttl.mul %a, %b : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %result = ttl.relu %prod : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  func.return %result : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""


class TestSqrtAbsFused(FusedOpTestBase):
    """
    Test sqrt(abs(a)) fused operation.

    Unary fusion example - sqrt requires positive inputs, so we fuse with abs.
    """

    OP_NAME = "sqrt_abs"
    ARITY = 1
    INPUT_RANGE = (-10.0, 10.0)  # Can be negative since we apply abs first

    def torch_reference(self, a: Tensor) -> Tensor:
        """Compute sqrt(abs(a))."""
        return torch.sqrt(torch.abs(a))

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Build MLIR with ttl.compute region containing tile_abs + tile_sqrt.

        Pattern: reader → CB0 → compute(abs, sqrt) → CB1 → writer
        Uses helper functions from dm_threads to ensure correct attributes.
        """
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        # Generate layout attributes (includes #dram and #layout)
        layout_attrs = generate_layout_attrs(config.grid_shape, config.dtype)

        # Generate reader and writer using helper functions
        reader_mlir = generate_unary_reader_mlir(config.grid_shape, config.dtype, bf)
        writer_mlir = generate_writer_mlir(
            config.grid_shape, config.dtype, bf, output_cb_index=1
        )

        # Manually write the compute function with custom fused operations
        compute_mlir = f"""
// Compute thread for sqrt(abs(a)) fused operation.
func.func @compute_sqrt_abs(%arg0: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
    -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    attributes {{ttl.base_cta_index = 2 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %output = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
  %cb_out = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>

  // Wait for data from reader.
  %a = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %output_cb = ttl.attach_cb %output, %cb_out : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  // Fused: abs then sqrt
  %abs_result = ttl.abs %a : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
  %result = ttl.sqrt %abs_result : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

  func.return %result : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
}}
"""

        return f"""{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""


1
