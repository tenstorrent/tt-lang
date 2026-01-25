# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for sequential compute operations in the same kernel.

This tests the DST register synchronization when multiple ttl.compute ops
are lowered to separate scf.for loop nests, each with their own sync ops.

The ttl-insert-inter-loop-cb-sync pass automatically inserts cb_wait between
consecutive compute loops when the output CB of one matches the input CB of
the next.

Three test patterns:
1. TestTwoComputesChained: compute1(a + b) -> CB2, compute2(r0 * r0) -> CB3
   Uses intermediate CB (CB2) with inter-loop sync (cb_wait before second loop).
   Result: (a + b)²

2. TestThreeComputePipeline: Three-stage pipeline with two intermediate CBs.
   compute1(a + b) -> CB2, compute2(CB2 * CB2) -> CB3, compute3(CB3 + a) -> CB4
   Tests inter-loop sync between each consecutive pair of compute ops.
   Result: (a + b)² + a

3. TestMultiInputCompute: compute1(a + b) -> CB2, compute2(a * r0) -> CB3
   Tests multiple input CBs where second compute reads from both CB0 and CB2.
   Result: a * (a + b)
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor
from ttmlir.ir import Context, Module

from ..base import ME2ETestBase
from ..config import E2EConfig
from ..builder.dtype_utils import torch_dtype_to_mlir_str
from ..builder.thread_builder import generate_layout_attrs
from ..builder.dm_builder import DMThreadBuilder

import ttl.dialects.ttl as ttl


class TwoComputeTestBase(ME2ETestBase):
    """
    Base class for two-compute operation tests.

    Tests DST register synchronization when multiple ttl.compute ops
    are lowered to separate scf.for loop nests.
    """

    # Override in subclasses.
    OP_NAME: str
    ARITY: int = 2
    INPUT_SHAPE: Tuple[int, int] = (1, 1)
    INPUT_DTYPE: torch.dtype = torch.bfloat16
    INPUT_RANGE: Tuple[float, float] = (-1.0, 1.0)

    @pytest.fixture(scope="class")
    def config(self) -> E2EConfig:
        """Get test configuration."""
        return E2EConfig(grid_shape=self.INPUT_SHAPE, dtype=self.INPUT_DTYPE)

    def torch_reference(self, *inputs: Tensor) -> Tensor:
        """
        Compute golden output using torch.

        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement torch_reference()")

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Return MLIR string template for the two-compute operation.

        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_mlir_template()")

    @pytest.mark.order(1)
    def test_build_module(self, config: E2EConfig) -> None:
        """Build TTL module for two-compute test from MLIR template."""
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

    @pytest.mark.order(2)
    def test_compile_to_ttkernel(self) -> None:
        """Run TTL-to-TTKernel pass pipeline on the generated module."""
        super().test_compile_to_ttkernel()

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        """Translate two-compute TTKernel to C++ kernels."""
        super().test_translate_to_cpp()


class TestTwoComputesChained(TwoComputeTestBase):
    """
    Test two sequential compute ops with intermediate CB and inter-loop sync.

    Pattern: compute1(a + b) -> CB2 (intermediate), compute2(r0 * r0) -> CB3 (output)

    The result is (a + b)^2.

    This test validates:
    1. Each compute op gets its own DST sync ops (acquire/commit/wait/release)
    2. The intermediate CB (CB2) passes data between computes
    3. The inter-loop CB sync pass inserts cb_wait before second compute
       (because first compute's output_cb == second compute's input_cb)
    4. The second compute correctly reads from the intermediate CB
    """

    OP_NAME = "two_computes_chained"
    ARITY = 2
    # Number of intermediate CBs (not backed by I/O tensors).
    NUM_INTERMEDIATE_CBS = 1

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute (a + b)^2."""
        return (a + b) ** 2

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Build MLIR with two sequential ttl.compute regions.

        Pattern:
          CB0, CB1 -> compute1(add) -> CB2 (intermediate)
          CB2 -> compute2(mul) -> CB3 (output)

        Uses 4 CBs: 2 inputs, 1 intermediate (CB2), 1 final output (CB3).

        Key: First compute writes to CB2, second compute reads from CB2.
        This triggers inter-loop CB sync (cb_wait before second loop).
        """
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)

        # Reader for 2 inputs, writer for CB3 (final output).
        # Note: output_tensor_indices=[2] because the output tensor is at index 2
        # in the io_tensors list (after input_a, input_b), even though it uses CB3.
        reader_mlir = dm_builder.build_reader(num_inputs=2, total_cbs=4)
        writer_mlir = dm_builder.build_writer(
            output_cbs=[3], total_cbs=4, output_tensor_indices=[2]
        )

        compute_mlir = f"""
  func.func @compute_two_ops(%a: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                             %b: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
      -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      attributes {{ttl.kernel_thread = #ttkernel.thread<compute>}} {{
    %init = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb2 = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>

    %a_ready = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %b_ready = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %init_cb = ttl.attach_cb %init, %cb2 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // First compute: a + b -> r0
    %r0 = ttl.compute
        ins(%a_ready, %b_ready : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                                 tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%a_tile: !ttcore.tile<32x32, {dtype}>,
         %b_tile: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, {dtype}>
      %view0 = ttl.cb_reserve %cb2 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %sum, %view0 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb2 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %sum : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Attach r0 to SAME CB2 for second compute input.
    // This triggers inter-loop CB sync (output_cb == input_cb).
    %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // New init for second compute output -> CB3.
    %init2 = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %cb3 = ttl.bind_cb {{cb_index = 3, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %init2_cb = ttl.attach_cb %init2, %cb3 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Second compute: r0 * r0 -> CB3 (reads from CB2, writes to CB3)
    %result = ttl.compute
        ins(%r0_cb, %r0_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                             tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init2_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%r0_tile1: !ttcore.tile<32x32, {dtype}>,
         %r0_tile2: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %product = ttl.tile_mul %r0_tile1, %r0_tile2 : !ttcore.tile<32x32, {dtype}>
      %view1 = ttl.cb_reserve %cb3 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %product, %view1 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb3 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %product : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

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

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        """Translate to C++ and verify two sets of DST sync ops."""
        super().test_translate_to_cpp()

        # Additional verification: check for two sets of DST sync ops.
        # Find the compute kernel file in the kernels directory.
        kernel_dir = self.OUTPUT_DIR / "kernels"
        compute_files = list(kernel_dir.glob("compute*.cpp"))
        assert compute_files, "No compute kernel file found"
        cpp_file = compute_files[0]
        with open(cpp_file) as f:
            source = f.read()

        assert (
            source.count("tile_regs_acquire()") == 2
        ), "Should have 2 tile_regs_acquire (one per compute)"
        assert (
            source.count("tile_regs_release()") == 2
        ), "Should have 2 tile_regs_release (one per compute)"
        assert (
            source.count("add_binary_tile_init()") == 1
        ), "Should have 1 add operation (first compute)"
        assert (
            source.count("mul_binary_tile_init()") == 1
        ), "Should have 1 mul operation (second compute)"

    @pytest.mark.order(4)
    @pytest.mark.requires_device
    def test_execute(self, device) -> None:
        """Execute on device with intermediate CB support."""
        super().test_execute(device)

    @pytest.mark.order(5)
    def test_validate_golden(self) -> None:
        """Validate result against golden ((a + b)^2)."""
        super().test_validate_golden()


class TestThreeComputePipeline(TwoComputeTestBase):
    """
    Test three sequential compute ops in a pipeline pattern.

    Pattern:
      compute1: a + b -> CB2 (intermediate1)
      compute2: CB2 * CB2 -> CB3 (intermediate2)
      compute3: CB3 + a -> CB4 (output)

    The result is (a + b)² + a.

    This test validates:
    1. Three compute ops each get their own DST sync ops
    2. Inter-loop CB sync is inserted between each consecutive pair:
       - cb_wait for CB2 before compute2 (first->second)
       - cb_wait for CB3 before compute3 (second->third)
    3. Proper CB chaining through the pipeline (CB2 -> CB3 -> CB4)
    4. Multiple intermediate CBs work correctly
    """

    OP_NAME = "three_compute_pipeline"
    ARITY = 2
    NUM_INTERMEDIATE_CBS = 2
    # Higher ULP threshold due to three chained bfloat16 operations accumulating error.
    ULP_THRESHOLD = 150.0

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute (a + b)² + a."""
        return (a + b) ** 2 + a

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Build MLIR with three sequential ttl.compute regions in a pipeline.

        Pattern:
          CB0, CB1 -> compute1(add) -> CB2 (intermediate1)
          CB2 -> compute2(mul) -> CB3 (intermediate2)
          CB3, CB0 -> compute3(add) -> CB4 (output)

        Uses 5 CBs: 2 inputs (CB0, CB1), 2 intermediate (CB2, CB3), 1 output (CB4).
        """
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)

        # Reader for 2 inputs, writer for CB4 (final output).
        reader_mlir = dm_builder.build_reader(num_inputs=2, total_cbs=5)
        writer_mlir = dm_builder.build_writer(
            output_cbs=[4], total_cbs=5, output_tensor_indices=[2]
        )

        compute_mlir = f"""
  func.func @compute_pipeline(%a: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                               %b: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
      -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      attributes {{ttl.kernel_thread = #ttkernel.thread<compute>}} {{
    %init = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb2 = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb3 = ttl.bind_cb {{cb_index = 3, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb4 = ttl.bind_cb {{cb_index = 4, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>

    %a_ready = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %b_ready = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %init_cb2 = ttl.attach_cb %init, %cb2 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Compute 1: a + b -> CB2 (intermediate1)
    %r0 = ttl.compute
        ins(%a_ready, %b_ready : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                                 tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init_cb2 : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%a_tile: !ttcore.tile<32x32, {dtype}>,
         %b_tile: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, {dtype}>
      %view0 = ttl.cb_reserve %cb2 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %sum, %view0 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb2 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %sum : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Attach r0 to CB2 for compute2 input.
    %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Init for compute2 output -> CB3.
    %init2 = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %init_cb3 = ttl.attach_cb %init2, %cb3 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Compute 2: r0 * r0 -> CB3 (intermediate2) = (a+b)²
    // Inter-loop sync: cb_wait for CB2 before this loop
    %r1 = ttl.compute
        ins(%r0_cb, %r0_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                             tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init_cb3 : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%r0_tile1: !ttcore.tile<32x32, {dtype}>,
         %r0_tile2: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %product = ttl.tile_mul %r0_tile1, %r0_tile2 : !ttcore.tile<32x32, {dtype}>
      %view1 = ttl.cb_reserve %cb3 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %product, %view1 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb3 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %product : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Attach r1 to CB3 for compute3 input.
    %r1_cb = ttl.attach_cb %r1, %cb3 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Init for compute3 output -> CB4.
    %init3 = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %init_cb4 = ttl.attach_cb %init3, %cb4 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Compute 3: a + r1 -> CB4 (output) = a + (a+b)² = (a+b)² + a
    // Inter-loop sync: cb_wait for CB3 before this loop
    // NOTE: a_ready comes first so that init_sfpu uses CB0 (the persistent input CB).
    %result = ttl.compute
        ins(%a_ready, %r1_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                               tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init_cb4 : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%a_tile2: !ttcore.tile<32x32, {dtype}>,
         %r1_tile: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %final_sum = ttl.tile_add %a_tile2, %r1_tile : !ttcore.tile<32x32, {dtype}>
      %view2 = ttl.cb_reserve %cb4 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %final_sum, %view2 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb4 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %final_sum : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

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

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        """Translate to C++ and verify three sets of DST sync ops."""
        super().test_translate_to_cpp()

        # Verify the generated C++ has three compute operations.
        kernel_dir = self.OUTPUT_DIR / "kernels"
        compute_files = list(kernel_dir.glob("compute*.cpp"))
        assert compute_files, "No compute kernel file found"
        cpp_file = compute_files[0]
        with open(cpp_file) as f:
            source = f.read()

        assert (
            source.count("tile_regs_acquire()") == 3
        ), "Should have 3 tile_regs_acquire (one per compute)"
        assert (
            source.count("tile_regs_release()") == 3
        ), "Should have 3 tile_regs_release (one per compute)"
        # 2 adds: compute1 (a+b), compute3 ((a+b)²+a)
        assert (
            source.count("add_binary_tile_init()") == 2
        ), "Should have 2 add operations"
        # 1 mul: compute2 (r0*r0)
        assert (
            source.count("mul_binary_tile_init()") == 1
        ), "Should have 1 mul operation"
        # Inter-loop syncs: cb_wait_front for CB2 (before compute2) and CB3 (before compute3)
        assert (
            source.count("cb_wait_front") >= 2
        ), "Should have at least 2 cb_wait_front for inter-loop sync"

    @pytest.mark.order(4)
    @pytest.mark.requires_device
    def test_execute(self, device) -> None:
        """Execute on device with three-stage pipeline."""
        super().test_execute(device)

    @pytest.mark.order(5)
    def test_validate_golden(self) -> None:
        """Validate result against golden ((a + b)² + a)."""
        super().test_validate_golden()


class TestMultiInputCompute(TwoComputeTestBase):
    """
    Test compute with multiple inputs from different CB sources.

    Pattern:
      compute1(a + b) -> CB2 (intermediate)
      compute2(a * intermediate) -> CB3 (output)

    The result is a * (a + b).

    This test validates:
    1. Multiple input CBs are tracked correctly (second compute has input_cbs=[0, 2])
    2. Inter-loop CB sync inserts cb_wait for CB2 (from first compute's output)
    3. No cb_wait for CB0 (from reader, not from another compute)
    4. The output_cbs array correctly tracks CB3
    """

    OP_NAME = "multi_input_compute"
    ARITY = 2
    NUM_INTERMEDIATE_CBS = 1

    def torch_reference(self, a: Tensor, b: Tensor) -> Tensor:
        """Compute a * (a + b)."""
        return a * (a + b)

    def get_mlir_template(self, config: E2EConfig) -> str:
        """
        Build MLIR with two computes where second has mixed input sources.

        Pattern:
          CB0, CB1 -> compute1(add) -> CB2 (intermediate)
          CB0, CB2 -> compute2(mul) -> CB3 (output)

        This exercises:
        - Multiple input CB tracking (second compute reads from CB0 AND CB2)
        - Selective inter-loop sync (cb_wait for CB2 only, not CB0)
        """
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)

        reader_mlir = dm_builder.build_reader(num_inputs=2, total_cbs=4)
        writer_mlir = dm_builder.build_writer(
            output_cbs=[3], total_cbs=4, output_tensor_indices=[2]
        )

        compute_mlir = f"""
  func.func @compute_multi_input(%a: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                                  %b: tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
      -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      attributes {{ttl.kernel_thread = #ttkernel.thread<compute>}} {{
    %init = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %cb2 = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>

    %a_ready = ttl.cb_wait %cb0 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %b_ready = ttl.cb_wait %cb1 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %init_cb = ttl.attach_cb %init, %cb2 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // First compute: a + b -> CB2 (intermediate)
    %r0 = ttl.compute
        ins(%a_ready, %b_ready : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                                 tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%a_tile: !ttcore.tile<32x32, {dtype}>,
         %b_tile: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, {dtype}>
      %view0 = ttl.cb_reserve %cb2 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %sum, %view0 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb2 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %sum : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Attach r0 to CB2 for second compute input.
    %r0_cb = ttl.attach_cb %r0, %cb2 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // New init for second compute output -> CB3.
    %init2 = tensor.empty() : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
    %cb3 = ttl.bind_cb {{cb_index = 3, buffer_factor = {bf}}} : !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
    %init2_cb = ttl.attach_cb %init2, %cb3 : (tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>, !ttl.cb<[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>) -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

    // Second compute: a * r0 -> CB3
    // input_cbs = [0, 2] (from CB0 and CB2)
    // Only CB2 needs cb_wait (from first compute's output_cb)
    %result = ttl.compute
        ins(%a_ready, %r0_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>,
                               tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        outs(%init2_cb : tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>)
        {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}} {{
    ^bb0(%a_tile2: !ttcore.tile<32x32, {dtype}>,
         %r0_tile: !ttcore.tile<32x32, {dtype}>,
         %out_tile: !ttcore.tile<32x32, {dtype}>):
      %product = ttl.tile_mul %a_tile2, %r0_tile : !ttcore.tile<32x32, {dtype}>
      %view1 = ttl.cb_reserve %cb3 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}> -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.store %product, %view1 : !ttcore.tile<32x32, {dtype}>, tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>
      ttl.cb_push %cb3 : <[{rows}, {cols}], !ttcore.tile<32x32, {dtype}>, {bf}>
      ttl.yield %product : !ttcore.tile<32x32, {dtype}>
    }} -> tensor<{rows}x{cols}x!ttcore.tile<32x32, {dtype}>>

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

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        """Translate to C++ and verify inter-loop sync."""
        super().test_translate_to_cpp()

        # Verify the generated C++ has correct structure.
        kernel_dir = self.OUTPUT_DIR / "kernels"
        compute_files = list(kernel_dir.glob("compute*.cpp"))
        assert compute_files, "No compute kernel file found"
        cpp_file = compute_files[0]
        with open(cpp_file) as f:
            source = f.read()

        # Should have cb_wait_front for CB2 (inter-loop sync)
        assert (
            "cb_wait_front" in source
        ), "Should have cb_wait_front for inter-loop sync"

        # Should have two sets of DST sync ops
        assert (
            source.count("tile_regs_acquire()") == 2
        ), "Should have 2 tile_regs_acquire (one per compute)"

    @pytest.mark.order(4)
    @pytest.mark.requires_device
    def test_execute(self, device) -> None:
        """Execute on device."""
        super().test_execute(device)

    @pytest.mark.order(5)
    def test_validate_golden(self) -> None:
        """Validate result against golden (a * (a + b))."""
        super().test_validate_golden()
