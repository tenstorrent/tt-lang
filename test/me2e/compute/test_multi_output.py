# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for multi-output compute operations.

Tests a single ttl.compute that produces two outputs (exp and neg of input).
Validates TTLInsertTileRegsSync auto-inserts stores for each output.

TODO: Device execution skipped - enable once multi-output is supported by runner.
"""

import pytest
import torch
from ttmlir.ir import Context, Module

from ..base import ME2ETestBase
from ..config import E2EConfig
from ..builder.dtype_utils import torch_dtype_to_mlir_str
from ..builder.thread_builder import generate_layout_attrs
from ..builder.dm_builder import DMThreadBuilder

import ttl.dialects.ttl as ttl


class TestMultiOutput(ME2ETestBase):
    """Test multi-output compute: exp(a) -> CB1, neg(a) -> CB2."""

    INPUT_SHAPE = (1, 1)

    @pytest.fixture(scope="class")
    def config(self) -> E2EConfig:
        return E2EConfig(grid_shape=self.INPUT_SHAPE, dtype=torch.bfloat16)

    def get_mlir_template(self, config: E2EConfig) -> str:
        rows, cols = config.grid_shape
        dtype = torch_dtype_to_mlir_str(config.dtype)
        bf = config.buffer_factor

        # Type aliases to reduce template verbosity.
        tile_t = f"!ttcore.tile<32x32, {dtype}>"
        tensor_t = f"tensor<{rows}x{cols}x{tile_t}>"
        cb_t = f"!ttl.cb<[{rows}, {cols}], {tile_t}, {bf}>"

        layout_attrs = generate_layout_attrs(config)
        dm_builder = DMThreadBuilder(config)
        reader_mlir = dm_builder.build_reader(num_inputs=1, total_cbs=3)
        writer_mlir = dm_builder.build_writer(output_cbs=[1, 2], total_cbs=3)

        compute_mlir = f"""
func.func @compute_multi_output(%a: {tensor_t}) -> ({tensor_t}, {tensor_t})
    attributes {{ttl.kernel_thread = #ttkernel.thread<compute>}} {{

  %init0 = tensor.empty() : {tensor_t}
  %init1 = tensor.empty() : {tensor_t}

  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {bf}}} : {cb_t}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {bf}}} : {cb_t}
  %cb2 = ttl.bind_cb {{cb_index = 2, buffer_factor = {bf}}} : {cb_t}

  %a_ready = ttl.cb_wait %cb0 : <[{rows}, {cols}], {tile_t}, {bf}> -> {tensor_t}
  %a_att = ttl.attach_cb %a_ready, %cb0 : ({tensor_t}, {cb_t}) -> {tensor_t}
  %out0_att = ttl.attach_cb %init0, %cb1 : ({tensor_t}, {cb_t}) -> {tensor_t}
  %out1_att = ttl.attach_cb %init1, %cb2 : ({tensor_t}, {cb_t}) -> {tensor_t}

  // Multi-output compute: exp(a) -> CB1, neg(a) -> CB2
  %r0, %r1 = ttl.compute
      ins(%a_att : {tensor_t})
      outs(%out0_att, %out1_att : {tensor_t}, {tensor_t})
      {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}} {{
  ^bb0(%in: {tile_t}, %out0: {tile_t}, %out1: {tile_t}):
    %exp_tile = ttl.tile_exp %in : {tile_t}
    %neg_tile = ttl.tile_neg %in : {tile_t}
    ttl.yield %exp_tile, %neg_tile : {tile_t}, {tile_t}
  }} -> ({tensor_t}, {tensor_t})

  func.return %r0, %r1 : {tensor_t}, {tensor_t}
}}
"""
        return f"{layout_attrs}\n\nmodule {{\n{reader_mlir}\n{compute_mlir}\n{writer_mlir}\n}}\n"

    @pytest.mark.order(1)
    def test_build_module(self, config: E2EConfig) -> None:
        mlir_str = self.get_mlir_template(config)

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)
        with ctx:
            module = Module.parse(mlir_str, ctx)
            module.operation.verify()

        with open(self.output_file("module.mlir"), "w") as f:
            f.write(str(module))

        # Save inputs/golden for potential future device execution.
        inputs = [torch.rand(config.tensor_shape, dtype=config.dtype) * 4 - 2]
        torch.save(inputs, self.output_file("inputs.pt"))
        torch.save(
            [torch.exp(inputs[0]), torch.neg(inputs[0])], self.output_file("golden.pt")
        )

    @pytest.mark.order(2)
    def test_compile_to_ttkernel(self) -> None:
        super().test_compile_to_ttkernel()

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        super().test_translate_to_cpp()

        # Verify 2 pack_tile calls (one per output).
        kernel_dir = self.OUTPUT_DIR / "kernels"
        cpp_file = list(kernel_dir.glob("compute*.cpp"))[0]
        assert cpp_file.read_text().count("pack_tile") == 2

    @pytest.mark.order(4)
    @pytest.mark.skip(reason="Runner doesn't support multi-output yet")
    def test_execute(self, device) -> None:
        pass

    @pytest.mark.order(5)
    @pytest.mark.skip(reason="No device execution")
    def test_validate_golden(self) -> None:
        pass
