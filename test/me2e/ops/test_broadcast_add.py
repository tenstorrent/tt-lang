# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Broadcast binary add regression test.

Validates that binary add with broadcast indexing maps (identity + col_broadcast)
correctly falls back to the SFPU copy_tile path instead of the FPU path.
FPU binary ops (add_tiles) use a shared CB tile index for both operands, so
they require identical indexing maps. When maps differ (broadcast), TTLAssignDST
skips FPU marking and the operation uses the SFPU path which handles each
operand independently.

Uses a 4x4 output grid with a 4x1 broadcast input, requiring DST subblocking
(16 tiles with dstPerIteration=2 exceeds DST capacity of 8 for bf16).
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor

from ..base import ME2ETestBase
from ..config import E2EConfig

import ttl.dialects.ttl as ttl


# Output grid: 4x4 tiles. Broadcast input B: 4x1 tiles.
OUTPUT_ROWS, OUTPUT_COLS = 4, 4
BCAST_ROWS, BCAST_COLS = 4, 1
DTYPE = torch.bfloat16
DTYPE_STR = "bf16"
BF = 2  # buffer factor


def _build_broadcast_add_mlir() -> str:
    """
    Build MLIR module for broadcast add: output = A + broadcast(B).

    A is 4x4 tiles, B is 4x1 tiles (column broadcast).
    Reader reads B[row, 0] for every (row, col) iteration.
    Compute uses ttl.add with different-shaped operands; convert-ttl-to-compute
    infers broadcast indexing maps from the shape mismatch.
    """
    tile = f"!ttcore.tile<32x32, {DTYPE_STR}>"
    # Tensor types (without layout, for compute CBs).
    a_tensor = f"tensor<{OUTPUT_ROWS}x{OUTPUT_COLS}x{tile}>"
    b_tensor = f"tensor<{BCAST_ROWS}x{BCAST_COLS}x{tile}>"
    out_tensor = a_tensor

    # DRAM tensor types (with layout attribute inside the type).
    a_dram = f"tensor<{OUTPUT_ROWS}x{OUTPUT_COLS}x{tile}, #layout>"
    b_dram = f"tensor<{BCAST_ROWS}x{BCAST_COLS}x{tile}, #layout_b>"
    out_dram = a_dram
    a_slice = f"tensor<1x1x{tile}, #layout>"
    b_slice = f"tensor<1x1x{tile}, #layout_b>"
    out_slice = a_slice

    # CB types for the reader/writer (1x1 tile processing).
    cb_type = f"!ttl.cb<[1, 1], {tile}, {BF}>"
    cb_spec = f"<[1, 1], {tile}, {BF}>"
    cb_tensor = f"tensor<1x1x{tile}>"

    # CB types for the compute (full domain shapes).
    cb_a_type = f"!ttl.cb<[{OUTPUT_ROWS}, {OUTPUT_COLS}], {tile}, {BF}>"
    cb_a_spec = f"<[{OUTPUT_ROWS}, {OUTPUT_COLS}], {tile}, {BF}>"
    cb_b_type = f"!ttl.cb<[{BCAST_ROWS}, {BCAST_COLS}], {tile}, {BF}>"
    cb_b_spec = f"<[{BCAST_ROWS}, {BCAST_COLS}], {tile}, {BF}>"
    cb_out_type = cb_a_type
    cb_out_spec = cb_a_spec

    # Layout attributes.
    layout_attrs = f"""
#buffer = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{OUTPUT_ROWS}x{OUTPUT_COLS}x{tile}, #buffer>, <interleaved>>
#layout_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{BCAST_ROWS}x{BCAST_COLS}x{tile}, #buffer>, <interleaved>>
#map_identity = affine_map<(d0, d1) -> (d0, d1)>
#map_col_bcast = affine_map<(d0, d1) -> (d0, 0)>
"""

    reader = f"""
// Reader: reads A (4x4) and B (4x1, broadcast) from DRAM into CBs.
func.func @reader_broadcast(%a: {a_dram}, %b: {b_dram})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32, 1 : i32],
                ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {BF}}} : {cb_type}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {BF}}} : {cb_type}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {OUTPUT_ROWS} : index
  %num_cols = arith.constant {OUTPUT_COLS} : index
  scf.for %row = %c0 to %num_rows step %c1 {{
    scf.for %col = %c0 to %num_cols step %c1 {{
      // Read A[row, col] into CB0.
      %reserve_a = ttl.cb_reserve %cb0 : {cb_spec} -> {cb_tensor}
      %slice_a = ttl.tensor_slice %a[%row, %col] : {a_dram} -> {a_slice}
      %xf_a = ttl.copy %slice_a, %cb0 : ({a_slice}, {cb_type}) -> !ttl.transfer_handle<read>
      ttl.wait %xf_a : !ttl.transfer_handle<read>
      ttl.cb_push %cb0 : {cb_spec}

      // Read B[row, 0] into CB1 (broadcast: always column 0).
      %reserve_b = ttl.cb_reserve %cb1 : {cb_spec} -> {cb_tensor}
      %slice_b = ttl.tensor_slice %b[%row, %c0] : {b_dram} -> {b_slice}
      %xf_b = ttl.copy %slice_b, %cb1 : ({b_slice}, {cb_type}) -> !ttl.transfer_handle<read>
      ttl.wait %xf_b : !ttl.transfer_handle<read>
      ttl.cb_push %cb1 : {cb_spec}
    }}
  }}
  func.return
}}"""

    compute = f"""
// Compute: add A + broadcast(B) using SFPU path (indexing maps differ).
// Uses ttl.compute directly with explicit broadcast indexing maps to bypass
// convert-ttl-to-compute (which currently does not detect broadcast from
// shape mismatch on ttl.add).
func.func @compute_broadcast_add()
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {BF}}} : {cb_a_type}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {BF}}} : {cb_b_type}
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {BF}}} : {cb_out_type}

  // Wait for full tensors from CBs.
  %a = ttl.cb_wait %cb0 : {cb_a_spec} -> {a_tensor}
  %a_att = ttl.attach_cb %a, %cb0 : ({a_tensor}, {cb_a_type}) -> {a_tensor}
  %b = ttl.cb_wait %cb1 : {cb_b_spec} -> {b_tensor}
  %b_att = ttl.attach_cb %b, %cb1 : ({b_tensor}, {cb_b_type}) -> {b_tensor}

  // Reserve output CB and create init tensor.
  %out = ttl.cb_reserve %cb_out : {cb_out_spec} -> {out_tensor}
  %init = tensor.empty() : {out_tensor}
  %init_att = ttl.attach_cb %init, %cb_out : ({out_tensor}, {cb_out_type}) -> {out_tensor}

  // Compute with explicit broadcast indexing maps:
  //   A: identity (d0, d1) -> (d0, d1)
  //   B: col broadcast (d0, d1) -> (d0, 0)
  //   output: identity (d0, d1) -> (d0, d1)
  %result = ttl.compute
      ins(%a_att, %b_att : {a_tensor}, {b_tensor})
      outs(%init_att : {out_tensor})
      {{indexing_maps = [#map_identity, #map_col_bcast, #map_identity],
       iterator_types = ["parallel", "parallel"]}} {{
  ^bb0(%a_tile: {tile}, %b_tile: {tile}, %out_tile: {tile}):
    %sum = ttl.tile_add %a_tile, %b_tile : {tile}
    ttl.tile_store %sum, %out : {tile}, {out_tensor}
    ttl.yield
  }} -> {out_tensor}

  // Push output, pop inputs.
  ttl.cb_push %cb_out : {cb_out_spec}
  ttl.cb_pop %cb1 : {cb_b_spec}
  ttl.cb_pop %cb0 : {cb_a_spec}
  return
}}"""

    writer = f"""
// Writer: writes output (4x4) from CB to DRAM.
func.func @writer(%out_arg: {out_dram})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [2 : i32],
                ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {BF}}} : {cb_type}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {OUTPUT_ROWS} : index
  %num_cols = arith.constant {OUTPUT_COLS} : index
  scf.for %row = %c0 to %num_rows step %c1 {{
    scf.for %col = %c0 to %num_cols step %c1 {{
      // Wait for compute result, copy to DRAM.
      %wait_out = ttl.cb_wait %cb_out : {cb_spec} -> {cb_tensor}
      %att_out = ttl.attach_cb %wait_out, %cb_out : ({cb_tensor}, {cb_type}) -> {cb_tensor}
      %slice_out = ttl.tensor_slice %out_arg[%row, %col] : {out_dram} -> {out_slice}
      %xf_out = ttl.copy %cb_out, %slice_out : ({cb_type}, {out_slice}) -> !ttl.transfer_handle<write>
      ttl.wait %xf_out : !ttl.transfer_handle<write>
      ttl.cb_pop %cb_out : {cb_spec}
    }}
  }}
  func.return
}}"""

    return f"""{layout_attrs}

module {{
{reader}

{compute}

{writer}
}}
"""


class TestBroadcastAdd(ME2ETestBase):
    """
    Regression test: binary add with column broadcast.

    Confirms that the SFPU fallback path produces correct results when
    FPU binary detection is skipped due to incompatible indexing maps.
    The 4x4 output grid requires DST subblocking, exercising the full
    pipeline: assign-dst -> subblock-compute-for-dst -> insert-sync ->
    lower-to-loops -> convert-to-ttkernel.
    """

    @pytest.mark.order(1)
    def test_build_module(self) -> None:
        """Build broadcast add module from hand-crafted MLIR."""
        from ttmlir.ir import Context, Module

        # Parse MLIR template.
        mlir_str = _build_broadcast_add_mlir()

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)
        with ctx:
            module = Module.parse(mlir_str, ctx)
            module.operation.verify()

        # Save module for subsequent stages.
        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(str(module))

        # Create inputs: A (128x128), B (128x32).
        a_shape = (OUTPUT_ROWS * 32, OUTPUT_COLS * 32)
        b_shape = (BCAST_ROWS * 32, BCAST_COLS * 32)
        torch.manual_seed(42)
        input_a = torch.rand(a_shape, dtype=DTYPE) * 2 - 1
        input_b = torch.rand(b_shape, dtype=DTYPE) * 2 - 1
        torch_inputs = [input_a, input_b]

        torch.save(torch_inputs, self.output_file("inputs.pt"))

        # Golden: A + B with column broadcast.
        # B is 4x1 tiles (128x32 elements). The hardware broadcasts each
        # 32-wide tile across all 4 column positions, so replicate B along
        # the column dimension to match A's shape.
        golden = input_a + input_b.repeat(1, OUTPUT_COLS)
        torch.save(golden, self.output_file("golden.pt"))

    @pytest.mark.order(2)
    def test_compile_to_ttkernel(self) -> None:
        """Compile broadcast add through TTL pipeline."""
        super().test_compile_to_ttkernel()

    @pytest.mark.order(3)
    def test_translate_to_cpp(self) -> None:
        """Translate broadcast add to C++ kernels."""
        super().test_translate_to_cpp()

    @pytest.mark.order(4)
    @pytest.mark.requires_device
    def test_execute(self, device) -> None:
        """Execute broadcast add on device."""
        super().test_execute(device)

    @pytest.mark.order(5)
    def test_validate_golden(self) -> None:
        """Validate broadcast add result against golden."""
        super().test_validate_golden()
