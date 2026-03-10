# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Broadcast + binary add ME2E test.

Tests explicit tile_bcast (col) followed by tile_add in a single ttl.compute
region. The tile_bcast reads from CB and writes to DST, then tile_add operates
on the broadcast result (DST) and another CB input (via copy_tile).

Uses a 4x4 tile grid requiring DST subblocking (16 tiles with
dstPerIteration=2 exceeds DST capacity of 8 for bf16).
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor

from ..base import ME2ETestBase
from ..config import E2EConfig

import ttl.dialects.ttl as ttl


ROWS, COLS = 4, 4
DTYPE = torch.bfloat16
DTYPE_STR = "bf16"
BF = ROWS * COLS  # buffer factor: must hold all tiles for bulk cb_wait


def _build_broadcast_add_mlir() -> str:
    """
    Build MLIR module for bcast(B) + A.

    Both A and B are 4x4 tile grids. Compute applies tile_bcast col on each
    B tile (broadcasting column 0 across the full 32x32 tile), then adds
    the result with the corresponding A tile.
    """
    tile = f"!ttcore.tile<32x32, {DTYPE_STR}>"
    tensor_ty = f"tensor<{ROWS}x{COLS}x{tile}>"

    # DRAM tensor types (with layout).
    dram_ty = f"tensor<{ROWS}x{COLS}x{tile}, #layout>"
    slice_ty = f"tensor<1x1x{tile}, #layout>"

    # CB types for reader/writer (1x1 tile processing).
    cb_type = f"!ttl.cb<[1, 1], {tile}, {BF}>"
    cb_spec = f"<[1, 1], {tile}, {BF}>"
    cb_tensor = f"tensor<1x1x{tile}>"

    # CB types for compute (full domain shapes).
    cb_full_type = f"!ttl.cb<[{ROWS}, {COLS}], {tile}, {BF}>"
    cb_full_spec = f"<[{ROWS}, {COLS}], {tile}, {BF}>"

    layout_attrs = f"""
#buffer = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{ROWS}x{COLS}x{tile}, #buffer>, <interleaved>>
#map = affine_map<(d0, d1) -> (d0, d1)>
"""

    reader = f"""
// Reader: reads A and B (both 4x4) from DRAM into CBs.
func.func @reader(%a: {dram_ty}, %b: {dram_ty})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32, 1 : i32],
                ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {BF}}} : {cb_type}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {BF}}} : {cb_type}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {ROWS} : index
  %num_cols = arith.constant {COLS} : index
  scf.for %row = %c0 to %num_rows step %c1 {{
    scf.for %col = %c0 to %num_cols step %c1 {{
      %reserve_a = ttl.cb_reserve %cb0 : {cb_spec} -> {cb_tensor}
      %slice_a = ttl.tensor_slice %a[%row, %col] : {dram_ty} -> {slice_ty}
      %xf_a = ttl.copy %slice_a, %cb0 : ({slice_ty}, {cb_type}) -> !ttl.transfer_handle<read>
      ttl.wait %xf_a : !ttl.transfer_handle<read>
      ttl.cb_push %cb0 : {cb_spec}

      %reserve_b = ttl.cb_reserve %cb1 : {cb_spec} -> {cb_tensor}
      %slice_b = ttl.tensor_slice %b[%row, %col] : {dram_ty} -> {slice_ty}
      %xf_b = ttl.copy %slice_b, %cb1 : ({slice_ty}, {cb_type}) -> !ttl.transfer_handle<read>
      ttl.wait %xf_b : !ttl.transfer_handle<read>
      ttl.cb_push %cb1 : {cb_spec}
    }}
  }}
  func.return
}}"""

    compute = f"""
// Compute: tile_bcast col on B, then tile_add with A.
// tile_bcast reads from CB (CBInputTileOp), tile_add operates on DST values.
func.func @compute_bcast_add()
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [],
                ttl.kernel_thread = #ttkernel.thread<compute>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {BF}}} : {cb_full_type}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {BF}}} : {cb_full_type}
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {BF}}} : {cb_full_type}

  %a = ttl.cb_wait %cb0 : {cb_full_spec} -> {tensor_ty}
  %a_att = ttl.attach_cb %a, %cb0 : ({tensor_ty}, {cb_full_type}) -> {tensor_ty}
  %b = ttl.cb_wait %cb1 : {cb_full_spec} -> {tensor_ty}
  %b_att = ttl.attach_cb %b, %cb1 : ({tensor_ty}, {cb_full_type}) -> {tensor_ty}

  %out = ttl.cb_reserve %cb_out : {cb_full_spec} -> {tensor_ty}
  %init = tensor.empty() : {tensor_ty}
  %init_att = ttl.attach_cb %init, %cb_out : ({tensor_ty}, {cb_full_type}) -> {tensor_ty}

  // Compute: bcast(B) col + A.
  // tile_bcast reads B from CB, writes to DST. tile_add reads both from DST.
  // A goes through copy_tile (CB -> DST) since tile_add is not a CB-input op.
  %result = ttl.compute
      ins(%b_att, %a_att : {tensor_ty}, {tensor_ty})
      outs(%init_att : {tensor_ty})
      {{indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]}} {{
  ^bb0(%b_tile: {tile}, %a_tile: {tile}, %out_tile: {tile}):
    %bcast = ttl.tile_bcast %b_tile, %out_tile 1 : i32 : ({tile}, {tile}) -> {tile}
    %sum = ttl.tile_add %bcast, %a_tile : {tile}
    ttl.tile_store %sum, %out : {tile}, {tensor_ty}
    ttl.yield
  }} -> {tensor_ty}

  ttl.cb_push %cb_out : {cb_full_spec}
  ttl.cb_pop %cb1 : {cb_full_spec}
  ttl.cb_pop %cb0 : {cb_full_spec}
  return
}}"""

    writer = f"""
// Writer: writes output (4x4) from CB to DRAM.
func.func @writer(%out_arg: {dram_ty})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [2 : i32],
                ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb_out = ttl.bind_cb {{cb_index = 2, buffer_factor = {BF}}} : {cb_type}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {ROWS} : index
  %num_cols = arith.constant {COLS} : index
  scf.for %row = %c0 to %num_rows step %c1 {{
    scf.for %col = %c0 to %num_cols step %c1 {{
      %wait_out = ttl.cb_wait %cb_out : {cb_spec} -> {cb_tensor}
      %att_out = ttl.attach_cb %wait_out, %cb_out : ({cb_tensor}, {cb_type}) -> {cb_tensor}
      %slice_out = ttl.tensor_slice %out_arg[%row, %col] : {dram_ty} -> {slice_ty}
      %xf_out = ttl.copy %cb_out, %slice_out : ({cb_type}, {slice_ty}) -> !ttl.transfer_handle<write>
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


def _col_broadcast_golden(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute golden for tile_bcast col + tile_add.

    tile_bcast col broadcasts column 0 of each 32x32 tile across all 32
    columns of that tile. The result is then added element-wise with A.
    """
    result = a.clone()
    tile_h, tile_w = 32, 32
    rows, cols = a.shape
    for tr in range(0, rows, tile_h):
        for tc in range(0, cols, tile_w):
            # Column 0 of this B tile, broadcast across all columns.
            b_col = b[tr : tr + tile_h, tc : tc + 1]  # (32, 1)
            b_bcast = b_col.expand(tile_h, tile_w)  # (32, 32)
            result[tr : tr + tile_h, tc : tc + tile_w] = (
                a[tr : tr + tile_h, tc : tc + tile_w] + b_bcast
            )
    return result


class TestBroadcastAdd(ME2ETestBase):
    """
    Test: explicit tile_bcast col + tile_add.

    Exercises the bcast-then-add pattern in a single ttl.compute region with
    a 4x4 tile grid requiring DST subblocking. tile_bcast reads from CB,
    tile_add operates on DST values.
    """

    @pytest.mark.order(1)
    def test_build_module(self) -> None:
        """Build broadcast add module from hand-crafted MLIR."""
        from ttmlir.ir import Context, Module

        mlir_str = _build_broadcast_add_mlir()

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)
        with ctx:
            module = Module.parse(mlir_str, ctx)
            module.operation.verify()

        module_file = self.output_file("module.mlir")
        with open(module_file, "w") as f:
            f.write(str(module))

        # Both A and B are 4x4 tiles (128x128 elements).
        shape = (ROWS * 32, COLS * 32)
        torch.manual_seed(42)
        input_a = torch.rand(shape, dtype=DTYPE) * 2 - 1
        input_b = torch.rand(shape, dtype=DTYPE) * 2 - 1
        torch_inputs = [input_a, input_b]

        torch.save(torch_inputs, self.output_file("inputs.pt"))

        golden = _col_broadcast_golden(input_a, input_b)
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
