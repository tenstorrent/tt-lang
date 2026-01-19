# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Data movement thread templates for E2E tests.

Provides shared reader/writer data movement threads for binary and unary operations.
These threads handle NOC data movement between DRAM and circular buffers.

Terminology (per TT-Metalium):
- Data Movement (Reader) Threads: Load data from DRAM/L1 into CBs (BRISC/NCRISC)
- Data Movement (Writer) Threads: Store data from CBs to DRAM/L1 (BRISC/NCRISC)
- Compute Threads: Perform math operations on data in CBs (TRISC)
"""

from typing import Tuple

import torch

from .dtype_utils import torch_dtype_to_mlir_str


def _get_tensor_type_str(grid_shape: Tuple[int, int], dtype_str: str) -> str:
    """Get tensor-of-tiles type string for the given grid shape.

    For TTL, tensors are 4D: [grid_row, grid_col, shard_row, shard_col].
    For single-core per tile (1x1 shard), this is [rows, cols, 1, 1].
    """
    rows, cols = grid_shape
    return f"tensor<{rows}x{cols}x1x1x!ttcore.tile<32x32, {dtype_str}>, #layout>"


def generate_binary_reader_mlir(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
    num_iterations: int = 1,
) -> str:
    """
    Generate MLIR for binary reader data movement thread.

    Reads two tensors from DRAM into CB0 and CB1.
    For multi-tile grids, generates nested loops to read one tile at a time.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.
        num_iterations: Number of loop iterations (tiles to process).

    Returns:
        MLIR string for the reader function.
    """
    rows, cols = grid_shape
    dtype_str = torch_dtype_to_mlir_str(dtype)
    tensor_type = _get_tensor_type_str(grid_shape, dtype_str)

    # CB processes 1 tile at a time.
    cb_rows, cb_cols = 1, 1
    cb_type = f"!ttl.cb<[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>"
    cb_tensor_type = f"tensor<{cb_rows}x{cb_cols}x!ttcore.tile<32x32, {dtype_str}>>"
    # Slice type is 4D with shard dims.
    slice_type = f"tensor<{rows}x{cols}x{cb_rows}x{cb_cols}x!ttcore.tile<32x32, {dtype_str}>, #layout>"

    if num_iterations > 1:
        # Multi-tile: nested loops over row/col using affine.apply for indices.
        loop_start = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {rows} : index
  %num_cols = arith.constant {cols} : index
  scf.for %tile_row = %c0 to %num_rows step %c1 {{
    scf.for %tile_col = %c0 to %num_cols step %c1 {{
      %row_idx = affine.apply affine_map<(d0) -> (d0)>(%tile_row)
      %col_idx = affine.apply affine_map<(d0) -> (d0)>(%tile_col)"""
        loop_end = """
    }
  }"""
        indent = "      "
        row_idx = "%row_idx"
        col_idx = "%col_idx"
    else:
        loop_start = """
  %c0 = arith.constant 0 : index"""
        loop_end = ""
        indent = "  "
        row_idx = "%c0"
        col_idx = "%c0"

    return f"""
// Reader data movement thread for binary ops: reads A and B from DRAM into CB0 and CB1.
func.func @reader_binary(%a: {tensor_type},
                         %b: {tensor_type})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32, 1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : {cb_type}
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {buffer_factor}}} : {cb_type}
{loop_start}

{indent}// Reserve CB0, slice and copy A tile.
{indent}%reserve_a = ttl.cb_reserve %cb0 : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {cb_tensor_type}
{indent}%slice_a = ttl.tensor_slice %a[{row_idx}, {col_idx}] : {tensor_type} -> {slice_type}
{indent}%xf_a = ttl.copy %slice_a, %cb0 : ({slice_type}, {cb_type}) -> !ttl.transfer_handle<read>
{indent}ttl.wait %xf_a : !ttl.transfer_handle<read>
{indent}ttl.cb_push %cb0 : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>

{indent}// Reserve CB1, slice and copy B tile.
{indent}%reserve_b = ttl.cb_reserve %cb1 : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {cb_tensor_type}
{indent}%slice_b = ttl.tensor_slice %b[{row_idx}, {col_idx}] : {tensor_type} -> {slice_type}
{indent}%xf_b = ttl.copy %slice_b, %cb1 : ({slice_type}, {cb_type}) -> !ttl.transfer_handle<read>
{indent}ttl.wait %xf_b : !ttl.transfer_handle<read>
{indent}ttl.cb_push %cb1 : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{loop_end}
  func.return
}}
"""


def generate_unary_reader_mlir(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
    num_iterations: int = 1,
) -> str:
    """
    Generate MLIR for unary reader data movement thread.

    Reads one tensor from DRAM into CB0.
    For multi-tile grids, generates nested loops to read one tile at a time.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.
        num_iterations: Number of loop iterations (tiles to process).

    Returns:
        MLIR string for the reader function.
    """
    rows, cols = grid_shape
    dtype_str = torch_dtype_to_mlir_str(dtype)
    tensor_type = _get_tensor_type_str(grid_shape, dtype_str)

    # CB processes 1 tile at a time.
    cb_rows, cb_cols = 1, 1
    cb_type = f"!ttl.cb<[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>"
    cb_tensor_type = f"tensor<{cb_rows}x{cb_cols}x!ttcore.tile<32x32, {dtype_str}>>"
    # Slice type is 4D with shard dims.
    slice_type = f"tensor<{rows}x{cols}x{cb_rows}x{cb_cols}x!ttcore.tile<32x32, {dtype_str}>, #layout>"

    if num_iterations > 1:
        # Multi-tile: nested loops over row/col using affine.apply for indices.
        loop_start = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {rows} : index
  %num_cols = arith.constant {cols} : index
  scf.for %tile_row = %c0 to %num_rows step %c1 {{
    scf.for %tile_col = %c0 to %num_cols step %c1 {{
      %row_idx = affine.apply affine_map<(d0) -> (d0)>(%tile_row)
      %col_idx = affine.apply affine_map<(d0) -> (d0)>(%tile_col)"""
        loop_end = """
    }
  }"""
        indent = "      "
        row_idx = "%row_idx"
        col_idx = "%col_idx"
    else:
        loop_start = """
  %c0 = arith.constant 0 : index"""
        loop_end = ""
        indent = "  "
        row_idx = "%c0"
        col_idx = "%c0"

    return f"""
// Reader data movement thread for unary ops: reads A from DRAM into CB0.
func.func @reader_unary(%a: {tensor_type})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : {cb_type}
{loop_start}

{indent}// Reserve CB0, slice and copy A tile.
{indent}%reserve_a = ttl.cb_reserve %cb0 : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {cb_tensor_type}
{indent}%slice_a = ttl.tensor_slice %a[{row_idx}, {col_idx}] : {tensor_type} -> {slice_type}
{indent}%xf_a = ttl.copy %slice_a, %cb0 : ({slice_type}, {cb_type}) -> !ttl.transfer_handle<read>
{indent}ttl.wait %xf_a : !ttl.transfer_handle<read>
{indent}ttl.cb_push %cb0 : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{loop_end}
  func.return
}}
"""


def generate_writer_mlir(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
    output_cb_index: int = 2,
    num_iterations: int = 1,
) -> str:
    """
    Generate MLIR for writer data movement thread.

    Writes output from CB to DRAM. Works for both unary and binary ops.
    For multi-tile grids, generates nested loops to write one tile at a time.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.
        output_cb_index: CB index for output (default 2 for binary, 1 for unary).
        num_iterations: Number of loop iterations (tiles to process).

    Returns:
        MLIR string for the writer function.
    """
    rows, cols = grid_shape
    dtype_str = torch_dtype_to_mlir_str(dtype)
    tensor_type = _get_tensor_type_str(grid_shape, dtype_str)

    # CB processes 1 tile at a time.
    cb_rows, cb_cols = 1, 1
    cb_type = f"!ttl.cb<[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>"
    cb_tensor_type = f"tensor<{cb_rows}x{cb_cols}x!ttcore.tile<32x32, {dtype_str}>>"
    # Slice type is 4D with shard dims.
    slice_type = f"tensor<{rows}x{cols}x{cb_rows}x{cb_cols}x!ttcore.tile<32x32, {dtype_str}>, #layout>"

    if num_iterations > 1:
        # Multi-tile: nested loops over row/col using affine.apply for indices.
        loop_start = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {rows} : index
  %num_cols = arith.constant {cols} : index
  scf.for %tile_row = %c0 to %num_rows step %c1 {{
    scf.for %tile_col = %c0 to %num_cols step %c1 {{
      %row_idx = affine.apply affine_map<(d0) -> (d0)>(%tile_row)
      %col_idx = affine.apply affine_map<(d0) -> (d0)>(%tile_col)"""
        loop_end = """
    }
  }"""
        indent = "      "
        row_idx = "%row_idx"
        col_idx = "%col_idx"
    else:
        loop_start = """
  %c0 = arith.constant 0 : index"""
        loop_end = ""
        indent = "  "
        row_idx = "%c0"
        col_idx = "%c0"

    return f"""
// Writer data movement thread: writes output from CB{output_cb_index} to DRAM.
func.func @writer(%out: {tensor_type})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [{output_cb_index} : i32], ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb_out = ttl.bind_cb {{cb_index = {output_cb_index}, buffer_factor = {buffer_factor}}} : {cb_type}
{loop_start}

{indent}// Wait for output CB, slice, copy to DRAM, pop.
{indent}%cb_wait_view = ttl.cb_wait %cb_out : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}> -> {cb_tensor_type}
{indent}%attached = ttl.attach_cb %cb_wait_view, %cb_out : ({cb_tensor_type}, {cb_type}) -> {cb_tensor_type}
{indent}%slice_out = ttl.tensor_slice %out[{row_idx}, {col_idx}] : {tensor_type} -> {slice_type}
{indent}%xf_out = ttl.copy %cb_out, %slice_out : ({cb_type}, {slice_type}) -> !ttl.transfer_handle<write>
{indent}ttl.wait %xf_out : !ttl.transfer_handle<write>
{indent}ttl.cb_pop %cb_out : <[{cb_rows}, {cb_cols}], !ttcore.tile<32x32, {dtype_str}>, {buffer_factor}>
{loop_end}
  func.return
}}
"""


def generate_layout_attrs(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> str:
    """
    Generate MLIR layout attributes for DRAM tensors.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.

    Returns:
        MLIR string with layout attribute definitions.
    """
    rows, cols = grid_shape
    dtype_str = torch_dtype_to_mlir_str(dtype)

    return f"""
#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{rows}x{cols}x1x1x!ttcore.tile<32x32, {dtype_str}>, #dram>, <interleaved>>
#map = affine_map<(d0, d1) -> (d0, d1)>
"""


def generate_full_module_mlir(
    op_str: str,
    arity: int,
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
    compute_mlir: str = "",
) -> str:
    """
    Generate complete MLIR module with reader, compute, and writer threads.

    Args:
        op_str: Operation name (e.g., "add", "exp").
        arity: Number of inputs (1 or 2).
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.
        compute_mlir: MLIR string for the compute function (if not provided, placeholder).

    Returns:
        Complete MLIR module string with data movement and compute threads.
    """
    layout_attrs = generate_layout_attrs(grid_shape, dtype)

    if arity == 2:
        reader_mlir = generate_binary_reader_mlir(grid_shape, dtype, buffer_factor)
        output_cb_index = 2
    else:
        reader_mlir = generate_unary_reader_mlir(grid_shape, dtype, buffer_factor)
        output_cb_index = 1

    writer_mlir = generate_writer_mlir(
        grid_shape, dtype, buffer_factor, output_cb_index
    )

    # Default compute placeholder if not provided.
    if not compute_mlir:
        compute_mlir = f"// Compute thread for {op_str} - to be inserted"

    return f"""// Auto-generated MLIR module for {op_str} operation.
// Arity: {arity}, Grid: {grid_shape}, Dtype: {dtype}

{layout_attrs}

module {{
{reader_mlir}

{compute_mlir}

{writer_mlir}
}}
"""
