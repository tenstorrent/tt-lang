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


def _get_tensor_size(grid_shape: Tuple[int, int]) -> int:
    """Get tensor size in elements (assuming 32x32 tiles)."""
    rows, cols = grid_shape
    return rows * 32 * cols * 32


def generate_binary_reader_mlir(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
) -> str:
    """
    Generate MLIR for binary reader data movement thread.

    Reads two tensors from DRAM into CB0 and CB1.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.

    Returns:
        MLIR string for the reader function.
    """
    rows, cols = grid_shape
    tensor_size = _get_tensor_size(grid_shape)
    dtype_str = torch_dtype_to_mlir_str(dtype)

    return f"""
// Reader data movement thread for binary ops: reads A and B from DRAM into CB0 and CB1.
func.func @reader_binary(%a: tensor<{tensor_size}x{dtype_str}, #layout>,
                         %b: tensor<{tensor_size}x{dtype_str}, #layout>)
    attributes {{ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>
  %cb1 = ttl.bind_cb {{cb_index = 1, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>

  // Copy A to CB0.
  %xf_a = ttl.copy %a, %cb0 : (tensor<{tensor_size}x{dtype_str}, #layout>, !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_a : !ttl.transfer_handle<read>

  // Copy B to CB1.
  %xf_b = ttl.copy %b, %cb1 : (tensor<{tensor_size}x{dtype_str}, #layout>, !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_b : !ttl.transfer_handle<read>

  func.return
}}
"""


def generate_unary_reader_mlir(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
) -> str:
    """
    Generate MLIR for unary reader data movement thread.

    Reads one tensor from DRAM into CB0.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.

    Returns:
        MLIR string for the reader function.
    """
    rows, cols = grid_shape
    tensor_size = _get_tensor_size(grid_shape)
    dtype_str = torch_dtype_to_mlir_str(dtype)

    return f"""
// Reader data movement thread for unary ops: reads A from DRAM into CB0.
func.func @reader_unary(%a: tensor<{tensor_size}x{dtype_str}, #layout>)
    attributes {{ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb0 = ttl.bind_cb {{cb_index = 0, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>

  // Copy A to CB0.
  %xf_a = ttl.copy %a, %cb0 : (tensor<{tensor_size}x{dtype_str}, #layout>, !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>) -> !ttl.transfer_handle<read>
  ttl.wait %xf_a : !ttl.transfer_handle<read>

  func.return
}}
"""


def generate_writer_mlir(
    grid_shape: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
    buffer_factor: int = 2,
    output_cb_index: int = 2,
) -> str:
    """
    Generate MLIR for writer data movement thread.

    Writes output from CB to DRAM. Works for both unary and binary ops.

    Args:
        grid_shape: Tile grid shape (rows, cols).
        dtype: Data type for tiles.
        buffer_factor: CB buffer factor.
        output_cb_index: CB index for output (default 2 for binary, 1 for unary).

    Returns:
        MLIR string for the writer function.
    """
    rows, cols = grid_shape
    tensor_size = _get_tensor_size(grid_shape)
    dtype_str = torch_dtype_to_mlir_str(dtype)

    return f"""
// Writer data movement thread: writes output from CB{output_cb_index} to DRAM.
func.func @writer(%out: tensor<{tensor_size}x{dtype_str}, #layout>)
    attributes {{ttl.kernel_thread = #ttkernel.thread<noc>}} {{
  %cb_out = ttl.bind_cb {{cb_index = {output_cb_index}, buffer_factor = {buffer_factor}}} : !ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>

  // Wait for data from compute thread.
  %cb_view = ttl.cb_wait %cb_out : <[{rows}, {cols}], {dtype_str}, {buffer_factor}> -> tensor<{rows}x{cols}x{dtype_str}>

  // Copy from CB to output tensor.
  %xf_out = ttl.copy %cb_out, %out : (!ttl.cb<[{rows}, {cols}], {dtype_str}, {buffer_factor}>, tensor<{tensor_size}x{dtype_str}, #layout>) -> !ttl.transfer_handle<write>
  ttl.wait %xf_out : !ttl.transfer_handle<write>

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
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>, #dram>, <interleaved>>
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
