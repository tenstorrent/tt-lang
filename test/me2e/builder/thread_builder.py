# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for thread builders in ME2E tests.

Provides common building blocks for constructing MLIR thread functions:
- Type factories (tile, CB, tensor types)
- Loop constructs (single-tile, multi-tile)
- CB operations (bind, reserve, wait, push, pop, attach)
- Transfer operations (slice, copy, wait)

Subclasses (DMThreadBuilder, ComputeThreadBuilder) use these building blocks
to create specific thread types with minimal boilerplate.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import torch
from ttmlir.ir import (
    Attribute,
    ArrayAttr,
    Context,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
)
from ttmlir.dialects import arith, func, scf
from ttmlir.dialects import ttcore

import ttl.dialects.ttl as ttl

from ..config import E2EConfig
from .dtype_utils import torch_dtype_to_mlir_str, torch_dtype_to_ttcore_datatype


class ThreadType(Enum):
    """Thread type for kernel functions."""

    NOC = "noc"
    COMPUTE = "compute"


@dataclass
class LoopContext:
    """Context for tile loop iteration."""

    row_idx: Any  # SSA value for row index
    col_idx: Any  # SSA value for column index


class ThreadBuilder(ABC):
    """
    Base class for thread builders.

    Provides building blocks for constructing MLIR thread functions.
    Subclasses override specific methods to create DM or compute threads.

    The builder operates on a shared Module and maintains cached types
    for efficiency.
    """

    def __init__(
        self,
        module: Module,
        ctx: Context,
        loc: Location,
        config: E2EConfig,
    ):
        """
        Initialize the thread builder.

        Args:
            module: The MLIR module to add functions to.
            ctx: The MLIR context.
            loc: The location for operations.
            config: Test configuration.
        """
        self.module = module
        self.ctx = ctx
        self.loc = loc
        self.config = config

        # Cache commonly used values.
        self._dtype_str = torch_dtype_to_mlir_str(config.dtype)
        self._dtype_int = torch_dtype_to_ttcore_datatype(config.dtype)
        self._rows, self._cols = config.grid_shape
        self._buffer_factor = config.buffer_factor
        self._num_iterations = config.num_tiles

        # CB processes 1 tile at a time.
        self._cb_rows = 1
        self._cb_cols = 1

        # Create cached types.
        self._tile_type = self._create_tile_type()
        self._cb_type = self._create_cb_type()
        self._tile_tensor_type = self._create_tile_tensor_type()

    # =========================================================================
    # Layer 4: Type Properties (Public Read-Only)
    # =========================================================================

    @property
    def tile_type(self):
        """Get the tile type (!ttcore.tile<32x32, dtype>)."""
        return self._tile_type

    @property
    def cb_type(self):
        """Get the CB type (!ttl.cb<[1,1], tile, buffer_factor>)."""
        return self._cb_type

    @property
    def tile_tensor_type(self):
        """Get the tile tensor type (tensor<1x1xtile>)."""
        return self._tile_tensor_type

    # =========================================================================
    # Layer 3: Low-Level Type Factories (Private)
    # =========================================================================

    def _create_tile_type(self):
        """Create the ttcore.tile type."""
        return ttcore.ir.TileType.get(self.ctx, 32, 32, self._dtype_int)

    def _create_cb_type(self):
        """Create the circular buffer type for 1x1 tile processing."""
        return ttl.CircularBufferType.get(
            self.ctx,
            [self._cb_rows, self._cb_cols],
            self._tile_type,
            self._buffer_factor,
        )

    def _create_tile_tensor_type(self):
        """Create the tensor type for CB operations (1x1 tiles)."""
        return RankedTensorType.get([self._cb_rows, self._cb_cols], self._tile_type)

    # =========================================================================
    # Layer 3: Low-Level CB Operations (Private)
    # =========================================================================

    def _bind_cb(self, cb_index: int):
        """Create ttl.bind_cb operation."""
        return ttl.bind_cb(
            self._cb_type,
            cb_index=cb_index,
            buffer_factor=self._buffer_factor,
            loc=self.loc,
        )

    def _cb_reserve(self, cb):
        """Create ttl.cb_reserve operation."""
        return ttl.cb_reserve(self._tile_tensor_type, cb, loc=self.loc)

    def _cb_wait(self, cb):
        """Create ttl.cb_wait operation."""
        return ttl.cb_wait(self._tile_tensor_type, cb, loc=self.loc)

    def _cb_push(self, cb):
        """Create ttl.cb_push operation."""
        ttl.cb_push(cb, loc=self.loc)

    def _cb_pop(self, cb):
        """Create ttl.cb_pop operation."""
        ttl.cb_pop(cb, loc=self.loc)

    def _attach_cb(self, tensor, cb):
        """Create ttl.attach_cb operation."""
        return ttl.attach_cb(self._tile_tensor_type, tensor, cb, loc=self.loc)

    def _tensor_store(self, tensor, cb):
        """Create ttl.tensor_store operation to mark explicit store."""
        ttl.tensor_store(tensor, cb, loc=self.loc)

    # =========================================================================
    # Layer 2: Loop and Function Utilities (Protected)
    # =========================================================================

    def _with_tile_loop(self, body_fn: Callable[[LoopContext], None]) -> None:
        """
        Execute body_fn within a tile iteration loop if needed.

        For single-tile (num_iterations == 1), calls body_fn directly.
        For multi-tile, creates nested scf.for loops over rows and cols.

        Args:
            body_fn: Callback that receives a LoopContext with row/col indices.
        """
        if self._num_iterations > 1:
            # Create loop bounds.
            c0 = arith.ConstantOp(IndexType.get(self.ctx), 0).result
            c1 = arith.ConstantOp(IndexType.get(self.ctx), 1).result
            num_rows = arith.ConstantOp(IndexType.get(self.ctx), self._rows).result
            num_cols = arith.ConstantOp(IndexType.get(self.ctx), self._cols).result

            # Nested loops over rows and cols.
            row_loop = scf.ForOp(c0, num_rows, c1)
            with InsertionPoint(row_loop.body):
                col_loop = scf.ForOp(c0, num_cols, c1)
                with InsertionPoint(col_loop.body):
                    # Get loop induction variables.
                    row_idx = row_loop.induction_variable
                    col_idx = col_loop.induction_variable
                    body_fn(LoopContext(row_idx=row_idx, col_idx=col_idx))
                    scf.YieldOp([])
                scf.YieldOp([])
        else:
            # Single tile - use constant 0 indices.
            c0 = arith.ConstantOp(IndexType.get(self.ctx), 0).result
            body_fn(LoopContext(row_idx=c0, col_idx=c0))

    def _create_function(
        self,
        name: str,
        arg_types: List,
        result_types: List,
        thread_type: ThreadType,
        crta_indices: List[int],
        base_cta_index: Optional[int] = None,
    ) -> Tuple[func.FuncOp, List]:
        """
        Create a function with thread attributes.

        Args:
            name: Function name.
            arg_types: List of argument types.
            result_types: List of result types.
            thread_type: NOC or COMPUTE thread type.
            crta_indices: CB runtime argument indices.
            base_cta_index: Index where TensorAccessorArgs start in compile_time_args.
                           If None, uses total CB count from config.

        Returns:
            Tuple of (FuncOp, list of entry block arguments).
        """
        func_type = FunctionType.get(arg_types, result_types)
        fn = func.FuncOp(name, func_type, loc=self.loc)

        # Set thread attributes.
        i32_type = IntegerType.get_signless(32, self.ctx)
        if base_cta_index is None:
            # Calculate from CB count: num_inputs + num_outputs.
            # For ME2E tests, this is typically arity + 1 (inputs + 1 output).
            base_cta_index = getattr(self, "_total_cb_count", 3)
        fn.attributes["ttl.base_cta_index"] = IntegerAttr.get(i32_type, base_cta_index)
        fn.attributes["ttl.crta_indices"] = ArrayAttr.get(
            [IntegerAttr.get(i32_type, idx) for idx in crta_indices], self.ctx
        )
        fn.attributes["ttl.kernel_thread"] = Attribute.parse(
            f"#ttkernel.thread<{thread_type.value}>", self.ctx
        )

        entry_block = fn.add_entry_block()
        return fn, list(entry_block.arguments)

    # =========================================================================
    # Layer 1: High-Level Thread Builders (Protected)
    # =========================================================================

    def _build_compute_thread(
        self,
        name: str,
        input_cbs: List[int],
        output_cbs: List[int],
        compute_fn: Callable[[List], List],
    ) -> None:
        """
        Build a complete compute thread function.

        Creates a function with:
        - CB bindings for inputs and outputs
        - Optional tile loop for multi-tile processing
        - CB lifecycle: wait inputs -> reserve outputs -> compute -> push outputs -> pop inputs

        Args:
            name: Function name.
            input_cbs: List of input CB indices.
            output_cbs: List of output CB indices.
            compute_fn: Callback that takes list of input tensors, returns list of output tensors.
        """
        with InsertionPoint(self.module.body):
            fn, entry_block = self._create_function(
                name=name,
                arg_types=[],
                result_types=[],
                thread_type=ThreadType.COMPUTE,
                crta_indices=[],
            )

            with InsertionPoint(fn.entry_block):
                # Bind all CBs.
                input_cb_vals = [self._bind_cb(idx) for idx in input_cbs]
                output_cb_vals = [self._bind_cb(idx) for idx in output_cbs]

                def loop_body(loop_ctx: LoopContext):
                    # Wait for input data and attach.
                    inputs = []
                    for cb in input_cb_vals:
                        tensor = self._cb_wait(cb)
                        attached = self._attach_cb(tensor, cb)
                        inputs.append(attached)

                    # Reserve output CBs and attach.
                    outputs = []
                    for cb in output_cb_vals:
                        reserved = self._cb_reserve(cb)
                        attached = self._attach_cb(reserved, cb)
                        outputs.append(attached)

                    # Execute compute function.
                    results = compute_fn(inputs)

                    # Store results to output CBs.
                    if not isinstance(results, list):
                        results = [results]
                    for result, cb in zip(results, output_cb_vals):
                        self._tensor_store(result, cb)
                        self._attach_cb(result, cb)

                    # Push outputs, pop inputs.
                    for cb in output_cb_vals:
                        self._cb_push(cb)
                    for cb in reversed(input_cb_vals):
                        self._cb_pop(cb)

                self._with_tile_loop(loop_body)
                func.ReturnOp([], loc=self.loc)


class StringBasedThreadBuilder:
    """
    String-based thread builder for DM threads.

    DM threads require DRAM tensor types with layout attributes (#layout),
    which are complex to create with Python bindings. This builder uses
    string templates with helper methods to eliminate duplication.
    """

    def __init__(self, config: E2EConfig):
        """
        Initialize the string-based builder.

        Args:
            config: Test configuration.
        """
        self.config = config
        self._dtype_str = torch_dtype_to_mlir_str(config.dtype)
        self._rows, self._cols = config.grid_shape
        self._buffer_factor = config.buffer_factor
        self._num_iterations = config.num_tiles
        self._cb_rows = 1
        self._cb_cols = 1
        # Layout attribute reference (matches generate_layout_attrs).
        self._layout_ref = "#layout"

    # =========================================================================
    # Type Strings
    # =========================================================================

    @property
    def dram_tensor_type_str(self) -> str:
        """Get tensor type string with layout attribute.

        Note: Named dram_tensor_type_str for historical reasons, but now supports
        any buffer type (DRAM or L1) based on config.

        Uses 2D shape [tiles_y, tiles_x] to match Python DSL approach.
        """
        return (
            f"tensor<{self._rows}x{self._cols}x"
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._layout_ref}>"
        )

    @property
    def cb_type_str(self) -> str:
        """Get CB type string."""
        return (
            f"!ttl.cb<[{self._cb_rows}, {self._cb_cols}], "
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._buffer_factor}>"
        )

    @property
    def cb_tensor_type_str(self) -> str:
        """Get CB tensor type string (for cb_reserve result)."""
        return (
            f"tensor<{self._cb_rows}x{self._cb_cols}x"
            f"!ttcore.tile<32x32, {self._dtype_str}>>"
        )

    @property
    def slice_type_str(self) -> str:
        """Get slice type string for tensor_slice result.

        Slice result has CB shape [cb_rows, cb_cols], matching the tile block
        being transferred.
        """
        return (
            f"tensor<{self._cb_rows}x{self._cb_cols}x"
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._layout_ref}>"
        )

    # =========================================================================
    # Loop Generation
    # =========================================================================

    def _generate_loop_start(self) -> Tuple[str, str, str, str]:
        """
        Generate loop start code and index variables.

        Returns:
            Tuple of (loop_start_code, loop_end_code, row_idx_var, col_idx_var).
        """
        if self._num_iterations > 1:
            loop_start = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %num_rows = arith.constant {self._rows} : index
  %num_cols = arith.constant {self._cols} : index
  scf.for %tile_row = %c0 to %num_rows step %c1 {{
    scf.for %tile_col = %c0 to %num_cols step %c1 {{"""
            loop_end = """
    }
  }"""
            return loop_start, loop_end, "%tile_row", "%tile_col"
        else:
            loop_start = """
  %c0 = arith.constant 0 : index"""
            return loop_start, "", "%c0", "%c0"

    # =========================================================================
    # CB Operation Strings
    # =========================================================================

    def _cb_reserve_str(self, cb_var: str, result_var: str) -> str:
        """Generate cb_reserve operation string."""
        return (
            f"{result_var} = ttl.cb_reserve {cb_var} : "
            f"<[{self._cb_rows}, {self._cb_cols}], "
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._buffer_factor}> "
            f"-> {self.cb_tensor_type_str}"
        )

    def _cb_wait_str(self, cb_var: str, result_var: str) -> str:
        """Generate cb_wait operation string."""
        return (
            f"{result_var} = ttl.cb_wait {cb_var} : "
            f"<[{self._cb_rows}, {self._cb_cols}], "
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._buffer_factor}> "
            f"-> {self.cb_tensor_type_str}"
        )

    def _cb_push_str(self, cb_var: str) -> str:
        """Generate cb_push operation string."""
        return (
            f"ttl.cb_push {cb_var} : "
            f"<[{self._cb_rows}, {self._cb_cols}], "
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._buffer_factor}>"
        )

    def _cb_pop_str(self, cb_var: str) -> str:
        """Generate cb_pop operation string."""
        return (
            f"ttl.cb_pop {cb_var} : "
            f"<[{self._cb_rows}, {self._cb_cols}], "
            f"!ttcore.tile<32x32, {self._dtype_str}>, {self._buffer_factor}>"
        )

    def _attach_cb_str(self, tensor_var: str, cb_var: str, result_var: str) -> str:
        """Generate attach_cb operation string."""
        return (
            f"{result_var} = ttl.attach_cb {tensor_var}, {cb_var} : "
            f"({self.cb_tensor_type_str}, {self.cb_type_str}) "
            f"-> {self.cb_tensor_type_str}"
        )

    def _tensor_slice_str(
        self, tensor_var: str, row_idx: str, col_idx: str, result_var: str
    ) -> str:
        """Generate tensor_slice operation string."""
        return (
            f"{result_var} = ttl.tensor_slice {tensor_var}[{row_idx}, {col_idx}] : "
            f"{self.dram_tensor_type_str} -> {self.slice_type_str}"
        )

    def _copy_read_str(self, slice_var: str, cb_var: str, result_var: str) -> str:
        """Generate copy (read) operation string."""
        return (
            f"{result_var} = ttl.copy {slice_var}, {cb_var} : "
            f"({self.slice_type_str}, {self.cb_type_str}) "
            f"-> !ttl.transfer_handle<read>"
        )

    def _copy_write_str(self, cb_var: str, slice_var: str, result_var: str) -> str:
        """Generate copy (write) operation string."""
        return (
            f"{result_var} = ttl.copy {cb_var}, {slice_var} : "
            f"({self.cb_type_str}, {self.slice_type_str}) "
            f"-> !ttl.transfer_handle<write>"
        )

    def _wait_read_str(self, handle_var: str) -> str:
        """Generate wait (read) operation string."""
        return f"ttl.wait {handle_var} : !ttl.transfer_handle<read>"

    def _wait_write_str(self, handle_var: str) -> str:
        """Generate wait (write) operation string."""
        return f"ttl.wait {handle_var} : !ttl.transfer_handle<write>"

    # =========================================================================
    # High-Level Helpers
    # =========================================================================

    def _read_to_cb_str(
        self,
        tensor_var: str,
        cb_var: str,
        row_idx: str,
        col_idx: str,
        prefix: str,
        indent: str,
    ) -> str:
        """
        Generate MLIR for reading a tile from DRAM tensor to CB.

        Sequence: cb_reserve -> tensor_slice -> copy -> wait -> cb_push

        Args:
            tensor_var: Source tensor SSA variable.
            cb_var: Destination CB SSA variable.
            row_idx: Row index variable.
            col_idx: Column index variable.
            prefix: Unique prefix for SSA variables.
            indent: Indentation string.

        Returns:
            MLIR string for the read operation.
        """
        lines = [
            f"{indent}// Reserve CB, slice and copy tile.",
            f"{indent}%reserve_{prefix} = {self._cb_reserve_str(cb_var, f'%reserve_{prefix}').split(' = ', 1)[1]}",
            f"{indent}%slice_{prefix} = {self._tensor_slice_str(tensor_var, row_idx, col_idx, f'%slice_{prefix}').split(' = ', 1)[1]}",
            f"{indent}%xf_{prefix} = {self._copy_read_str(f'%slice_{prefix}', cb_var, f'%xf_{prefix}').split(' = ', 1)[1]}",
            f"{indent}{self._wait_read_str(f'%xf_{prefix}')}",
            f"{indent}{self._cb_push_str(cb_var)}",
        ]
        return "\n".join(lines)

    def _write_from_cb_str(
        self,
        cb_var: str,
        tensor_var: str,
        row_idx: str,
        col_idx: str,
        prefix: str,
        indent: str,
    ) -> str:
        """
        Generate MLIR for writing a tile from CB to DRAM tensor.

        Sequence: cb_wait -> attach_cb -> tensor_slice -> copy -> wait -> cb_pop

        Args:
            cb_var: Source CB SSA variable.
            tensor_var: Destination tensor SSA variable.
            row_idx: Row index variable.
            col_idx: Column index variable.
            prefix: Unique prefix for SSA variables.
            indent: Indentation string.

        Returns:
            MLIR string for the write operation.
        """
        lines = [
            f"{indent}// Wait for CB, slice, copy to DRAM, pop.",
            f"{indent}%wait_{prefix} = {self._cb_wait_str(cb_var, f'%wait_{prefix}').split(' = ', 1)[1]}",
            f"{indent}%attached_{prefix} = {self._attach_cb_str(f'%wait_{prefix}', cb_var, f'%attached_{prefix}').split(' = ', 1)[1]}",
            f"{indent}%slice_{prefix} = {self._tensor_slice_str(tensor_var, row_idx, col_idx, f'%slice_{prefix}').split(' = ', 1)[1]}",
            f"{indent}%xf_{prefix} = {self._copy_write_str(cb_var, f'%slice_{prefix}', f'%xf_{prefix}').split(' = ', 1)[1]}",
            f"{indent}{self._wait_write_str(f'%xf_{prefix}')}",
            f"{indent}{self._cb_pop_str(cb_var)}",
        ]
        return "\n".join(lines)


def generate_layout_attrs(config: E2EConfig) -> str:
    """
    Generate MLIR layout attributes for tensors.

    Uses config.buffer_type and config.memory_layout to generate appropriate
    layout attributes.

    Args:
        config: Test configuration.

    Returns:
        MLIR string with layout attribute definitions.
    """
    rows, cols = config.grid_shape
    dtype_str = torch_dtype_to_mlir_str(config.dtype)

    # Buffer type attribute (dram or l1).
    buffer_type = config.buffer_type.value  # "dram" or "l1"
    buffer_attr = f"#buffer = #ttnn.buffer_type<{buffer_type}>"

    # Memory layout (interleaved or sharded variants).
    # Uses 2D memref [tiles_y, tiles_x] to match Python DSL approach.
    layout_type = config.memory_layout.value  # "interleaved", "height_sharded", etc.
    layout_attr = f"#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{rows}x{cols}x!ttcore.tile<32x32, {dtype_str}>, #buffer>, <{layout_type}>>"

    return f"""
{buffer_attr}
{layout_attr}
#map = affine_map<(d0, d1) -> (d0, d1)>
"""
