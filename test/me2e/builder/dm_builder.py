# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Data movement thread builder for E2E tests.

Provides builders for reader and writer threads that handle NOC data movement
between DRAM and circular buffers. These are minimal wrappers around the
base StringBasedThreadBuilder.
"""

from typing import List

from .thread_builder import StringBasedThreadBuilder


class DMThreadBuilder(StringBasedThreadBuilder):
    """
    Data movement thread builder - readers and writers.

    Generates MLIR for NOC threads that move data between DRAM and CBs.
    Uses string-based generation due to DRAM tensor layout attributes.
    """

    def build_reader(self, num_inputs: int) -> str:
        """
        Build reader thread: DRAM tensors -> CBs 0..num_inputs-1.

        Args:
            num_inputs: Number of input tensors (1 for unary, 2 for binary).

        Returns:
            MLIR string for the reader function.
        """
        # Generate function signature.
        name = "reader_binary" if num_inputs == 2 else "reader_unary"
        args = ", ".join(
            [f"%in{i}: {self.dram_tensor_type_str}" for i in range(num_inputs)]
        )
        crta = ", ".join([f"{i} : i32" for i in range(num_inputs)])

        # Generate CB bindings.
        cb_binds = "\n".join(
            [
                f"  %cb{i} = ttl.bind_cb {{cb_index = {i}, buffer_factor = {self._buffer_factor}}} : {self.cb_type_str}"
                for i in range(num_inputs)
            ]
        )

        # Generate loop structure.
        loop_start, loop_end, row_idx, col_idx = self._generate_loop_start()
        indent = "      " if self._num_iterations > 1 else "  "

        # Generate read operations for each input.
        read_ops = "\n\n".join(
            [
                self._read_to_cb_str(
                    tensor_var=f"%in{i}",
                    cb_var=f"%cb{i}",
                    row_idx=row_idx,
                    col_idx=col_idx,
                    prefix=f"in{i}",
                    indent=indent,
                )
                for i in range(num_inputs)
            ]
        )

        return f"""
// Reader data movement thread: reads {num_inputs} tensor(s) from DRAM into CBs.
func.func @{name}({args})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [{crta}], ttl.kernel_thread = #ttkernel.thread<noc>}} {{
{cb_binds}
{loop_start}

{read_ops}
{loop_end}
  func.return
}}
"""

    def build_writer(self, output_cbs: List[int]) -> str:
        """
        Build writer thread: CBs -> DRAM tensors.

        Args:
            output_cbs: List of output CB indices.

        Returns:
            MLIR string for the writer function.
        """
        num_outputs = len(output_cbs)

        # Generate function signature.
        args = ", ".join(
            [f"%out{i}: {self.dram_tensor_type_str}" for i in range(num_outputs)]
        )
        crta = ", ".join([f"{cb} : i32" for cb in output_cbs])

        # Generate CB bindings.
        cb_binds = "\n".join(
            [
                f"  %cb_out{i} = ttl.bind_cb {{cb_index = {output_cbs[i]}, buffer_factor = {self._buffer_factor}}} : {self.cb_type_str}"
                for i in range(num_outputs)
            ]
        )

        # Generate loop structure.
        loop_start, loop_end, row_idx, col_idx = self._generate_loop_start()
        indent = "      " if self._num_iterations > 1 else "  "

        # Generate write operations for each output.
        write_ops = "\n\n".join(
            [
                self._write_from_cb_str(
                    cb_var=f"%cb_out{i}",
                    tensor_var=f"%out{i}",
                    row_idx=row_idx,
                    col_idx=col_idx,
                    prefix=f"out{i}",
                    indent=indent,
                )
                for i in range(num_outputs)
            ]
        )

        return f"""
// Writer data movement thread: writes {num_outputs} output(s) from CBs to DRAM.
func.func @writer({args})
    attributes {{ttl.base_cta_index = 3 : i32, ttl.crta_indices = [{crta}], ttl.kernel_thread = #ttkernel.thread<noc>}} {{
{cb_binds}
{loop_start}

{write_ops}
{loop_end}
  func.return
}}
"""
