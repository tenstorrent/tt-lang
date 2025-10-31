# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Circular buffer operations for inter-thread communication."""

from ttmlir.dialects import d2m

from ._src.d2m_ast import syntax


@syntax("!d2m.cb")
class CircularBuffer:
    """
    Circular buffer for inter-thread communication.

    Circular buffers provide producer-consumer synchronization between
    compute and data movement threads.
    """

    def pop(ast_self: "CircularBuffer") -> "TensorBlock":
        """Wait for and consume data from the circular buffer."""
        return d2m.wait(d2m.ir.CBType.cast(ast_self.type).getUnderlying(), ast_self)

    def reserve(ast_self: "CircularBuffer") -> "TensorBlock":
        """Reserve space in the circular buffer for writing."""
        return d2m.reserve(d2m.ir.CBType.cast(ast_self.type).getUnderlying(), ast_self)
