# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Circular buffer operations for inter-thread communication."""

from ttmlir.dialects import d2m
from ttmlir.ir import *

from ._src.d2m_ast import syntax


@syntax("!d2m.cb")
class CircularBuffer:
    """
    Circular buffer for inter-thread communication.

    Circular buffers provide producer-consumer synchronization between
    compute and data movement threads.
    """

    def wait(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Wait for data from the circular buffer (consumer acquire).

        Use in consumer threads to acquire data. Must be followed by pop()
        to signal consumption is complete.

        Returns:
            TensorBlock: The acquired data.

        Example:
            shard = cb.wait()
            result = compute(shard)
            cb.pop()
        """
        return d2m.wait(d2m.ir.CBType.cast(ast_self.type).get_underlying(), ast_self)

    def pop(ast_self: "CircularBuffer") -> None:
        """
        Signal that data has been consumed (consumer release).

        Use in consumer threads after wait() to signal that data has been
        consumed and space is available for producers.

        Example:
            shard = cb.wait()
            result = compute(shard)
            cb.pop()  # Signal consumption complete
        """
        if not hasattr(d2m, "pop"):
            raise AttributeError(
                "d2m.pop is not available. Please ensure your tt-mlir build "
                "includes the d2m.pop operation in the Python bindings."
            )
        d2m.pop(ast_self)

    def reserve(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Reserve space in the circular buffer (producer acquire).

        Use in producer threads to acquire space for writing. Must be followed
        by push() to signal data is ready.

        Returns:
            TensorBlock: The reserved space.

        Example:
            shard = cb.reserve()
            dma(stream[idx], shard).wait()
            cb.push()
        """
        return d2m.reserve(d2m.ir.CBType.cast(ast_self.type).get_underlying(), ast_self)

    def push(ast_self: "CircularBuffer") -> None:
        """
        Signal that data is ready in the circular buffer (producer release).

        Use in producer threads after reserve() to signal that data has been
        written and is ready for consumers.

        Example:
            shard = cb.reserve()
            dma(stream[idx], shard).wait()
            cb.push()  # Signal data ready
        """
        if not hasattr(d2m, "push"):
            raise AttributeError(
                "d2m.push is not available. Please ensure your tt-mlir build "
                "includes the d2m.push operation in the Python bindings."
            )
        d2m.push(ast_self)
