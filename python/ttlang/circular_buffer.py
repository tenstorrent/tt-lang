# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Circular buffer operations for inter-thread communication."""

from ttmlir.ir import *
from ttmlir.dialects import arith

from ttlang.dialects import ttl

from ._src.d2m_ast import syntax


def _create_num_pages_constant():
    """Create i32 constant 1 for num_pages argument to CB operations."""
    return arith.ConstantOp(IntegerType.get_signless(32), IntegerAttr.get(IntegerType.get_signless(32), 1))


def _get_cb_underlying_type(cb_type):
    """Get the underlying tensor type from a CB type.

    CB type format: !ttl.cb<[shape], element_type, buffer_factor>
    Returns: tensor<shapexelement_type>
    """
    type_str = str(cb_type)

    # Extract shape between [ and ]
    shape_start = type_str.find("[") + 1
    shape_end = type_str.find("]")
    if shape_start <= 0 or shape_end < 0:
        raise ValueError(f"Invalid CB type format: {type_str}")
    shape_str = type_str[shape_start:shape_end]
    shape = [int(x.strip()) for x in shape_str.split(",")]

    # Element type is between "], " and the last comma (before buffer_factor)
    elem_start = shape_end + 3  # Skip "], "
    elem_end = type_str.rfind(",")
    if elem_end < elem_start:
        raise ValueError(f"Invalid CB type format: {type_str}")
    elem_str = type_str[elem_start:elem_end].strip()

    # Build tensor type string
    shape_dims = "x".join(str(s) for s in shape)
    tensor_type_str = f"tensor<{shape_dims}x{elem_str}>"

    return Type.parse(tensor_type_str, cb_type.context)


@syntax("!ttl.cb")
class CircularBuffer:
    """
    Circular buffer for inter-thread communication.

    Circular buffers provide producer-consumer synchronization between
    compute and data movement threads. Uses TTL dialect operations.
    """

    def wait(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Wait for data from the circular buffer (consumer acquire).

        Use in consumer threads to acquire data. Must be followed by pop()
        to signal consumption is complete.

        Returns:
            TensorBlock: The acquired data.
        """
        num_pages = _create_num_pages_constant()
        underlying = _get_cb_underlying_type(ast_self.type)
        return ttl.cb_wait(underlying, ast_self, num_pages)

    def pop(ast_self: "CircularBuffer") -> None:
        """
        Signal that data has been consumed (consumer release).

        Use in consumer threads after wait() to signal that data has been
        consumed and space is available for producers.
        """
        num_pages = _create_num_pages_constant()
        ttl.cb_pop(ast_self, num_pages)

    def reserve(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Reserve space in the circular buffer (producer acquire).

        Use in producer threads to acquire space for writing. Must be followed
        by push() to signal data is ready.

        Returns:
            TensorBlock: The reserved space.
        """
        num_pages = _create_num_pages_constant()
        underlying = _get_cb_underlying_type(ast_self.type)
        return ttl.cb_reserve(underlying, ast_self, num_pages)

    def push(ast_self: "CircularBuffer") -> None:
        """
        Signal that data is ready in the circular buffer (producer release).

        Use in producer threads after reserve() to signal that data has been
        written and is ready for consumers.
        """
        num_pages = _create_num_pages_constant()
        ttl.cb_push(ast_self, num_pages)
