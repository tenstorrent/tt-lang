# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Circular buffer operations for inter-thread communication."""

import re
from ttmlir.ir import *

from .dialects import ttl
from ._src.d2m_ast import syntax


def _get_cb_tensor_type(cb_val):
    """Extract the tensor type from a TTL CB type by parsing it."""
    # CB type is !ttl.cb<[shape], element_type, buffer_factor>
    cb_type_str = str(cb_val.type)
    # Parse: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    # Element type may contain commas inside <>, so match up to last comma
    match = re.match(r"!ttl\.cb<\[([^\]]+)\], (.+), (\d+)>$", cb_type_str)
    if match:
        shape_str = match.group(1)
        elem_str = match.group(2)
        shape = [int(s.strip()) for s in shape_str.split(",")]
        ctx = cb_val.type.context
        elem_type = Type.parse(elem_str, ctx)
        return RankedTensorType.get(shape, elem_type)
    raise ValueError(f"Could not parse CB type: {cb_type_str}")


@syntax("!ttl.cb")
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
            TensorBlock: The acquired data with CB association.

        Example:
            shard = cb.wait()
            result = compute(shard)
            cb.pop()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_wait(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

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
        ttl.cb_pop(ast_self)

    def reserve(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Reserve space in the circular buffer (producer acquire).

        Use in producer threads to acquire space for writing. Must be followed
        by push() to signal data is ready.

        Returns:
            TensorBlock: The reserved space with CB association.

        Example:
            cb.reserve()
            copy(stream[idx], cb).wait()
            cb.push()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_reserve(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

    def push(ast_self: "CircularBuffer") -> None:
        """
        Signal that data is ready in the circular buffer (producer release).

        Use in producer threads after reserve() to signal that data has been
        written and is ready for consumers.

        Example:
            shard = cb.reserve()
            copy(stream[idx], shard).wait()
            cb.push()  # Signal data ready
        """
        ttl.cb_push(ast_self)
