# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Debug printing support for TT-Lang simulator.

Implements the debug printing functionality specified in TTLang spec section 10.2,
allowing users to print tensors, blocks, and dataflow buffers with custom formatting.
"""

from typing import Any, Union
import builtins

from .context import get_context
from .ttnnsim import Tensor
from .dfb import Block, DataflowBuffer
from .blockstate import AccessState, BlockAcquisition, ThreadType
from .diagnostics import warn_once_per_location
from .greenlet_scheduler import get_current_core_id

# Type alias for TT-Lang printable objects
TTLangObject = Union[Tensor, Block, DataflowBuffer]


def _format_tensor(tensor: Tensor, num_pages: int = 1) -> str:
    """Format a Tensor for printing.

    Args:
        tensor: The tensor to format
        num_pages: Number of pages to print (default 1)

    Returns:
        Formatted string representation
    """
    lines = [f"<Tensor shape={tensor.shape} dtype={tensor.dtype}>"]

    # Convert to torch tensor
    data = tensor.to_torch()

    # Calculate tiles per page (assuming TILE_SIZE=32, one tile = 32x32 elements)
    # For now, just print the raw data with page info
    lines.append(f"  Printing first {num_pages} page(s):")
    lines.append(f"  Data (first elements): {data.flatten()[:min(64, data.numel())]}")

    return "\n".join(lines)


def _warn_block_in_illegal_state(block: Block, message: str) -> None:
    """Issue a warning that a block is in an illegal state for printing.

    Tracks which cores hit each source location and only prints once per location.

    Args:
        block: The block being printed
        message: The warning message to display
    """
    warn_once_per_location(
        get_context().warnings.block_print_warnings,
        message,
        get_current_core_id(),
    )


def _format_block(block: Block) -> str:
    """Format a Block for printing.

    Args:
        block: The block to format

    Returns:
        Formatted string representation, or a warning message if block is in illegal state
    """
    # Check if block is in an illegal state for printing
    # Illegal states:
    # 1. DM thread + reserve + (MW or NAW)
    # 2. DM thread + wait + NAW
    if not block.is_temporary:
        if block.thread_type == ThreadType.DM:
            if block.acquisition == BlockAcquisition.RESERVE:
                if block.access_state in (AccessState.MW, AccessState.NAW):
                    warning_msg = (
                        f"Block in {block.access_state.name} state cannot be read. "
                        f"Block is in illegal state for printing "
                        f"(Acquisition=RESERVE, Thread=DM, Access={block.access_state.name})"
                    )
                    _warn_block_in_illegal_state(block, warning_msg)
                    return f"<Block shape={block.shape} [WARNING: Cannot read - in {block.access_state.name} state]>"
            elif block.acquisition == BlockAcquisition.WAIT:
                if block.access_state == AccessState.NAW:
                    warning_msg = (
                        f"Block in {block.access_state.name} state cannot be read. "
                        f"Block is in illegal state for printing "
                        f"(Acquisition=WAIT, Thread=DM, Access={block.access_state.name})"
                    )
                    _warn_block_in_illegal_state(block, warning_msg)
                    return f"<Block shape={block.shape} [WARNING: Cannot read - in {block.access_state.name} state]>"

    lines = [f"<Block shape={block.shape}>"]

    # Print block content - use to_tensor() method to get backing tensor
    data = block.to_tensor().to_torch()
    lines.append(f"  Data shape: {data.shape}")
    lines.append(f"  Data (first elements): {data.flatten()[:min(64, data.numel())]}")

    return "\n".join(lines)


def _format_dfb(dfb: DataflowBuffer) -> str:
    """Format a DataflowBuffer for printing.

    Args:
        dfb: The dataflow buffer to format

    Returns:
        Formatted string representation with metadata
    """
    lines = [f"<DataflowBuffer name='{getattr(dfb, '_name', 'unnamed')}'>"]
    lines.append(f"  shape: {dfb.shape}")
    lines.append(f"  likeness_tensor: {dfb.likeness_tensor}")
    lines.append(f"  block_count: {dfb.block_count}")
    lines.append(f"  capacity: {dfb.capacity_tiles} tiles")
    lines.append(f"  rd_ptr (head): {dfb.head}")
    lines.append(f"  visible: {dfb.visible} operations")
    lines.append(f"  reserved: {dfb.reserved} operations")
    lines.append(f"  free: {dfb.free} operations")

    return "\n".join(lines)


def ttlang_print(*args: Any, **kwargs: Any) -> None:
    """Custom print function for TT-Lang kernels.

    Supports printing:
    - String constants and scalar variables (forwarded to builtin print)
    - ttnn.Tensor objects with optional num_pages attribute
    - ttl.Block objects (prints content)
    - ttl.DataflowBuffer objects (prints metadata)

    Limitation: Only one TT-Lang object can be printed per call, along with
    any number of strings and scalars.

    Examples:
        print("C: ", C, num_pages=2)
        print("it=", it, " mt=", mt, "c_blk: ", c_blk)
        print(c_dfb)
    """
    # Separate TT-Lang objects from regular args
    ttlang_objects: list[TTLangObject] = []
    regular_args: list[Any] = []

    for arg in args:
        if isinstance(arg, (Tensor, Block, DataflowBuffer)):
            ttlang_objects.append(arg)
        else:
            regular_args.append(arg)

    # If no TT-Lang objects, just use builtin print
    if not ttlang_objects:
        builtins.print(*regular_args, **kwargs)
        return

    # Enforce spec limitation: only one TT-Lang object per print
    if len(ttlang_objects) > 1:
        raise ValueError(
            "print() can only print one TT-Lang object at a time "
            f"(got {len(ttlang_objects)}). Use separate print() calls."
        )

    obj = ttlang_objects[0]

    # Extract TT-Lang specific attributes
    num_pages = kwargs.pop("num_pages", 1)

    # Format the TT-Lang object
    if isinstance(obj, Tensor):
        formatted = _format_tensor(obj, num_pages=num_pages)
    elif isinstance(obj, Block):
        formatted = _format_block(obj)
    else:
        # Must be DataflowBuffer (only remaining type in TTLangObject)
        formatted = _format_dfb(obj)

    # Print regular args followed by formatted object
    if regular_args:
        builtins.print(*regular_args, end="")
        builtins.print(formatted, **kwargs)
    else:
        builtins.print(formatted, **kwargs)
