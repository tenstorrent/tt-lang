# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Statistics collection for simulator operations.

Tracks tensor read/write operations and provides summary reporting.
"""

import math
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .ttnnsim import Tensor

from .constants import TILE_SHAPE
from .typedefs import AnyPipe, CoreCoord, CoreRange

# Statistics collection state
_stats_enabled = False
_stats_by_name: dict[str, dict[str, int]] = defaultdict(
    lambda: {"reads": 0, "writes": 0, "tiles_read": 0, "tiles_written": 0}
)
_pipe_stats_by_name: dict[str, dict[str, int]] = defaultdict(
    lambda: {"reads": 0, "writes": 0, "tiles_read": 0, "tiles_written": 0}
)
_cb_stats_by_name: dict[str, dict[str, int]] = defaultdict(
    lambda: {"reserves": 0, "waits": 0, "tiles_reserved": 0, "tiles_waited": 0}
)
_stats_lock = threading.Lock()
_cb_name_counter = 0


def enable_stats() -> None:
    """Enable statistics collection."""
    global _stats_enabled
    _stats_enabled = True


def disable_stats() -> None:
    """Disable statistics collection."""
    global _stats_enabled
    _stats_enabled = False


def is_stats_enabled() -> bool:
    """Check if statistics collection is enabled."""
    return _stats_enabled


def reset_stats() -> None:
    """Reset all collected statistics."""
    global _stats_by_name, _pipe_stats_by_name, _cb_stats_by_name, _cb_name_counter
    with _stats_lock:
        _stats_by_name.clear()
        _pipe_stats_by_name.clear()
        _cb_stats_by_name.clear()
        _cb_name_counter = 0


def register_tensor_name(tensor: "Tensor", name: str) -> None:
    """Register a name for a tensor.

    Args:
        tensor: The tensor to name
        name: The name to associate with this tensor
    """
    # Set as attribute so it can be propagated to slices
    tensor._name = name  # type: ignore


def _get_tensor_name(tensor: "Tensor") -> str:
    """Get the name of a tensor, using _name attribute or generating one.

    Args:
        tensor: The tensor to get a name for

    Returns:
        The tensor's name
    """
    name = getattr(tensor, "_name", None)
    if name:
        return name
    # Generate a name based on object ID
    return f"tensor_{id(tensor) % 10000}"


def _calculate_tensor_tiles(tensor: "Tensor") -> int:
    """Calculate the number of tiles in a tensor.

    Args:
        tensor: The tensor to calculate tiles for

    Returns:
        Number of tiles (rounded up for partial tiles)
    """
    if len(tensor.shape) != 2:
        # For non-2D tensors, return 0 or calculate differently
        return 0

    # Calculate tiles in each dimension (round up for partial tiles)
    tiles_h = math.ceil(tensor.shape[0] / TILE_SHAPE[0])
    tiles_w = math.ceil(tensor.shape[1] / TILE_SHAPE[1])
    return tiles_h * tiles_w


def record_tensor_read(tensor: "Tensor") -> None:
    """Record a read operation on a tensor.

    Args:
        tensor: The tensor being read from
    """
    if not _stats_enabled:
        return

    name = _get_tensor_name(tensor)
    num_tiles = _calculate_tensor_tiles(tensor)

    with _stats_lock:
        _stats_by_name[name]["reads"] += 1
        _stats_by_name[name]["tiles_read"] += num_tiles


def record_tensor_write(tensor: "Tensor") -> None:
    """Record a write operation on a tensor.

    Args:
        tensor: The tensor being written to
    """
    if not _stats_enabled:
        return

    name = _get_tensor_name(tensor)
    num_tiles = _calculate_tensor_tiles(tensor)

    with _stats_lock:
        _stats_by_name[name]["writes"] += 1
        _stats_by_name[name]["tiles_written"] += num_tiles


def _get_pipe_name(pipe: AnyPipe) -> str:
    """Get a name for a pipe based on its source and destination.

    Args:
        pipe: The pipe to get a name for

    Returns:
        The pipe's name
    """
    # Format: pipe_src_dst
    src: CoreCoord = pipe.src_core
    dst: Union[CoreCoord, CoreRange] = pipe.dst_core_range

    # Format source
    match src:
        case tuple():
            src_str = f"({','.join(str(x) for x in src)})"
        case _:
            src_str = str(src)

    # Format destination
    match dst:
        case tuple():
            # Format as range notation, with slices shown as start:stop
            parts: list[str] = []
            for x in dst:
                match x:
                    case slice():
                        start = x.start if x.start is not None else 0
                        stop = x.stop if x.stop is not None else "?"
                        parts.append(f"{start}:{stop}")
                    case _:
                        parts.append(str(x))
            dst_str = f"({','.join(parts)})"
        case _:
            dst_str = str(dst)

    return f"pipe_{src_str}_to_{dst_str}"


def _calculate_block_tiles(block_data: "list[Tensor]") -> int:
    """Calculate the total number of tiles in a block's data.

    Args:
        block_data: List of tensors in the block

    Returns:
        Total number of tiles
    """
    return sum(_calculate_tensor_tiles(tensor) for tensor in block_data)


def record_pipe_write(pipe: AnyPipe, block_data: "list[Tensor]") -> None:
    """Record a write (send) operation on a pipe.

    Args:
        pipe: The pipe being written to
        block_data: List of tensors being sent through the pipe
    """
    if not _stats_enabled:
        return

    name = _get_pipe_name(pipe)
    num_tiles = _calculate_block_tiles(block_data)

    with _stats_lock:
        _pipe_stats_by_name[name]["writes"] += 1
        _pipe_stats_by_name[name]["tiles_written"] += num_tiles


def record_pipe_read(pipe: AnyPipe, block_data: "list[Tensor]") -> None:
    """Record a read (receive) operation on a pipe.

    Args:
        pipe: The pipe being read from
        block_data: List of tensors being received from the pipe
    """
    if not _stats_enabled:
        return

    name = _get_pipe_name(pipe)
    num_tiles = _calculate_block_tiles(block_data)

    with _stats_lock:
        _pipe_stats_by_name[name]["reads"] += 1
        _pipe_stats_by_name[name]["tiles_read"] += num_tiles


def register_cb_name(cb: Any, name: str) -> None:
    """Register a name for a circular buffer.

    Args:
        cb: The CB to name
        name: The name to associate with this CB
    """
    # Set as attribute so it persists
    cb._stats_name = name  # type: ignore


def _get_cb_name(cb: Any) -> str:
    """Get the name of a circular buffer.

    Args:
        cb: The CB to get a name for

    Returns:
        The CB's name
    """
    # Check if CB has a registered name
    name = getattr(cb, "_stats_name", None)
    if name:
        return name

    # Generate a name based on CB ID if available
    cb_id = getattr(cb, "_cb_id", None)
    if cb_id is not None:
        return f"cb_{cb_id}"

    # Fall back to generating a unique name
    global _cb_name_counter
    with _stats_lock:
        _cb_name_counter += 1
        counter = _cb_name_counter
    return f"cb_unnamed_{counter}"


def record_cb_reserve(cb: Any, num_tiles: int) -> None:
    """Record a reserve (write allocation) operation on a CB.

    Args:
        cb: The CB being reserved from
        num_tiles: Number of tiles being reserved
    """
    if not _stats_enabled:
        return

    name = _get_cb_name(cb)

    with _stats_lock:
        _cb_stats_by_name[name]["reserves"] += 1
        _cb_stats_by_name[name]["tiles_reserved"] += num_tiles


def record_cb_wait(cb: Any, num_tiles: int) -> None:
    """Record a wait (read) operation on a CB.

    Args:
        cb: The CB being waited on
        num_tiles: Number of tiles being waited for
    """
    if not _stats_enabled:
        return

    name = _get_cb_name(cb)

    with _stats_lock:
        _cb_stats_by_name[name]["waits"] += 1
        _cb_stats_by_name[name]["tiles_waited"] += num_tiles


def print_stats() -> None:
    """Print collected tensor, pipe, and CB statistics."""
    # Take a snapshot of the statistics under lock
    with _stats_lock:
        has_tensor_stats = bool(_stats_by_name)
        has_pipe_stats = bool(_pipe_stats_by_name)
        has_cb_stats = bool(_cb_stats_by_name)

        if not has_tensor_stats and not has_pipe_stats and not has_cb_stats:
            print("\nNo statistics collected.")
            return

        # Copy data to avoid holding lock during printing
        tensor_stats_copy = dict(_stats_by_name)
        pipe_stats_copy = dict(_pipe_stats_by_name)
        cb_stats_copy = dict(_cb_stats_by_name)

    # Print tensor statistics
    if has_tensor_stats:
        print("\n" + "=" * 64)
        print("Tensor Access Statistics")
        print("=" * 64)

        # Sort by name for consistent output
        sorted_names = sorted(tensor_stats_copy.items())

        print(
            f"{'Tensor':<20} {'Reads':>8} {'Writes':>8} {'Tiles Read':>12} {'Tiles Written':>12}"
        )
        print("-" * 64)

        total_reads = 0
        total_writes = 0
        total_tiles_read = 0
        total_tiles_written = 0

        for name, stats in sorted_names:
            reads = stats["reads"]
            writes = stats["writes"]
            tiles_read = stats["tiles_read"]
            tiles_written = stats["tiles_written"]

            total_reads += reads
            total_writes += writes
            total_tiles_read += tiles_read
            total_tiles_written += tiles_written

            print(
                f"{name:<20} {reads:>8} {writes:>8} {tiles_read:>12} {tiles_written:>12}"
            )

        print("-" * 64)
        print(
            f"{'TOTAL':<20} {total_reads:>8} {total_writes:>8} {total_tiles_read:>12} {total_tiles_written:>12}"
        )
        print("=" * 64)

    # Print pipe statistics
    if has_pipe_stats:
        print("\n" + "=" * 74)
        print("Pipe Transfer Statistics")
        print("=" * 74)

        # Sort by name for consistent output
        sorted_pipes = sorted(pipe_stats_copy.items())

        print(
            f"{'Pipe':<30} {'Reads':>8} {'Writes':>8} {'Tiles Read':>12} {'Tiles Written':>12}"
        )
        print("-" * 74)

        total_reads = 0
        total_writes = 0
        total_tiles_read = 0
        total_tiles_written = 0

        for name, stats in sorted_pipes:
            reads = stats["reads"]
            writes = stats["writes"]
            tiles_read = stats["tiles_read"]
            tiles_written = stats["tiles_written"]

            total_reads += reads
            total_writes += writes
            total_tiles_read += tiles_read
            total_tiles_written += tiles_written

            print(
                f"{name:<30} {reads:>8} {writes:>8} {tiles_read:>12} {tiles_written:>12}"
            )

        print("-" * 74)
        print(
            f"{'TOTAL':<30} {total_reads:>8} {total_writes:>8} {total_tiles_read:>12} {total_tiles_written:>12}"
        )
        print("=" * 74)

    # Print CB statistics
    if has_cb_stats:
        print("\n" + "=" * 74)
        print("Circular Buffer Statistics")
        print("=" * 74)

        # Sort by name for consistent output
        sorted_cbs = sorted(cb_stats_copy.items())

        print(
            f"{'CB':<20} {'Reserves':>10} {'Waits':>10} {'Tiles Reserved':>16} {'Tiles Waited':>14}"
        )
        print("-" * 74)

        total_reserves = 0
        total_waits = 0
        total_tiles_reserved = 0
        total_tiles_waited = 0

        for name, stats in sorted_cbs:
            reserves = stats["reserves"]
            waits = stats["waits"]
            tiles_reserved = stats["tiles_reserved"]
            tiles_waited = stats["tiles_waited"]

            total_reserves += reserves
            total_waits += waits
            total_tiles_reserved += tiles_reserved
            total_tiles_waited += tiles_waited

            print(
                f"{name:<20} {reserves:>10} {waits:>10} {tiles_reserved:>16} {tiles_waited:>14}"
            )

        print("-" * 74)
        print(
            f"{'TOTAL':<20} {total_reserves:>10} {total_waits:>10} {total_tiles_reserved:>16} {total_tiles_waited:>14}"
        )
        print("=" * 74)
