# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Statistics collection for simulator operations.

Tracks tensor read/write operations and provides summary reporting.
"""

from typing import Any, Union

from .ttnnsim import Tensor

from .context import get_context
from .ttnnsim import tile_count_from_tensor
from .pipe import AnyPipe
from .typedefs import CoreCoord, CoreRange


def enable_stats() -> None:
    """Enable statistics collection."""
    get_context().stats.enabled = True


def disable_stats() -> None:
    """Disable statistics collection."""
    get_context().stats.enabled = False


def is_stats_enabled() -> bool:
    """Check if statistics collection is enabled."""
    return get_context().stats.enabled


def reset_stats() -> None:
    """Reset all collected statistics."""
    ctx = get_context()
    ctx.stats.stats_by_name.clear()
    ctx.stats.pipe_stats_by_name.clear()
    ctx.stats.dfb_stats_by_name.clear()
    ctx.stats.dfb_name_counter = 0


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
    """Calculate the number of tiles in a tensor."""
    return tile_count_from_tensor(tensor)


def record_tensor_read(tensor: "Tensor") -> None:
    """Record a read operation on a tensor.

    Args:
        tensor: The tensor being read from
    """
    ctx = get_context()
    if not ctx.stats.enabled:
        return

    name = _get_tensor_name(tensor)
    num_tiles = _calculate_tensor_tiles(tensor)

    ctx.stats.stats_by_name[name]["reads"] += 1
    ctx.stats.stats_by_name[name]["tiles_read"] += num_tiles


def record_tensor_write(tensor: "Tensor") -> None:
    """Record a write operation on a tensor.

    Args:
        tensor: The tensor being written to
    """
    ctx = get_context()
    if not ctx.stats.enabled:
        return

    name = _get_tensor_name(tensor)
    num_tiles = _calculate_tensor_tiles(tensor)

    ctx.stats.stats_by_name[name]["writes"] += 1
    ctx.stats.stats_by_name[name]["tiles_written"] += num_tiles


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


def record_pipe_write(pipe: AnyPipe, block_data: "Tensor") -> None:
    """Record a write (send) operation on a pipe.

    Args:
        pipe: The pipe being written to
        block_data: Tensor being sent through the pipe
    """
    ctx = get_context()
    if not ctx.stats.enabled:
        return

    name = _get_pipe_name(pipe)
    num_tiles = _calculate_tensor_tiles(block_data)

    ctx.stats.pipe_stats_by_name[name]["writes"] += 1
    ctx.stats.pipe_stats_by_name[name]["tiles_written"] += num_tiles


def record_pipe_read(pipe: AnyPipe, block_data: "Tensor") -> None:
    """Record a read (receive) operation on a pipe.

    Args:
        pipe: The pipe being read from
        block_data: Tensor being received from the pipe
    """
    ctx = get_context()
    if not ctx.stats.enabled:
        return

    name = _get_pipe_name(pipe)
    num_tiles = _calculate_tensor_tiles(block_data)

    ctx.stats.pipe_stats_by_name[name]["reads"] += 1
    ctx.stats.pipe_stats_by_name[name]["tiles_read"] += num_tiles


def register_dfb_name(dfb: Any, name: str) -> None:
    """Register a name for a dataflow buffer.

    Args:
        dfb: The DFB to name
        name: The name to associate with this DFB
    """
    # Set as attribute so it persists
    dfb._stats_name = name  # type: ignore


def _get_dfb_name(dfb: Any) -> str:
    """Get the name of a dataflow buffer.

    Args:
        dfb: The DFB to get a name for

    Returns:
        The DFB's name
    """
    # Check if DFB has a registered name
    name = getattr(dfb, "_stats_name", None)
    if name:
        return name

    # Generate a name based on DFB ID if available
    dfb_id = getattr(dfb, "_dfb_id", None)
    if dfb_id is not None:
        return f"dfb_{dfb_id}"

    # Fall back to generating a unique name
    ctx = get_context()
    ctx.stats.dfb_name_counter += 1
    return f"dfb_unnamed_{ctx.stats.dfb_name_counter}"


def record_dfb_reserve(dfb: Any, num_tiles: int) -> None:
    """Record a reserve (write allocation) operation on a DFB.

    Args:
        dfb: The DFB being reserved from
        num_tiles: Number of tiles being reserved
    """
    ctx = get_context()
    if not ctx.stats.enabled:
        return

    name = _get_dfb_name(dfb)

    ctx.stats.dfb_stats_by_name[name]["reserves"] += 1
    ctx.stats.dfb_stats_by_name[name]["tiles_reserved"] += num_tiles


def record_dfb_wait(dfb: Any, num_tiles: int) -> None:
    """Record a wait (read) operation on a DFB.

    Args:
        dfb: The DFB being waited on
        num_tiles: Number of tiles being waited for
    """
    ctx = get_context()
    if not ctx.stats.enabled:
        return

    name = _get_dfb_name(dfb)

    ctx.stats.dfb_stats_by_name[name]["waits"] += 1
    ctx.stats.dfb_stats_by_name[name]["tiles_waited"] += num_tiles


def print_stats() -> None:
    """Print collected tensor, pipe, and DFB statistics."""
    ctx = get_context()
    has_tensor_stats = bool(ctx.stats.stats_by_name)
    has_pipe_stats = bool(ctx.stats.pipe_stats_by_name)
    has_dfb_stats = bool(ctx.stats.dfb_stats_by_name)

    if not has_tensor_stats and not has_pipe_stats and not has_dfb_stats:
        print("\nNo statistics collected.")
        return

    tensor_stats_copy = dict(ctx.stats.stats_by_name)
    pipe_stats_copy = dict(ctx.stats.pipe_stats_by_name)
    dfb_stats_copy = dict(ctx.stats.dfb_stats_by_name)

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

    # Print DFB statistics
    if has_dfb_stats:
        print("\n" + "=" * 74)
        print("Dataflow Buffer Statistics")
        print("=" * 74)

        # Sort by name for consistent output
        sorted_dfbs = sorted(dfb_stats_copy.items())

        print(
            f"{'DFB':<20} {'Reserves':>10} {'Waits':>10} {'Tiles Reserved':>16} {'Tiles Waited':>14}"
        )
        print("-" * 74)

        total_reserves = 0
        total_waits = 0
        total_tiles_reserved = 0
        total_tiles_waited = 0

        for name, stats in sorted_dfbs:
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
