# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TT-Lang Simulator launcher (ttlang-sim).

Runs tt-lang kernels written for the compiler on the simulator backend
without requiring any code changes to the kernel files.

Usage:
    ttlang-sim examples/eltwise_add.py
    ttlang-sim examples/singlecore_matmul.py --show-stats --grid 4,4
"""

import sys
import argparse
from pathlib import Path
from typing import Any

from .stats import enable_stats, print_stats
from .kernel import set_default_grid
from .greenlet_scheduler import set_scheduler_algorithm


def setup_simulator_imports() -> None:
    """
    Inject simulator implementations into sys.modules so they shadow the compiler APIs.

    This allows kernel code written for the compiler to transparently use simulator
    implementations when run under ttlang-sim.
    """
    # Import simulator implementations
    from . import ttl, ttnn

    # Shadow compiler imports with simulator versions
    sys.modules["ttl"] = ttl  # type: ignore[assignment]
    sys.modules["ttnn"] = ttnn  # type: ignore[assignment]


def run_file(filepath: str, argv: list[str]) -> None:
    """
    Execute a kernel file with simulator backend.

    Args:
        filepath: Path to the Python file to execute
        argv: Command-line arguments to pass to the script
    """
    file_path = Path(filepath)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Add script's directory to sys.path to enable relative imports
    sys.path.insert(0, str(file_path.parent))

    # Set up sys.argv for the executed script
    sys.argv = [str(file_path)] + argv

    # Read and execute the file
    with open(file_path) as f:
        code = compile(f.read(), str(file_path), "exec")
        # Get the shadowed modules from sys.modules so they're available in exec
        exec_globals: dict[str, Any] = {
            "__name__": "__main__",
            "__file__": str(file_path),
            "__builtins__": __builtins__,
        }
        try:
            exec(code, exec_globals)
        except RuntimeError as e:
            # RuntimeError with __cause__ is from greenlet scheduler exception handling
            # (including deadlocks) and already has formatted error printed - suppress traceback
            if e.__cause__ is not None:
                sys.exit(1)
            print(f"\nError executing {file_path.name}:", file=sys.stderr)
            _print_filtered_traceback(e, file_path)
            sys.exit(1)
        except Exception:
            print(f"\nError executing {file_path.name}:", file=sys.stderr)
            raise


def _print_filtered_traceback(exc: Exception, user_file: Path) -> None:
    """Print traceback filtering out internal simulator frames.

    Only shows frames from user code, omitting internal simulator implementation
    details from python/sim/*.
    """
    import traceback
    from traceback import FrameSummary

    # Extract traceback entries
    tb_entries = traceback.extract_tb(exc.__traceback__)

    # Filter to only user code frames
    user_frames: list[FrameSummary] = []
    for frame in tb_entries:
        # Skip internal simulator frames
        if any(
            pattern in frame.filename
            for pattern in [
                "/python/sim/ttlang_sim.py",
                "/python/sim/kernel.py",
                "/python/sim/program.py",
                "/python/sim/greenlet_scheduler.py",
                "<frozen runpy>",
            ]
        ):
            continue
        user_frames.append(frame)

    # Print filtered traceback
    if user_frames:
        print("Traceback (most recent call last):", file=sys.stderr)
        for frame in user_frames:
            print(
                f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}',
                file=sys.stderr,
            )
            if frame.line:
                print(f"    {frame.line}", file=sys.stderr)

    # Print the exception message
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttlang-sim",
        description="Run tt-lang kernels on the simulator backend",
        epilog="Examples:\n"
        "  ttlang-sim examples/eltwise_add.py\n"
        "  ttlang-sim examples/singlecore_matmul.py --show-stats\n"
        "  ttlang-sim examples/tutorial/multicore.py --grid 4,4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="Python file (.py) to run",
    )

    parser.add_argument(
        "--grid",
        type=str,
        metavar="ROWS,COLS",
        help="Default grid size for kernels with grid='auto' (e.g., --grid 4,4). Defaults to 8,8",
    )

    parser.add_argument(
        "--show-stats",
        action="store_true",
        dest="show_stats",
        help="Print tensor read/write statistics after execution",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["greedy", "fair"],
        default="fair",
        dest="scheduler",
        help="Scheduler algorithm: 'greedy' (run until block) or 'fair' (least recently run)",
    )

    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the script",
    )

    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        sys.exit(1)

    # Set up simulator imports before running any code
    setup_simulator_imports()

    # Configure scheduler algorithm if specified
    if args.scheduler:

        set_scheduler_algorithm(args.scheduler)

    # Enable tensor statistics collection if requested
    if args.show_stats:

        enable_stats()

    # Configure default grid if specified
    if args.grid:
        try:
            parts = args.grid.split(",")
            if len(parts) != 2:
                raise ValueError("Grid must be specified as ROWS,COLS")
            rows, cols = int(parts[0].strip()), int(parts[1].strip())
            if rows <= 0 or cols <= 0:
                raise ValueError("Grid dimensions must be positive")

            set_default_grid((rows, cols))
        except ValueError as e:
            print(f"Error: Invalid grid specification: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the target
    try:
        if not args.target.endswith(".py"):
            print(f"Error: Target must be a .py file: {args.target}", file=sys.stderr)
            sys.exit(1)
        run_file(args.target, args.script_args)
    finally:
        # Print tensor statistics if enabled
        if args.show_stats:
            print_stats()


if __name__ == "__main__":
    main()
