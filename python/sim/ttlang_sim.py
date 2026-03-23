# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TT-Lang Simulator launcher (ttlang-sim).

Runs tt-lang kernels written for the compiler on the simulator backend
without requiring any code changes to the kernel files.

Usage:
    ttlang-sim examples/eltwise_add.py
    ttlang-sim examples/single_node_matmul.py --show-stats --grid 4,4
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


def execute_script_with_simulator(
    script_path: Path,
    capture_output: bool = False,
    argv: list[str] | None = None,
) -> tuple[int, str]:
    """
    Execute a script with simulator backend.

    Args:
        script_path: Path to the Python file to execute
        capture_output: If True, capture and return stdout/stderr; if False, print directly
        argv: Command-line arguments to pass to the script (for sys.argv)

    Returns:
        (exit_code, output) tuple where exit_code is 0 on success, 1 on error,
        and output is captured text if capture_output=True, empty string otherwise
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr

    if argv is None:
        argv = []

    # Set up sys.argv for the executed script
    original_argv = sys.argv
    sys.argv = [str(script_path)] + argv

    output_capture = io.StringIO() if capture_output else None
    exec_globals: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__builtins__": __builtins__,
    }

    try:
        code = compile(script_path.read_text(), str(script_path), "exec")

        if capture_output:
            assert output_capture is not None  # Guaranteed by capture_output=True
            with redirect_stdout(output_capture), redirect_stderr(output_capture):  # type: ignore
                exit_code = _execute_code(
                    code, exec_globals, script_path, output_capture
                )
        else:
            exit_code = _execute_code(code, exec_globals, script_path, None)

        output = output_capture.getvalue() if capture_output and output_capture else ""
        return exit_code, output

    finally:
        sys.argv = original_argv


def _execute_code(
    code: Any,
    exec_globals: dict[str, Any],
    script_path: Path,
    error_output: Any,
) -> int:
    """Execute compiled code and return exit code."""
    import traceback

    try:
        exec(code, exec_globals)
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else int(bool(e.code))
    except RuntimeError as e:
        # RuntimeError with __cause__ is from greenlet scheduler (including deadlocks)
        if e.__cause__ is not None:
            if error_output:
                traceback.print_exception(
                    type(e), e, e.__traceback__, file=error_output
                )
            else:
                traceback.print_exception(type(e), e, e.__traceback__)
            return 1
        else:
            if error_output:
                print(f"\nError executing {script_path.name}:", file=error_output)
                traceback.print_exception(
                    type(e), e, e.__traceback__, file=error_output
                )
            else:
                print(f"\nError executing {script_path.name}:", file=sys.stderr)
                _print_filtered_traceback(e, script_path)
            return 1
    except Exception as e:
        if error_output:
            traceback.print_exception(type(e), e, e.__traceback__, file=error_output)
        else:
            print(f"\nError executing {script_path.name}:", file=sys.stderr)
            raise
        return 1


def run_file(filepath: str, argv: list[str]) -> None:
    """
    Execute a kernel file with simulator backend (CLI wrapper).

    Args:
        filepath: Path to the Python file to execute
        argv: Command-line arguments to pass to the script
    """
    file_path = Path(filepath)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    exit_code, _ = execute_script_with_simulator(
        file_path, capture_output=False, argv=argv
    )
    if exit_code != 0:
        sys.exit(exit_code)


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
        "  ttlang-sim examples/single_node_matmul.py --show-stats\n"
        "  ttlang-sim examples/tutorial/multinode.py --grid 4,4",
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
        "--max-dfbs",
        type=int,
        metavar="N",
        dest="max_dfbs",
        help="Maximum number of DataflowBuffers (CBs) per core (default: 32)",
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

    # Configure max_dfbs limit if specified
    if args.max_dfbs is not None:
        try:
            from .program import set_max_dfbs

            set_max_dfbs(args.max_dfbs)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

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
