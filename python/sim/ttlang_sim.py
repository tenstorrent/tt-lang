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
import json
from pathlib import Path
from typing import Any

from .operation import set_default_grid
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


def _get_version() -> str:
    """Return the tt-lang version string for ttlang-sim --version."""
    try:
        from ttl.version import __version__  # type: ignore[import-untyped]

        return __version__  # type: ignore[return-value]
    except ImportError:
        return "unknown"


def _write_jsonl_trace(path: Path, events: list) -> None:
    """Write trace events to a JSON Lines file.

    Each line is a self-contained JSON object with flat fields:
    event, tick, kernel, and any event-specific data.

    Args:
        path: Output file path.
        events: List of TraceEvent objects to serialise.
    """
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            record: dict[str, Any] = {
                "tick": ev.tick,
                "kernel": ev.kernel,
                "event": ev.event,
            }
            record.update(ev.data)
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttlang-sim",
        description="Run tt-lang kernels on the simulator backend",
        epilog="Examples:\n"
        "  ttlang-sim examples/eltwise_add.py\n"
        "  ttlang-sim examples/elementwise-tutorial/step_3_multinode.py --grid 4,4\n"
        "  ttlang-sim examples/eltwise_add.py --max-l1 1572864",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ttlang-sim {_get_version()}",
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
        "--max-l1",
        type=int,
        metavar="BYTES",
        dest="max_l1",
        help="Maximum L1 memory per core in bytes; warns if total CB capacity exceeds this (default: 1336 KiB)",
    )

    parser.add_argument(
        "--num-devices",
        type=int,
        metavar="N",
        dest="num_devices",
        default=4,
        help="Number of simulated devices returned by GetNumAvailableDevices() (default: 4)",
    )

    parser.add_argument(
        "--promote-bf16",
        action="store_true",
        dest="promote_bf16",
        help="Redirect bfloat16 to float32 for faster computation on hardware without native bfloat16 support (e.g. Apple Silicon). Doubles tensor memory usage.",
    )

    parser.add_argument(
        "--trace",
        nargs="?",
        const="trace.jsonl",
        metavar="FILE",
        dest="trace",
        help="Write trace events to FILE in JSON Lines format (default: trace.jsonl)",
    )

    parser.add_argument(
        "--trace-events",
        type=str,
        metavar="CATEGORIES",
        dest="trace_events",
        help=(
            "Comma-separated list of event categories to record "
            "(operation, kernel, dfb, copy, pipe). "
            "Mutually exclusive with --no-trace-events. Requires --trace."
        ),
    )

    parser.add_argument(
        "--no-trace-events",
        type=str,
        metavar="CATEGORIES",
        dest="no_trace_events",
        help=(
            "Comma-separated list of event categories to suppress. "
            "Mutually exclusive with --trace-events. Requires --trace."
        ),
    )

    args, script_args = parser.parse_known_intermixed_args()
    args.script_args = script_args

    if not args.target:
        parser.print_help()
        sys.exit(1)

    # Set up simulator imports before running any code
    setup_simulator_imports()

    # Configure simulated device count
    if args.num_devices != 4:
        try:
            from .ttnnsim import set_num_devices

            set_num_devices(args.num_devices)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Configure max_dfbs limit if specified
    if args.max_dfbs is not None:
        try:
            from .program import set_max_dfbs

            set_max_dfbs(args.max_dfbs)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Configure L1 memory limit if specified
    if args.max_l1 is not None:
        try:
            from .program import set_max_l1_bytes

            set_max_l1_bytes(args.max_l1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Configure scheduler algorithm if specified
    if args.scheduler:

        set_scheduler_algorithm(args.scheduler)

    # Enable bfloat16-to-float32 promotion if requested
    if args.promote_bf16:
        from .ttnnsim import set_matmul_promote_bf16

        set_matmul_promote_bf16(True)

    # Enable tensor statistics collection if requested
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

    # Validate and configure tracing
    if args.trace_events and not args.trace:
        print("Error: --trace-events requires --trace", file=sys.stderr)
        sys.exit(1)
    if args.no_trace_events and not args.trace:
        print("Error: --no-trace-events requires --trace", file=sys.stderr)
        sys.exit(1)
    if args.trace_events and args.no_trace_events:
        print(
            "Error: --trace-events and --no-trace-events are mutually exclusive",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.trace:
        from .trace import ALL_CATEGORIES
        from .context import get_context

        if args.trace_events:
            cats = {c.strip() for c in args.trace_events.split(",")}
            unknown = cats - ALL_CATEGORIES
            if unknown:
                print(
                    f"Error: Unknown trace categories: {', '.join(sorted(unknown))}. "
                    f"Known: {', '.join(sorted(ALL_CATEGORIES))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            trace_set: frozenset[str] = frozenset(cats)
        elif args.no_trace_events:
            cats = {c.strip() for c in args.no_trace_events.split(",")}
            unknown = cats - ALL_CATEGORIES
            if unknown:
                print(
                    f"Error: Unknown trace categories: {', '.join(sorted(unknown))}. "
                    f"Known: {', '.join(sorted(ALL_CATEGORIES))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            trace_set = ALL_CATEGORIES - frozenset(cats)
        else:
            trace_set = ALL_CATEGORIES

        get_context().config.trace_set = trace_set

    # Run the target
    try:
        if not args.target.endswith(".py"):
            print(f"Error: Target must be a .py file: {args.target}", file=sys.stderr)
            sys.exit(1)
        run_file(args.target, args.script_args)
    finally:
        # Write trace events to file if requested
        if args.trace:
            from .context import get_context

            _write_jsonl_trace(Path(args.trace), get_context().trace_events)


if __name__ == "__main__":
    main()
