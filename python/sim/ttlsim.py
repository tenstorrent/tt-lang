# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TT-Lang Simulator launcher (ttlsim).

Runs tt-lang kernels written for the compiler on the simulator backend
without requiring any code changes to the kernel files.

Usage:
    python ttlsim examples/metal_examples/singlecore_matmul/ttlang/singlecore_matmul.py
    python ttlsim test/sim/my_test.py --verbose
    python ttlsim -m test.sim.my_kernel
"""

import sys
import argparse
from pathlib import Path
from typing import Any


def setup_simulator_imports() -> None:
    """
    Inject simulator implementations into sys.modules so they shadow the compiler APIs.

    This allows kernel code written for the compiler to transparently use simulator
    implementations when run under ttlsim.
    """
    # Import simulator implementations
    from sim import ttl, ttnn

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
        exec(code, exec_globals)


def run_module(module_name: str, argv: list[str]) -> None:
    """
    Import and run a kernel module with simulator backend.

    Args:
        module_name: Dotted module name (e.g., 'test.sim.my_kernel')
        argv: Command-line arguments to pass to the module
    """
    sys.argv = [module_name] + argv

    import importlib

    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error: Could not import module '{module_name}': {e}", file=sys.stderr)
        sys.exit(1)

    # Run main if it exists, otherwise just importing executes the module
    if hasattr(mod, "main"):
        mod.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttlsim",
        description="Run tt-lang kernels on the simulator backend",
        epilog="Examples:\n"
        "  python ttlsim examples/metal_examples/singlecore_matmul/ttlang/singlecore_matmul.py\n"
        "  python ttlsim test/sim/test_add.py -v\n"
        "  python ttlsim -m test.sim.my_kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="Python file (.py) or module name to run",
    )

    parser.add_argument(
        "-m",
        "--module",
        action="store_true",
        help="Treat target as a module name instead of a file path",
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

    # Run the target
    if args.module:
        run_module(args.target, args.script_args)
    elif args.target.endswith(".py"):
        run_file(args.target, args.script_args)
    else:
        print(f"Error: Invalid target: {args.target}", file=sys.stderr)
        print("Target must be a .py file or use -m for module name", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
