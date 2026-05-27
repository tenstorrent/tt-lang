# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Configure a no-ttnn tt-lang wheel to use an existing tt-metal build.

Internal `tt-lang-light` wheels depend on a matching `tt-lang` wheel built with
`TTLANG_TTNN_DEP_MODE=external`. That wheel intentionally omits
`Requires-Dist: ttnn`, so pip does not install the public `ttnn` wheel or
replace a user's newer local tt-metal build.

This module backs the installed `tt-lang-setup-external-tt-metal` console
script. It detects either a tt-metal install layout or a source/build layout and
either prints shell exports or runs a supplied command with `TT_METAL_HOME`,
`TT_METAL_RUNTIME_ROOT`, `PYTHONPATH`, and `LD_LIBRARY_PATH` configured. The
generated environment lets Python import `ttnn` from the selected tt-metal tree
while importing `ttl` from the installed tt-lang wheel.

The selected tt-metal tree is trusted input. Passing `--check` imports `ttnn`
from that tree and therefore may execute native code from `_ttnn.so`.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalTTMetalEnv:
    tt_metal_home: Path
    python_entries: tuple[Path, ...]
    library_entries: tuple[Path, ...]


def _resolve_existing_dir(raw_path: str, label: str) -> Path:
    resolved_path = Path(raw_path).expanduser().resolve()
    if not resolved_path.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved_path}")
    if ":" in str(resolved_path):
        raise ValueError(f"{label} must not contain ':' because it is used in paths")
    return resolved_path


def detect_external_tt_metal(
    tt_metal_dir: str | Path,
    build_dir: str | Path | None = None,
) -> ExternalTTMetalEnv:
    tt_metal_root = _resolve_existing_dir(str(tt_metal_dir), "tt-metal directory")

    install_ttnn_extension = (
        tt_metal_root / "python_packages" / "ttnn" / "ttnn" / "_ttnn.so"
    )
    if install_ttnn_extension.exists():
        return ExternalTTMetalEnv(
            tt_metal_home=tt_metal_root,
            python_entries=(
                tt_metal_root / "python_packages" / "ttnn",
                tt_metal_root / "python_packages" / "tools",
            ),
            library_entries=(tt_metal_root / "lib",),
        )

    native_ttnn_package = tt_metal_root / "ttnn" / "ttnn"
    native_tt_metal_source = tt_metal_root / "tt_metal"
    if native_ttnn_package.is_dir() and native_tt_metal_source.is_dir():
        native_build_dir = (
            _resolve_existing_dir(str(build_dir), "tt-metal build directory")
            if build_dir is not None
            else tt_metal_root / "build"
        )
        native_library_dir = native_build_dir / "lib"
        if not native_library_dir.is_dir():
            raise ValueError(
                "external tt-metal source tree has no built libraries at "
                f"{native_library_dir}; build tt-metal first or pass "
                "--build-dir"
            )
        return ExternalTTMetalEnv(
            tt_metal_home=tt_metal_root,
            python_entries=(tt_metal_root / "ttnn", tt_metal_root / "tools"),
            library_entries=(
                native_library_dir,
                native_build_dir / "tt_metal",
                native_build_dir / "ttnn",
                native_build_dir / "tt_stl",
                native_build_dir / "_deps" / "fmt-build",
                native_build_dir / "tt_metal" / "third_party" / "umd" / "device",
            ),
        )

    raise ValueError(
        f"{tt_metal_root} is neither an install layout with "
        "python_packages/ttnn/ttnn/_ttnn.so nor a native tt-metal source tree "
        "with ttnn/ttnn and tt_metal directories"
    )


def environment_for_external_tt_metal(
    settings: ExternalTTMetalEnv,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment["TT_METAL_HOME"] = str(settings.tt_metal_home)
    environment["TT_METAL_RUNTIME_ROOT"] = str(settings.tt_metal_home)
    environment["PYTHONPATH"] = _prepend_paths(
        settings.python_entries, environment.get("PYTHONPATH", "")
    )
    environment["LD_LIBRARY_PATH"] = _prepend_paths(
        settings.library_entries, environment.get("LD_LIBRARY_PATH", "")
    )
    return environment


def emit_shell_exports(settings: ExternalTTMetalEnv) -> str:
    python_prefix = _shell_path_list(settings.python_entries)
    library_prefix = _shell_path_list(settings.library_entries)
    tt_metal_home = shlex.quote(str(settings.tt_metal_home))
    return "\n".join(
        [
            f"export TT_METAL_HOME={tt_metal_home}",
            'export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"',
            f"export PYTHONPATH={python_prefix}${{PYTHONPATH:+:$PYTHONPATH}}",
            (
                "export LD_LIBRARY_PATH="
                f"{library_prefix}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
            ),
        ]
    )


def _prepend_paths(entries: tuple[Path, ...], existing_value: str) -> str:
    entry_text = ":".join(str(entry) for entry in entries)
    if not existing_value:
        return entry_text
    return f"{entry_text}:{existing_value}"


def _shell_path_list(entries: tuple[Path, ...]) -> str:
    return shlex.quote(":".join(str(entry) for entry in entries))


def _check_import(settings: ExternalTTMetalEnv) -> int:
    environment = environment_for_external_tt_metal(settings)
    result = subprocess.run(
        [sys.executable, "-c", "import ttnn; print(ttnn.__file__)"],
        env=environment,
        text=True,
        check=False,
    )
    return result.returncode


def run_command_with_external_tt_metal(
    settings: ExternalTTMetalEnv, command: list[str]
) -> int:
    result = subprocess.run(
        command,
        env=environment_for_external_tt_metal(settings),
        check=False,
    )
    return result.returncode


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    try:
        separator_index = argv.index("--")
    except ValueError:
        return argv, []

    command = argv[separator_index + 1 :]
    if not command:
        raise ValueError("command after '--' is required")
    return argv[:separator_index], command


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        parser_argv, command = _split_command(raw_argv)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(
        prog="tt-lang-setup-external-tt-metal",
        description=(
            "Print shell exports for using a tt-lang wheel built with "
            "TTLANG_TTNN_DEP_MODE=external, or run a command with that "
            "environment."
        ),
    )
    parser.add_argument(
        "tt_metal_dir",
        nargs="?",
        help="existing tt-metal source or install directory",
    )
    parser.add_argument(
        "--tt-metal-dir",
        dest="tt_metal_dir_option",
        help="existing tt-metal source or install directory",
    )
    parser.add_argument(
        "--build-dir",
        help="native tt-metal build directory; defaults to <tt-metal-dir>/build",
    )
    parser.add_argument(
        "--format",
        choices=("shell",),
        default="shell",
        help="output format (default: shell)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "validate that the resulting environment can import ttnn; only use "
            "with trusted tt-metal builds because this imports native code"
        ),
    )
    args = parser.parse_args(parser_argv)

    tt_metal_dir = args.tt_metal_dir_option or args.tt_metal_dir
    if not tt_metal_dir:
        parser.error("tt-metal directory is required")

    try:
        settings = detect_external_tt_metal(tt_metal_dir, args.build_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.check:
        import_status = _check_import(settings)
        if import_status != 0:
            return import_status

    if command:
        return run_command_with_external_tt_metal(settings, command)

    print(emit_shell_exports(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
