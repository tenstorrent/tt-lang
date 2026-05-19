#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Wheel smoke test: import every top-level subpackage that the wheel ships
# and assert the version was populated by CMake (not the "0.0.0" fallback in
# python/ttl/version.py).
#
# Run inside a venv with the wheel installed:
#   pip install dist/*.whl
#   python .github/scripts/smoke-test-wheel.py
#
# Also checks that `sim_stats` and the `ttlang-sim` / `ttlang-sim-stats` console
# scripts were installed with the wheel (bundled entry points, not separate packages).

import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    import ttl
    import ttl.sim

    from ttl.sim.ttlang_sim import main as _ttlang_sim_entry  # noqa: F401

    sim_only = getattr(ttl, "_SIM_ONLY_INSTALL", False)
    package = "tt-lang-sim" if sim_only else "tt-lang"
    if not sim_only:
        import ttl.pykernel  # noqa: F401
        from ttl import operation, compute, datamovement  # noqa: F401
        from ttl.pykernel._src.kernel_ast import TTCompilerBase  # noqa: F401

    version = ttl.__version__
    if version == "0.0.0":
        print(
            f"{package} version is the fallback {version!r}; "
            "version metadata was not bundled in the wheel.",
            file=sys.stderr,
        )
        return 1

    print(f"{package} {version}: imports OK")

    # sim_stats package + ttlang-sim-stats entry point (pip console_scripts)
    import sim_stats  # noqa: F401
    from sim_stats.__main__ import main as _sim_stats_main  # noqa: F401

    # venv-aware; sys.executable may be a symlink out of the venv.
    scripts_dir = Path(sysconfig.get_path("scripts"))
    exe_suffix = ".exe" if sys.platform == "win32" else ""

    def run_help(label: str, argv: list[str]) -> int:
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"{label} --help timed out after 60s", file=sys.stderr)
            return 1
        if r.returncode != 0:
            print(
                f"{label} --help failed (exit {r.returncode}):\n{r.stderr}",
                file=sys.stderr,
            )
            return 1
        return 0

    for script in ("ttlang-sim", "ttlang-sim-stats"):
        script_path = scripts_dir / f"{script}{exe_suffix}"
        if not script_path.is_file():
            print(
                f"missing console script (expected next to this Python): {script_path}",
                file=sys.stderr,
            )
            return 1
        if run_help(script, [str(script_path), "--help"]) != 0:
            return 1

    if (
        run_help("python -m sim_stats", [sys.executable, "-m", "sim_stats", "--help"])
        != 0
    ):
        return 1

    print(f"{package} {version}: sim_stats + ttlang-sim-stats OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
