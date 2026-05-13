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
# Also checks that `sim_stats` and the `ttlang-sim-stats` console script were
# installed with the wheel (bundled entry point, not a separate package).

import subprocess
import sys
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

    scripts_dir = Path(sys.executable).resolve().parent
    stats_name = (
        "ttlang-sim-stats.exe" if sys.platform == "win32" else "ttlang-sim-stats"
    )
    stats_path = scripts_dir / stats_name
    if not stats_path.is_file():
        print(
            f"missing console script (expected next to this Python): {stats_path}",
            file=sys.stderr,
        )
        return 1

    r = subprocess.run(
        [str(stats_path), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        print(
            f"ttlang-sim-stats --help failed (exit {r.returncode}):\n{r.stderr}",
            file=sys.stderr,
        )
        return 1

    r2 = subprocess.run(
        [sys.executable, "-m", "sim_stats", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r2.returncode != 0:
        print(
            f"python -m sim_stats --help failed (exit {r2.returncode}):\n{r2.stderr}",
            file=sys.stderr,
        )
        return 1

    print(f"{package} {version}: sim_stats + ttlang-sim-stats OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
