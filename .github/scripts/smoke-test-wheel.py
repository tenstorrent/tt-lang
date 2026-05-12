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

import sys


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
