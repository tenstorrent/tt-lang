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
    import ttl.pykernel

    from ttl import operation, compute, datamovement  # noqa: F401
    from ttl.pykernel._src.kernel_ast import TTCompilerBase  # noqa: F401
    from ttl.sim.ttlang_sim import main as _ttlang_sim_entry  # noqa: F401

    version = ttl.__version__
    if version == "0.0.0":
        print(
            f"tt-lang version is the fallback {version!r}; "
            "CMake-generated ttl/config.py was not bundled in the wheel.",
            file=sys.stderr,
        )
        return 1

    print(f"tt-lang {version}: imports OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
