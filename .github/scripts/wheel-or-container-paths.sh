#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Paths whose change should trigger PR-time wheel build + dist-container
# tutorial coverage. Sourced by wheel-or-container-changed.sh. Disjoint
# purpose from UPLIFT_PATHS (which decides whether to rebuild the image).

WHEEL_OR_CONTAINER_PATHS=(
    .github/containers/Dockerfile
    .github/containers/Dockerfile.base
    .github/scripts/run-tutorials.sh
    .github/scripts/smoke-test-wheel.py
    bin
    CMakeLists.txt
    examples
    packaging
    pyproject.toml
    python/CMakeLists.txt
    python/setup.py
)
