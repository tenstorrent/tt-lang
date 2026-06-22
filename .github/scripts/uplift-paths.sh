#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Paths whose change between two commits indicates container image content
# would differ -- i.e. a new image must be built. Sourced by
# detect-uplift.sh (drift signal) and get-version-tag.sh (deterministic
# docker-tag suffix).
#
# Container contents:
#   - System packages + SFPI/firmware (driven by tt-metal-version)
#   - Pre-built LLVM artifacts        (driven by third-party/llvm-project)
#   - Pre-built tt-metal artifacts    (driven by third-party/tt-metal)
#   - Python runtime deps             (requirements-runtime.txt)
# tt-mlir and tt-lang are built fresh by call-build.yml against the
# pre-built LLVM inside the container, so they are NOT in this list.

UPLIFT_PATHS=(
    third-party/tt-metal-version
    third-party/llvm-project
    third-party/tt-metal
    .github/containers/Dockerfile.base
    .github/containers/Dockerfile.wheel-manylinux-2-34
    requirements-runtime.txt
)
