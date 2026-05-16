#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Paths whose change between two commits indicates a toolchain or base-image
# uplift. Sourced by detect-uplift.sh (drift signal) and get-version-tag.sh
# (deterministic docker-tag suffix).

UPLIFT_PATHS=(
    third-party/tt-metal-version
    third-party/llvm-project
    third-party/tt-mlir
    third-party/tt-metal
    .github/containers/Dockerfile.base
    pyproject.toml
    requirements-runtime.txt
)
