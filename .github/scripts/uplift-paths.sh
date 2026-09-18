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
#   - Toolchain build logic and deps  (CMake modules, scripts, requirements)
# tt-lang is built fresh by call-build.yml against the pre-built LLVM inside
# the container, so ordinary source changes are NOT in this list.

UPLIFT_PATHS=(
    .dockerignore
    CMakeLists.txt
    cmake/modules/BuildLLVM.cmake
    cmake/modules/BuildTTMetal.cmake
    cmake/modules/GetVersionFromGit.cmake
    cmake/modules/TTLangCompilerSetup.cmake
    cmake/modules/TTLangPython.cmake
    cmake/modules/TTLangToolchainComponent.cmake
    cmake/modules/TTLangToolchainOptions.cmake
    cmake/modules/TTLangUtils.cmake
    third-party/tt-metal-version
    third-party/llvm-project
    third-party/tt-metal
    third-party/patches
    .github/containers/Dockerfile
    .github/containers/Dockerfile.base
    .github/containers/cleanup-toolchain.sh
    .github/containers/install-exabox-worker.sh
    .github/scripts/normalize-toolchain-install.sh
    bin/tt-triage
    dev-requirements.txt
    docs/requirements.txt
    requirements.txt
    requirements-runtime.txt
    scripts/build-and-install.sh
    scripts/copy-ttmetal-runtime-artifacts.sh
    scripts/install-ttmetal.sh
    scripts/verify-sha.sh
)
