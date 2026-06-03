#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Configure tt-lang for wheel builds. Reads optional external tt-metal /
# toolchain locations from env so the workflow's `env:` block stays the single
# source of truth for these inputs.
#
# Required env: none.
# Optional env:
#   TTLANG_EXTERNAL_TT_METAL_DIR        -> -DTTLANG_EXTERNAL_TT_METAL_DIR
#   TTLANG_EXTERNAL_TT_METAL_BUILD_DIR  -> -DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR
#   TTLANG_PYTHON_VENV                  -> -DTTLANG_PYTHON_VENV
#
# Usage: configure-ttlang-build.sh <build_dir>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <build_dir>" >&2
    exit 2
fi

build_dir="$1"

cmake_args=(
    -G Ninja
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
    -DTTLANG_USE_TOOLCHAIN=ON
)

if [[ -n "${TTLANG_EXTERNAL_TT_METAL_DIR:-}" ]]; then
    cmake_args+=(-DTTLANG_EXTERNAL_TT_METAL_DIR="$TTLANG_EXTERNAL_TT_METAL_DIR")
fi
if [[ -n "${TTLANG_EXTERNAL_TT_METAL_BUILD_DIR:-}" ]]; then
    cmake_args+=(-DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR="$TTLANG_EXTERNAL_TT_METAL_BUILD_DIR")
fi
if [[ -n "${TTLANG_PYTHON_VENV:-}" ]]; then
    cmake_args+=(-DTTLANG_PYTHON_VENV="$TTLANG_PYTHON_VENV")
fi

cmake "${cmake_args[@]}"
