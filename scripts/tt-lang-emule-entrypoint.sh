#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly TTLANG_SOURCE_DIR="${TTLANG_EMULE_SOURCE_DIR:-/workspace}"
readonly TTLANG_BUILD_DIR="${TTLANG_EMULE_BUILD_DIR:-/ttlang-build}"
readonly TT_METAL_SOURCE_DIR="/opt/tt-emule-runtime/tt-metal"
readonly TT_METAL_BUILD_DIR="${TT_METAL_SOURCE_DIR}/build_emule"
readonly CLUSTER_DESCRIPTORS="${TT_METAL_SOURCE_DIR}/tt_metal/third_party/umd/tests/cluster_descriptor_examples"

if [ "$#" -eq 0 ]; then
    echo "tt-lang emule container: no Python script was provided." >&2
    exit 2
fi
if [ ! -f "$1" ]; then
    echo "tt-lang emule container: script not found: $1" >&2
    exit 2
fi

export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH="${TT_METAL_MOCK_CLUSTER_DESC_PATH:-${CLUSTER_DESCRIPTORS}/wormhole_N150.yaml}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tt-metal-cache}"
export TT_EMULE_JIT_CACHE_DIR="${TT_EMULE_JIT_CACHE_DIR:-${TT_METAL_CACHE}/emule-jit}"
export MESH_DEVICE="${MESH_DEVICE:-N150}"
unset TTLANG_COMPILE_ONLY TTLANG_SIM_ONLY

if [ ! -f "${TT_METAL_MOCK_CLUSTER_DESC_PATH}" ]; then
    echo "tt-lang emule container: cluster descriptor not found: ${TT_METAL_MOCK_CLUSTER_DESC_PATH}" >&2
    exit 1
fi

cmake -G Ninja -S "$TTLANG_SOURCE_DIR" -B "$TTLANG_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DLLVM_USE_LINKER=lld-20 \
    -DTTLANG_USE_TOOLCHAIN=ON \
    -DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF \
    -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain \
    -DTTLANG_EXTERNAL_TT_METAL_DIR="$TT_METAL_SOURCE_DIR" \
    -DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR="$TT_METAL_BUILD_DIR" \
    -DTTLANG_ENABLE_PERF_TRACE=OFF

cmake --build "$TTLANG_BUILD_DIR" --parallel "${TTLANG_EMULE_JOBS:-$(nproc)}"

# activate is generated for bash and may legitimately reference variables that
# are absent from a non-interactive container shell.
set +u
source "${TTLANG_BUILD_DIR}/env/activate"
set -u

export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH
export TT_METAL_CACHE
export TT_EMULE_JIT_CACHE_DIR
export MESH_DEVICE
unset TTLANG_COMPILE_ONLY TTLANG_SIM_ONLY

exec python "$@"
