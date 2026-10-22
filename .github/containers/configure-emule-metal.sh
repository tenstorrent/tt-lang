#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Configure the pinned tt-metal source for the emule runtime image. Without an
# emulator source, configuring downloads tt-metal's CPM packages into the
# source tree's .cpmcache and the SFPI toolchain into runtime/sfpi; with one,
# it uses those downloads and must not need the network.

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <tt-metal-source> <build-directory> [<tt-emule-source>]" >&2
    exit 2
fi
readonly SOURCE="$1"
readonly BUILD="$2"

if [ "$#" -eq 3 ]; then
    emule_options=(
        -DTT_METAL_USE_EMULE=ON
        "-DTT_EMULE_PATH=$3"
        "-DFETCHCONTENT_SOURCE_DIR_SFPI=$SOURCE/runtime/sfpi"
    )
else
    emule_options=(-DTT_METAL_USE_EMULE=OFF)
fi

cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$SOURCE/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD" \
    -DPython3_EXECUTABLE=/opt/ttlang-toolchain/venv/bin/python \
    -DWITH_PYTHON_BINDINGS=ON \
    -DTT_UNITY_BUILDS=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PROGRAMMING_EXAMPLES=OFF \
    -DTT_METAL_BUILD_TESTS=OFF \
    -DTTNN_BUILD_TESTS=OFF \
    -DBUILD_TT_TRAIN=OFF \
    -DBUILD_TELEMETRY=OFF \
    -DENABLE_TTNN_SHARED_SUBLIBS=OFF \
    -DTT_ENABLE_LIGHT_METAL_TRACE=OFF \
    -DENABLE_LIBCXX=OFF \
    "${emule_options[@]}"
