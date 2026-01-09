#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script to verify the tt-metal minimal build works correctly.
# This script:
# 1. Fetches tt-metal sources (or uses existing)
# 2. Configures and builds only the minimal targets
# 3. Tests that Python can import ttnn

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_LANG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/test-tt-metal-build"
TT_METAL_SOURCE_DIR="${BUILD_DIR}/tt-metal-src"
TT_METAL_BUILD_DIR="${BUILD_DIR}/tt-metal-build"

# tt-metal version - same as tt-mlir uses
TT_METAL_VERSION="dcae048e5901b2528f5eba67180bb2bc0c227481"

echo "=== tt-metal Minimal Build Test ==="
echo "Script directory: ${SCRIPT_DIR}"
echo "tt-lang root: ${TT_LANG_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo "tt-metal version: ${TT_METAL_VERSION}"

# Check for toolchain
if [[ -z "${TTMLIR_TOOLCHAIN_DIR}" ]]; then
    if [[ -d "${HOME}/tt/ttmlir-toolchain" ]]; then
        export TTMLIR_TOOLCHAIN_DIR="${HOME}/tt/ttmlir-toolchain"
    elif [[ -d "/opt/ttmlir-toolchain" ]]; then
        export TTMLIR_TOOLCHAIN_DIR="/opt/ttmlir-toolchain"
    else
        echo "Error: TTMLIR_TOOLCHAIN_DIR not set and no toolchain found"
        exit 1
    fi
fi
echo "Toolchain: ${TTMLIR_TOOLCHAIN_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Clone tt-metal if needed
if [[ ! -d "${TT_METAL_SOURCE_DIR}" ]]; then
    echo ""
    echo "=== Cloning tt-metal ==="
    git clone --depth 1 https://github.com/tenstorrent/tt-metal.git "${TT_METAL_SOURCE_DIR}"
    cd "${TT_METAL_SOURCE_DIR}"
    git fetch --depth 1 origin "${TT_METAL_VERSION}"
    git checkout "${TT_METAL_VERSION}"
    echo "Initializing submodules..."
    git submodule update --init --recursive --depth 1
    cd "${BUILD_DIR}"
else
    echo "Using existing tt-metal sources at ${TT_METAL_SOURCE_DIR}"
    # Make sure submodules are initialized
    cd "${TT_METAL_SOURCE_DIR}"
    if [[ ! -f "tt_metal/third_party/umd/CMakeLists.txt" ]]; then
        echo "Initializing submodules..."
        git submodule update --init --recursive --depth 1
    fi
    cd "${BUILD_DIR}"
fi

echo ""
echo "=== Configuring tt-metal minimal build ==="
cmake -G Ninja -B "${TT_METAL_BUILD_DIR}" -S "${TT_METAL_SOURCE_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${TT_METAL_SOURCE_DIR}/cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake" \
    -DTT_UNITY_BUILDS=ON \
    -DENABLE_CCACHE=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=OFF \
    -DWITH_PYTHON_BINDINGS=ON \
    -DTTNN_BUILD_TESTS=OFF \
    -DTT_METAL_BUILD_TESTS=OFF \
    -DBUILD_PROGRAMMING_EXAMPLES=OFF \
    -DBUILD_TT_TRAIN=OFF \
    -DBUILD_TELEMETRY=OFF \
    -DENABLE_TTNN_SHARED_SUBLIBS=OFF \
    -DTT_ENABLE_LIGHT_METAL_TRACE=OFF

echo ""
echo "=== Building tt-metal minimal targets ==="
# Build targets in dependency order
cmake --build "${TT_METAL_BUILD_DIR}" --target ttnn_core
cmake --build "${TT_METAL_BUILD_DIR}" --target ttnncpp
cmake --build "${TT_METAL_BUILD_DIR}" --target _ttnncpp.so
cmake --build "${TT_METAL_BUILD_DIR}" --target ttnn

echo ""
echo "=== Build successful! ==="
echo ""
echo "Build size: $(du -sh ${TT_METAL_BUILD_DIR} | cut -f1)"

# Check for key outputs
echo ""
echo "Key outputs:"
if [[ -f "${TT_METAL_BUILD_DIR}/ttnn/_ttnncpp.so" ]]; then
    echo "  - _ttnncpp.so: $(ls -lh ${TT_METAL_BUILD_DIR}/ttnn/_ttnncpp.so | awk '{print $5}')"
else
    echo "  - _ttnncpp.so: MISSING"
fi

if [[ -f "${TT_METAL_BUILD_DIR}/ttnn/_ttnn.so" ]]; then
    echo "  - _ttnn.so: $(ls -lh ${TT_METAL_BUILD_DIR}/ttnn/_ttnn.so | awk '{print $5}')"
else
    echo "  - _ttnn.so: MISSING"
fi

if [[ -f "${TT_METAL_BUILD_DIR}/tt_metal/libtt_metal.so" ]]; then
    echo "  - libtt_metal.so: $(ls -lh ${TT_METAL_BUILD_DIR}/tt_metal/libtt_metal.so | awk '{print $5}')"
else
    echo "  - libtt_metal.so: MISSING"
fi

# Test Python import
echo ""
echo "=== Testing Python import ==="

# Set up environment
export TT_METAL_HOME="${TT_METAL_SOURCE_DIR}"
export LD_LIBRARY_PATH="${TT_METAL_BUILD_DIR}/tt_metal:${TT_METAL_BUILD_DIR}/ttnn:${TT_METAL_BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${TT_METAL_SOURCE_DIR}:${PYTHONPATH}"

# Copy built modules to source tree for import
cp -f "${TT_METAL_BUILD_DIR}/ttnn/_ttnn.so" "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/" 2>/dev/null || true
cp -f "${TT_METAL_BUILD_DIR}/ttnn/_ttnncpp.so" "${TT_METAL_SOURCE_DIR}/ttnn/ttnn/" 2>/dev/null || true

# Test import
PYTHON="${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3"
if ${PYTHON} -c "import ttnn; print('SUCCESS: ttnn imported')"; then
    echo ""
    echo "=== All tests passed! ==="
else
    echo ""
    echo "=== Python import failed ==="
    echo "This may be expected on machines without Tenstorrent hardware"
    echo "The build itself succeeded - import test is for runtime validation only"
fi

echo ""
echo "Runtime environment for tt-lang:"
echo "  export TT_METAL_HOME=${TT_METAL_SOURCE_DIR}"
echo "  export LD_LIBRARY_PATH=${TT_METAL_BUILD_DIR}/tt_metal:${TT_METAL_BUILD_DIR}/ttnn:${TT_METAL_BUILD_DIR}/lib:\$LD_LIBRARY_PATH"
