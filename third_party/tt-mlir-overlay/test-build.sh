#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script to verify the tt-mlir minimal overlay builds correctly.
# This script:
# 1. Fetches tt-mlir sources
# 2. Configures and builds only the minimal targets via the overlay

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_LANG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/test-build"
TTMLIR_SOURCE_DIR="${BUILD_DIR}/tt-mlir-src"
TTMLIR_BINARY_DIR="${BUILD_DIR}/tt-mlir-build"

# Read tt-mlir commit from tt-lang
TTMLIR_COMMIT_FILE="${TT_LANG_ROOT}/third-party/tt-mlir.commit"
if [[ -f "${TTMLIR_COMMIT_FILE}" ]]; then
    TTMLIR_COMMIT=$(cat "${TTMLIR_COMMIT_FILE}" | tr -d '[:space:]')
else
    echo "Error: Could not find ${TTMLIR_COMMIT_FILE}"
    exit 1
fi

echo "=== tt-mlir Minimal Overlay Test Build ==="
echo "Script directory: ${SCRIPT_DIR}"
echo "tt-lang root: ${TT_LANG_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo "tt-mlir commit: ${TTMLIR_COMMIT}"

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

# Clone tt-mlir if needed
if [[ ! -d "${TTMLIR_SOURCE_DIR}" ]]; then
    echo ""
    echo "=== Cloning tt-mlir ==="
    git clone --depth 1 https://github.com/tenstorrent/tt-mlir.git "${TTMLIR_SOURCE_DIR}"
    cd "${TTMLIR_SOURCE_DIR}"
    git fetch --depth 1 origin "${TTMLIR_COMMIT}"
    git checkout "${TTMLIR_COMMIT}"
    cd "${BUILD_DIR}"
else
    echo "Using existing tt-mlir sources at ${TTMLIR_SOURCE_DIR}"
fi

# Create a wrapper CMakeLists.txt to test the overlay
cat > "${BUILD_DIR}/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.24.0)
project(TTMLIRMinimalOverlayTest)

# Find LLVM/MLIR from toolchain
find_package(MLIR REQUIRED CONFIG HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir")
find_package(LLVM REQUIRED CONFIG HINTS "${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Python support
if(MLIR_ENABLE_BINDINGS_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "Using Python: ${Python3_EXECUTABLE}")
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif()

# Set output directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Include the overlay (OVERLAY_DIR is passed from parent)
add_subdirectory("${OVERLAY_DIR}" "${CMAKE_BINARY_DIR}/overlay")
EOF

echo ""
echo "=== Configuring overlay build ==="
cmake -G Ninja -B "${TTMLIR_BINARY_DIR}" -S "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR}" \
    -Dttmlir_SOURCE_DIR="${TTMLIR_SOURCE_DIR}" \
    -Dttmlir_BINARY_DIR="${TTMLIR_BINARY_DIR}" \
    -DOVERLAY_DIR="${SCRIPT_DIR}" \
    -DPython3_EXECUTABLE="${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3" \
    -DPython3_ROOT_DIR="${TTMLIR_TOOLCHAIN_DIR}/venv" \
    -DFLATBUFFERS_COMPILER="${TTMLIR_TOOLCHAIN_DIR}/bin/flatc" \
    -DTTMLIR_MINIMAL_ENABLE_PYTHON=OFF

echo ""
echo "=== Building minimal overlay ==="
# TTMLIRMinimal is an INTERFACE target, so we build the actual targets it links:
# 1. Dialect libraries
cmake --build "${TTMLIR_BINARY_DIR}" --target MLIRTTCoreDialect MLIRTTMetalDialect MLIRTTKernelDialect MLIRTTIRDialect
# 2. Conversion and translation libraries
cmake --build "${TTMLIR_BINARY_DIR}" --target TTMLIRTTKernelToEmitC TTKernelTargetCpp
# 3. Python bindings (disabled for now - needs more work to avoid TTNN/D2M deps)
# cmake --build "${TTMLIR_BINARY_DIR}" --target TTMLIRMinimalCAPI _ttmlir_minimal || echo "Python bindings build skipped or failed"

echo ""
echo "=== Build successful! ==="
echo "Minimal targets built:"
echo "  - MLIRTTCoreDialect"
echo "  - MLIRTTMetalDialect"
echo "  - MLIRTTKernelDialect"
echo "  - MLIRTTIRDialect (minimal)"
echo "  - TTMLIRTTKernelToEmitC"
echo "  - TTKernelTargetCpp"

if [[ -f "${TTMLIR_BINARY_DIR}/python_packages/ttmlir/_mlir_libs/_ttmlir.cpython-311-x86_64-linux-gnu.so" ]] || \
   [[ -f "${TTMLIR_BINARY_DIR}/python_packages/ttmlir/_mlir_libs/_ttmlir.so" ]]; then
    echo "  - Python bindings (_ttmlir module)"
fi
