#!/usr/bin/env bash
# Build the LLVM/MLIR toolchain from tt-mlir source.
#
# Usage: ./tools/build-toolchain.sh <tt-mlir-source-dir> [build-dir]
#
# Environment:
#   TTMLIR_TOOLCHAIN_DIR - Required. Directory where toolchain will be installed.
#
# The build directory defaults to <tt-mlir-source-dir>/env/build if not specified.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <tt-mlir-source-dir> [build-dir]" >&2
    exit 1
fi

TTMLIR_SOURCE_DIR="$1"
BUILD_DIR="${2:-${TTMLIR_SOURCE_DIR}/env/build}"

if [[ -z "${TTMLIR_TOOLCHAIN_DIR:-}" ]]; then
    echo "Error: TTMLIR_TOOLCHAIN_DIR environment variable is not set." >&2
    echo "Set it to the toolchain install directory, e.g.: export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain" >&2
    exit 1
fi

if [[ ! -d "${TTMLIR_SOURCE_DIR}/env" ]]; then
    echo "Error: ${TTMLIR_SOURCE_DIR}/env does not exist. Is this a tt-mlir source directory?" >&2
    exit 1
fi

echo "Building LLVM/MLIR toolchain..."
echo "  Source: ${TTMLIR_SOURCE_DIR}/env"
echo "  Build:  ${BUILD_DIR}"
echo "  Install: ${TTMLIR_TOOLCHAIN_DIR}"

# shellcheck source=/dev/null
source "${TTMLIR_SOURCE_DIR}/env/activate"

cmake -G Ninja -B "${BUILD_DIR}" -S "${TTMLIR_SOURCE_DIR}/env"
cmake --build "${BUILD_DIR}"

echo "Toolchain built and installed to: ${TTMLIR_TOOLCHAIN_DIR}"
