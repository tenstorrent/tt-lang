#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

TOOLCHAIN_DIR="${1:-/opt/ttmlir-toolchain}"

echo "Cleaning up toolchain at: ${TOOLCHAIN_DIR}"
echo "Size before cleanup:"
du -sh "${TOOLCHAIN_DIR}"

# Remove temporary files and source directories
rm -rf "${TOOLCHAIN_DIR}/tmp"/* "${TOOLCHAIN_DIR}/src" 2>/dev/null || true

# Remove debug symbols (.dSYM bundles) from binaries
find "${TOOLCHAIN_DIR}" -name "*.dSYM" -type d -exec rm -rf {} + 2>/dev/null || true

# Strip debug symbols from binaries to reduce size dramatically
echo "Stripping debug symbols from binaries..."
find "${TOOLCHAIN_DIR}/bin" -type f -perm +111 -exec strip -x {} \; 2>/dev/null || true

# Remove unnecessary LLVM/MLIR tools that aren't needed for building tt-lang
# Keep only: llvm-config, llvm-tblgen, mlir-tblgen, FileCheck, and linker tools
echo "Removing unnecessary LLVM/MLIR tools..."
cd "${TOOLCHAIN_DIR}/bin"
# List of tools to KEEP (everything else gets deleted)
KEEP_TOOLS=(
  "llvm-config" "llvm-tblgen" "llvm-ar" "llvm-ranlib" "llvm-nm"
  "mlir-tblgen"
  "FileCheck" "not" "count"
  "ld.lld" "lld" "ld64.lld" "wasm-ld"
)

# Create a temporary directory with tools to keep
mkdir -p /tmp/keep_bins
for tool in "${KEEP_TOOLS[@]}"; do
  find . -maxdepth 1 -name "$tool" -exec cp -p {} /tmp/keep_bins/ \; 2>/dev/null || true
done

# Remove all binaries
rm -f *

# Restore kept binaries
mv /tmp/keep_bins/* . 2>/dev/null || true
rmdir /tmp/keep_bins

cd - > /dev/null

echo "Size after cleanup:"
du -sh "${TOOLCHAIN_DIR}"
