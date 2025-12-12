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
rm -rf "${TOOLCHAIN_DIR}/tmp"/* 2>/dev/null || true

# Remove debug symbols (.dSYM bundles) from binaries
find "${TOOLCHAIN_DIR}" -name "*.dSYM" -type d -exec rm -rf {} + 2>/dev/null || true

# Strip debug symbols from binaries to reduce size dramatically
echo "Stripping debug symbols from binaries..."
find "${TOOLCHAIN_DIR}/bin" -type f -perm +111 -exec strip -x {} \; 2>/dev/null || true

echo "Size after cleanup:"
du -sh "${TOOLCHAIN_DIR}"
