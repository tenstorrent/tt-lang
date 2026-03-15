#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Collect C++ coverage data from LLVM profraw files and generate reports.
#
# Prerequisites:
#   - Build with -DTTLANG_CODE_COVERAGE=ON
#   - Run tests with LLVM_PROFILE_FILE set (e.g., LLVM_PROFILE_FILE=build/coverage/%p.profraw)
#   - llvm-profdata and llvm-cov must be on PATH
#
# Usage:
#   scripts/collect-cpp-coverage.sh [build_dir]
#
# Outputs:
#   build/coverage/coverage.profdata  - merged profile data
#   build/coverage/lcov.info          - LCOV format for CI upload
#   build/coverage/cpp-html/          - HTML report

set -euo pipefail

BUILD_DIR="${1:-build}"
COV_DIR="${BUILD_DIR}/coverage"

# Find tools (prefer versioned names from LLVM packages)
LLVM_PROFDATA="${LLVM_PROFDATA:-$(command -v llvm-profdata || command -v llvm-profdata-18 || true)}"
LLVM_COV="${LLVM_COV:-$(command -v llvm-cov || command -v llvm-cov-18 || true)}"

if [[ -z "$LLVM_PROFDATA" || -z "$LLVM_COV" ]]; then
    echo "Error: llvm-profdata and llvm-cov must be on PATH" >&2
    exit 1
fi

# Find profraw files
PROFRAW_FILES=()
while IFS= read -r -d '' f; do
    PROFRAW_FILES+=("$f")
done < <(find "$COV_DIR" -name '*.profraw' -print0 2>/dev/null)

if [[ ${#PROFRAW_FILES[@]} -eq 0 ]]; then
    echo "Error: no .profraw files found in ${COV_DIR}" >&2
    echo "Run tests with: LLVM_PROFILE_FILE=${COV_DIR}/%p.profraw" >&2
    exit 1
fi

echo "Found ${#PROFRAW_FILES[@]} profraw file(s)"

# Merge profiles
echo "Merging profiles..."
"$LLVM_PROFDATA" merge -sparse "${PROFRAW_FILES[@]}" -o "${COV_DIR}/coverage.profdata"

# Find instrumented binaries and shared objects to report on.
# llvm-cov needs the first object as a positional arg and the rest via -object.
OBJECT_FILES=()
while IFS= read -r -d '' f; do
    OBJECT_FILES+=("$f")
done < <(find "${BUILD_DIR}/bin" -maxdepth 1 -type f -executable -print0 2>/dev/null)
while IFS= read -r -d '' f; do
    OBJECT_FILES+=("$f")
done < <(find "${BUILD_DIR}/python_packages" -name '*.so' -print0 2>/dev/null)
while IFS= read -r -d '' f; do
    OBJECT_FILES+=("$f")
done < <(find "${BUILD_DIR}/lib" -name '*.so' -print0 2>/dev/null)

if [[ ${#OBJECT_FILES[@]} -eq 0 ]]; then
    echo "Error: no instrumented binaries found in ${BUILD_DIR}" >&2
    exit 1
fi

# llvm-cov takes the first binary as a positional arg, rest as -object flags
OBJECTS=("${OBJECT_FILES[0]}")
for ((i = 1; i < ${#OBJECT_FILES[@]}; i++)); do
    OBJECTS+=("-object" "${OBJECT_FILES[$i]}")
done

# Resolve tt-lang source root for filtering.
SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Source filters: exclude everything that is not tt-lang's own source.
FILTERS=(
    -ignore-filename-regex='/opt/'
    -ignore-filename-regex='/third-party/'
    -ignore-filename-regex='/build/'
    -ignore-filename-regex='/nanobind/'
    -ignore-filename-regex='/pybind11/'
    -ignore-filename-regex='/site-packages/'
    -ignore-filename-regex='/usr/'
    -ignore-filename-regex='/python/'
)

echo "Source root: ${SOURCE_DIR}"

# Generate LCOV report for CI upload
echo "Generating LCOV report..."
"$LLVM_COV" export \
    -format=lcov \
    -instr-profile="${COV_DIR}/coverage.profdata" \
    "${OBJECTS[@]}" \
    "${FILTERS[@]}" \
    > "${COV_DIR}/lcov.info"

# Generate HTML report
echo "Generating HTML report..."
"$LLVM_COV" show \
    -format=html \
    -output-dir="${COV_DIR}/cpp-html" \
    -instr-profile="${COV_DIR}/coverage.profdata" \
    "${OBJECTS[@]}" \
    "${FILTERS[@]}"

echo "Coverage reports generated:"
echo "  LCOV:  ${COV_DIR}/lcov.info"
echo "  HTML:  ${COV_DIR}/cpp-html/index.html"
