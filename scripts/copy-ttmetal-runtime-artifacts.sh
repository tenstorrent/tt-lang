#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Copy tt-metal runtime artifacts needed for JIT device compilation.
#
# Usage:
#   copy-ttmetal-runtime-artifacts.sh <src-dir> <dest-dir>
#   copy-ttmetal-runtime-artifacts.sh --restore <toolchain-dir> <source-dir>
#
# Default mode (save): copy from tt-metal source tree into toolchain.
# Restore mode: copy from toolchain back into source tree, skipping
#   artifacts that already exist in the source tree.
#
# Artifacts:
#   runtime/hw/                  - linker scripts and object files (build-generated)
#   runtime/sfpi/                - SFPI compiler intrinsics (JIT kernel compilation)
#   tt_metal/third_party/tt_llk/ - LLK headers (ckernel_structs.h, etc.)
#   tt_metal/soc_descriptors/    - SoC architecture descriptors (device open)
#   tt_metal/core_descriptors/   - Core architecture descriptors (device open)

set -euo pipefail

RESTORE=false
if [ "${1:-}" = "--restore" ]; then
    RESTORE=true
    shift
fi

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--restore] <src-dir> <dest-dir>"
    exit 1
fi

SRC="$1"
DEST="$2"

# Each entry: <check-file> <artifact-dir>
# check-file is used in restore mode to skip if already present in dest.
ARTIFACTS=(
    "runtime/hw/toolchain"                            "runtime/hw"
    "runtime/sfpi/include"                            "runtime/sfpi"
    "tt_metal/third_party/tt_llk/README.md"           "tt_metal/third_party/tt_llk"
    "tt_metal/soc_descriptors/blackhole_140_arch.yaml" "tt_metal/soc_descriptors"
    "tt_metal/core_descriptors/blackhole_140_arch.yaml" "tt_metal/core_descriptors"
)

ERRORS=0

for ((i=0; i<${#ARTIFACTS[@]}; i+=2)); do
    check_file="${ARTIFACTS[i]}"
    artifact="${ARTIFACTS[i+1]}"
    parent_dir="$(dirname "$artifact")"

    if $RESTORE && [ -e "$DEST/$check_file" ]; then
        continue
    fi

    if [ -d "$SRC/$artifact" ]; then
        mkdir -p "$DEST/$parent_dir"
        cp -a "$SRC/$artifact" "$DEST/$parent_dir/"
        echo "Copied $artifact"
    else
        echo "WARNING: $artifact not found at $SRC/$artifact"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    echo "WARNING: $ERRORS artifact(s) missing. Device runtime (JIT firmware builds) may fail."
fi
