#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Fail if a built wheel exports STB_GNU_UNIQUE symbols.
#
# GCC emits LLVM/MLIR singletons as STB_GNU_UNIQUE, which glibc merges
# process-wide even under RTLD_LOCAL. Importing multiple MLIR extension modules
# can then abort with duplicate option registration. The build compiles with
# -fno-gnu-unique to keep them weak; this guards against that regressing.
#
# Usage: assert-no-gnu-unique.sh <wheel>

set -euo pipefail

wheel="${1:?usage: assert-no-gnu-unique.sh <wheel>}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

unzip -q -o "$wheel" -d "$workdir"

violations=0
while IFS= read -r so; do
    count="$(nm -D --defined-only "$so" 2>/dev/null | awk '$2 == "u"' | wc -l)"
    if [ "$count" -ne 0 ]; then
        echo "ERROR: $(basename "$so") exports $count STB_GNU_UNIQUE symbol(s):" >&2
        nm -D --defined-only "$so" 2>/dev/null | awk '$2 == "u" {print "  " $0}' | head -5 >&2
        violations=$((violations + count))
    fi
done < <(find "$workdir" -name '*.so' -type f)

if [ "$violations" -ne 0 ]; then
    echo "Wheel exports gnu-unique symbols; build with -fno-gnu-unique." >&2
    exit 1
fi

echo "gnu-unique check passed: no exported STB_GNU_UNIQUE symbols in $(basename "$wheel")"
