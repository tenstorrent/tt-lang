#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run example scripts on TT hardware using the real Python compiler stack
# (build/env/activate, `import ttl`, ttnn device). Not pytest; each script is a
# standalone driver.
#
# Discovers candidates under examples/ (top-level *.py only) and
# examples/tutorial/ (recursive *.py) that contain the substring @ttl.kernel.
#
# Optional opt-out (first 80 lines of the file):
#     # <description>
#     #
#     # TTLANG_HARDWARE_CI: skip-compiler
# Use skip-compiler for simulator-only scripts or any example that must not run
# in this hardware compiler batch.
#
# Usage: from repo root after build, with venv active:
#   source build/env/activate
#   bash .github/scripts/compile-and-run-examples.sh
#
# Optional first argument: repo root (default: current directory).

set -euo pipefail

ROOT="$(cd "${1:-.}" && pwd)"
SCAN_LINES=80
SKIP_TAG="TTLANG_HARDWARE_CI: skip-compiler"

file_has_tag() {
  local path="$1"
  local tag="$2"
  head -n "${SCAN_LINES}" "${path}" | grep -Fq "# ${tag}"
}

has_ttl_kernel() {
  grep -Fq "@ttl.kernel" "$1"
}

collect_scripts() {
  shopt -s nullglob
  local f
  for f in "${ROOT}/examples"/*.py; do
    [[ -f "$f" ]] || continue
    has_ttl_kernel "$f" || continue
    printf '%s\n' "${f#"${ROOT}/"}"
  done
  shopt -u nullglob

  if [[ -d "${ROOT}/examples/tutorial" ]]; then
    while IFS= read -r -d '' f; do
      has_ttl_kernel "$f" || continue
      printf '%s\n' "${f#"${ROOT}/"}"
    done < <(find "${ROOT}/examples/tutorial" -type f -name "*.py" -print0)
  fi
}

mapfile -t SCRIPTS < <(collect_scripts | sort -u)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "compile-and-run-examples.sh: no examples matched (@ttl.kernel in examples/*.py or examples/tutorial/**/*.py)" >&2
  exit 1
fi

for script in "${SCRIPTS[@]}"; do
  path="${ROOT}/${script}"
  if [[ ! -f "${path}" ]]; then
    echo "error: script not found: ${script}" >&2
    exit 1
  fi

  if file_has_tag "${path}" "${SKIP_TAG}"; then
    echo "=== SKIP (hardware CI compiler step): ${script}  # ${SKIP_TAG} ==="
    continue
  fi

  echo "=== python3 ${script} ==="
  (cd "${ROOT}" && python3 "${script}")
done

echo "compile-and-run-examples.sh: done"
