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
# examples/tutorial/ (recursive *.py) that contain the substring @ttl.operation.
#
# Tags (first 80 lines of the file):
#     # TTLANG_HARDWARE_CI: skip-compiler
#         Do not run at all (simulator-only, negative tests, etc.).
#     # TTLANG_HARDWARE_CI: xfail-compiler
#         Run, but expect a non-zero exit. XFAIL if it fails as expected;
#         XPASS (unexpected pass) if it succeeds -- treated as a failure so
#         the tag can be removed.
#
# Usage: from repo root after build, with venv active:
#   source build/env/activate
#   bash .github/scripts/compile-and-run-examples.sh
#
# Optional first argument: repo root (default: current directory).

set -uo pipefail

ROOT="$(cd "${1:-.}" && pwd)"
SCAN_LINES=80
SKIP_TAG="TTLANG_HARDWARE_CI: skip-compiler"
XFAIL_TAG="TTLANG_HARDWARE_CI: xfail-compiler"

# ── helpers ──────────────────────────────────────────────────────────────────

file_has_tag() {
  local path="$1" tag="$2"
  head -n "${SCAN_LINES}" "${path}" | grep -Fq "# ${tag}"
}

has_ttl_kernel() {
  grep -Fq "@ttl.operation" "$1"
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

  for subdir in tutorial errors; do
    if [[ -d "${ROOT}/examples/${subdir}" ]]; then
      while IFS= read -r -d '' f; do
        has_ttl_kernel "$f" || continue
        printf '%s\n' "${f#"${ROOT}/"}"
      done < <(find "${ROOT}/examples/${subdir}" -type f -name "*.py" -print0)
    fi
  done
}

# ── discover ─────────────────────────────────────────────────────────────────

mapfile -t SCRIPTS < <(collect_scripts | sort -u)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "compile-and-run-examples.sh: no examples matched (@ttl.operation in examples/*.py or examples/tutorial/**/*.py)" >&2
  exit 1
fi

# ── run ──────────────────────────────────────────────────────────────────────

declare -a RESULTS=()
N_PASS=0  N_FAIL=0  N_SKIP=0  N_XFAIL=0  N_XPASS=0

for script in "${SCRIPTS[@]}"; do
  path="${ROOT}/${script}"
  if [[ ! -f "${path}" ]]; then
    echo "error: script not found: ${script}" >&2
    exit 1
  fi

  # ── skip ──
  if file_has_tag "${path}" "${SKIP_TAG}"; then
    RESULTS+=("${script} ... SKIP")
    (( N_SKIP++ ))
    continue
  fi

  # ── determine expectation ──
  expect_fail=false
  if file_has_tag "${path}" "${XFAIL_TAG}"; then
    expect_fail=true
  fi

  # ── execute ──
  echo "--- ${script} ---"
  rc=0
  (cd "${ROOT}" && python3 "${script}") || rc=$?

  # ── classify result ──
  if [[ ${rc} -eq 0 ]]; then
    if ${expect_fail}; then
      RESULTS+=("${script} ... XPASS (unexpected pass)")
      (( N_XPASS++ ))
    else
      RESULTS+=("${script} ... PASS")
      (( N_PASS++ ))
    fi
  else
    if ${expect_fail}; then
      RESULTS+=("${script} ... XFAIL (expected failure, rc=${rc})")
      (( N_XFAIL++ ))
    else
      RESULTS+=("${script} ... FAIL (rc=${rc})")
      (( N_FAIL++ ))
    fi
  fi
done

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  compile-and-run-examples: results"
echo "========================================"
for r in "${RESULTS[@]}"; do
  echo "  ${r}"
done
echo "----------------------------------------"
printf "  PASS: %d  FAIL: %d  SKIP: %d  XFAIL: %d  XPASS: %d  Total: %d\n" \
  "${N_PASS}" "${N_FAIL}" "${N_SKIP}" "${N_XFAIL}" "${N_XPASS}" "${#SCRIPTS[@]}"
echo "========================================"

if [[ ${N_FAIL} -gt 0 || ${N_XPASS} -gt 0 ]]; then
  exit 1
fi
