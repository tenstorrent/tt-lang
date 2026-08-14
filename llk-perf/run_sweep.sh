#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the tt-lang-scoped LLK perf sweep and collect its CSVs into perf_data/.
#
# Nothing upstream is modified. Every file here is an addition with a tt-lang
# name, staged into the tt-llk checkout for the duration of a run and removed
# afterwards, so the submodule is untouched whether or not a sweep is running.
#
# The staging itself is unavoidable: the harness finds test modules through its
# own conftest and kernel sources through `-I<tests dir>`, and neither reaches
# outside the checkout. Making our files additions rather than replacements is
# what keeps that harmless -- an upstream test can never be displaced by a
# narrowed one, which would otherwise hand a `-m perf` gather tt-lang-scoped
# results with no indication anything had changed.
#
# Usage:
#   llk-perf/run_sweep.sh [test-name ...]     # default: every test but matmul
#   llk-perf/run_sweep.sh perf_ttlang_math_matmul
#
# Requires a device and the venv at $LLK/tests/venv-llk.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLK="$REPO/third-party/tt-metal/tt_metal/tt-llk"
HERE="$REPO/llk-perf"

# Build artefacts go somewhere only this sweep uses. The harness otherwise shares
# /tmp/tt-llk-build with every other tt-llk run on the machine, and a concurrent
# one wipes it mid-build. RUNNER_TEMP is the redirect the harness already honours
# (resolve_artefacts_path in helpers/test_config.py), so this needs no change to
# it.
export RUNNER_TEMP="${RUNNER_TEMP:-/tmp/ttlang-llk-build}"

# Measure the LLK tt-lang actually compiles against, not the harness's debug
# build. LLK_ASSERT is on by default here and off in production -- tt-llk's own
# notes say production "compile[s] the macros to ((void)0)" -- and it is not
# free: bf16 datacopy measures 55.12 cycles/tile on unpack with asserts against
# 41.62 without, 24.5% inflation, tracking the assert count in each function
# (llk_unpack_AB.h has 7, llk_pack.h 8, the datacopy 2). Every row measured with
# them describes a kernel nobody runs.
export TT_LLK_DISABLE_ASSERTS=1

# perf_ttlang_math_matmul is excluded by default: 12288 variants against ~500 for
# everything else combined, and 12 of its kernel builds still fail.
DEFAULT_TESTS=(
    perf_ttlang_datacopy
    perf_ttlang_eltwise_binary_fpu
    perf_ttlang_eltwise_unary_sfpu
    perf_ttlang_eltwise_binary_sfpu
    perf_ttlang_eltwise_typecast
    perf_ttlang_eltwise_unary_sfpu_int32
    perf_ttlang_reduce
)
TESTS=("${@:-${DEFAULT_TESTS[@]}}")
[ $# -gt 0 ] && TESTS=("$@")

# Everything staged is removed on the way out, however the run ends. Only files
# this script created are listed, so upstream is never touched even on failure.
staged=()
cleanup() {
    for f in ${staged[@]+"${staged[@]}"}; do
        rm -f "$f"
    done
}
trap cleanup EXIT

echo "==> staging into $LLK (build artefacts in $RUNNER_TEMP)"
for f in "$HERE"/python_tests/*.py "$HERE"/sources/*.cpp; do
    case "$f" in
    *.py) target="$LLK/tests/python_tests/$(basename "$f")" ;;
    *) target="$LLK/tests/sources/$(basename "$f")" ;;
    esac
    if [ -e "$target" ]; then
        # A tt-lang name colliding with an upstream file means the rename scheme
        # has broken down and we would be replacing rather than adding.
        echo "    refusing to overwrite $target" >&2
        exit 1
    fi
    cp "$f" "$target"
    staged+=("$target")
done

cd "$LLK/tests"
# shellcheck disable=SC1091
source venv-llk/bin/activate
export CHIP_ARCH=${CHIP_ARCH:-blackhole}

for t in "${TESTS[@]}"; do
    echo "==> $t: compiling"
    python -m pytest --compile-producer -n "$(nproc)" -q "python_tests/$t.py"
    echo "==> $t: running on device"
    python -m pytest --compile-consumer -q "python_tests/$t.py"

    src="$LLK/perf_data/$t/$t.post.csv"
    if [ -f "$src" ]; then
        mkdir -p "$REPO/perf_data/$t"
        cp "$LLK/perf_data/$t/$t.csv" "$src" "$REPO/perf_data/$t/"
        echo "==> $t: collected $(( $(wc -l < "$src") - 1 )) rows"
    else
        echo "==> $t: NO CSV PRODUCED at $src" >&2
    fi
done

echo
echo "==> regenerate the cost table with:"
echo "    python3 scripts/gen_cost_table.py -o lib/OpCost/CostTableBlackhole.inc"
