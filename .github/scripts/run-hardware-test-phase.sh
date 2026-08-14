#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run one hardware-CI phase from a tt-lang source checkout.
#
# Env:
#   RUNS_ON identifies the hardware configuration (default: n150).
#   EXABOX_WORKER_HOME selects the valid home directory in an Exabox worker.
#   EXABOX_REPORT_DIR receives reports copied from an Exabox worker.
#
# Usage: run-hardware-test-phase.sh <phase>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE="${1:?usage: run-hardware-test-phase.sh <phase>}"

cd "$REPO_ROOT"

if [ -n "${EXABOX_WORKER_HOME:-}" ]; then
    export HOME="$EXABOX_WORKER_HOME"
    # Exabox orchestration values refer to the CPU runner and would override
    # the tt-metal installation selected by build/env/activate.
    unset TT_METAL_RUNTIME_ROOT TT_METAL_HOME LD_LIBRARY_PATH
fi

activate_build() {
    set +u
    # shellcheck disable=SC1091
    source build/env/activate
    set -u
}

collect_exabox_reports() {
    local report_root="${EXABOX_REPORT_DIR:?EXABOX_REPORT_DIR is required}"
    local host_report_dir
    case "$report_root" in
        / | '')
            echo "EXABOX_REPORT_DIR must not be / or empty" >&2
            return 2
            ;;
        /*) ;;
        *)
            echo "EXABOX_REPORT_DIR must be absolute: $report_root" >&2
            return 2
            ;;
    esac

    host_report_dir="${report_root}/$(hostname)"
    rm -rf -- "$host_report_dir"
    mkdir -p "$host_report_dir"

    if [ ! -d build/test ]; then
        echo "No hardware test reports found."
        return 0
    fi

    local report relative_report copied=0
    while IFS= read -r -d '' report; do
        relative_report="${report#build/test/}"
        mkdir -p "$host_report_dir/$(dirname "$relative_report")"
        cp "$report" "$host_report_dir/$relative_report"
        copied=$((copied + 1))
    done < <(find build/test -type f -name '*report*.xml' -print0)
    echo "Copied $copied hardware test report(s) to $host_report_dir."
}

case "$PHASE" in
    configure)
        cmake -G Ninja -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DTTLANG_USE_TOOLCHAIN=ON \
            -DTTLANG_ENABLE_PERF_TRACE=ON
        ;;
    build)
        activate_build
        cmake --build build
        cmake --build build --target ttlang-test-tools
        ;;
    install-dependencies)
        activate_build
        python3 -m pip install -r dev-requirements.txt
        ;;
    reset)
        activate_build
        .github/scripts/reset-tt-cards.sh
        ;;
    smoketest)
        activate_build
        python3 test/python/smoketest.py
        ;;
    simple-add)
        activate_build
        if [ "${RUNS_ON:-n150}" = "galaxy-bh" ]; then
            TT_VISIBLE_DEVICES=0 python test/python/simple_add.py
        else
            python test/python/simple_add.py
        fi
        ;;
    simulator)
        activate_build
        python3 -m pytest test/sim -m requires_ttnn -v -n auto \
            --tb=short --timeout=60 --timeout-method=signal \
            --junitxml=build/test/sim-report.xml
        ;;
    python-lit)
        activate_build
        .github/scripts/run-hardware-lit.sh \
            build/test/python build/test/python-lit-report
        ;;
    python-pytests)
        activate_build
        .github/scripts/run-hardware-pytests.sh \
            test/python build/test/pytest-report
        ;;
    me2e)
        activate_build
        .github/scripts/run-hardware-pytests.sh \
            test/me2e build/test/me2e-report
        ;;
    examples)
        activate_build
        bash .github/scripts/compile-and-run-examples.sh
        ;;
    tutorials)
        activate_build
        .github/scripts/run-hardware-pytests.sh \
            test/tutorial build/test/tutorial-report
        ;;
    collect-exabox-reports)
        collect_exabox_reports
        ;;
    *)
        echo "Unknown hardware test phase: $PHASE" >&2
        exit 2
        ;;
esac
