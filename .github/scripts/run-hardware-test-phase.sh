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
#   HW_TEST_WORKERS caps host-parallel test processes when set.
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

report_host_cpu_budget() {
    local available_cpu_count="unavailable"
    local cgroup_cpu_budget="unavailable"
    local cgroup_cpu_dir machine_cpu_count="unavailable" period quota
    if command -v nproc > /dev/null 2>&1; then
        if ! machine_cpu_count="$(nproc --all 2>/dev/null)"; then
            machine_cpu_count="unavailable"
        fi
        if ! available_cpu_count="$(nproc 2>/dev/null)"; then
            available_cpu_count="unavailable"
        fi
    fi
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        if read -r quota period < /sys/fs/cgroup/cpu.max; then
            cgroup_cpu_budget="v2:${quota}/${period}"
        fi
    else
        for cgroup_cpu_dir in \
            /sys/fs/cgroup/cpu \
            /sys/fs/cgroup/cpu,cpuacct; do
            if [ -r "$cgroup_cpu_dir/cpu.cfs_quota_us" ] && \
                [ -r "$cgroup_cpu_dir/cpu.cfs_period_us" ] && \
                read -r quota < "$cgroup_cpu_dir/cpu.cfs_quota_us" && \
                read -r period < "$cgroup_cpu_dir/cpu.cfs_period_us"; then
                cgroup_cpu_budget="v1:${quota}/${period}"
                break
            fi
        done
    fi
    printf 'Host CPU budget: machine=%s available=%s cgroup=%s\n' \
        "$machine_cpu_count" "$available_cpu_count" "$cgroup_cpu_budget"
}

collect_exabox_reports() {
    local report_root="${EXABOX_REPORT_DIR:?EXABOX_REPORT_DIR is required}"
    local host_report_dir
    case "$report_root" in
        /)
            echo "EXABOX_REPORT_DIR must not be /" >&2
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
        report_host_cpu_budget
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
    fabric-pytests)
        activate_build
        unset TT_VISIBLE_DEVICES
        timeout --signal=TERM --kill-after=15s 1800 \
            python3 -m pytest \
                -c build/test/pytest.ini \
                --rootdir="${REPO_ROOT}/test" \
                test/python/fabric \
                -v -x --tb=long --timeout=300 --timeout-method=thread \
                --junitxml=build/test/pytest-report-fabric-full.xml
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
        simulator_workers="${HW_TEST_WORKERS:-auto}"
        case "$simulator_workers" in
            auto) ;;
            0 | *[!0-9]*)
                echo "run-hardware-test-phase.sh: worker count must be a positive integer or auto, got '$simulator_workers'" >&2
                exit 2
                ;;
        esac
        python3 -m pytest test/sim -m requires_ttnn -v -n "$simulator_workers" \
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
