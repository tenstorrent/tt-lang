#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Dispatch tests for bin/ttlang-sim-stats. Mirrors test_ttlang_sim.sh; the
# sim_stats package lives top-level (not under ttl/), so both layouts dispatch
# to `python -m sim_stats` — what differs is the PYTHONPATH that gets exported.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

LAUNCHER="$(dirname "$SCRIPTS_DIR")/../bin/ttlang-sim-stats"

make_mock_python() {
    local target="$1"
    cat > "$target" <<'EOF'
#!/usr/bin/env bash
echo "PYTHONPATH=${PYTHONPATH:-}"
for a in "$@"; do
    echo "argv=$a"
done
exit 0
EOF
    chmod +x "$target"
}

make_layout() {
    local root="$1"
    shift
    mkdir -p "$root/bin"
    cp "$LAUNCHER" "$root/bin/ttlang-sim-stats"
    for layout in "$@"; do
        case "$layout" in
            source)     mkdir -p "$root/python/sim_stats" ;;
            installed)  mkdir -p "$root/python_packages/sim_stats" ;;
            *)          echo "make_layout: unknown $layout" >&2; return 1 ;;
        esac
    done
}

# --- Source layout ---
start_case "source layout: sim_stats + PYTHONPATH=<root>/python"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" source
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim-stats" --version)
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python" "source: PYTHONPATH"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '1p')" "argv=-m" "source: argv[0]"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '2p')" "argv=sim_stats" "source: argv[1]"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '3p')" "argv=--version" "source: argv[2]"
rm -rf "$root"
trap - EXIT

# --- Installed layout ---
start_case "installed layout: sim_stats + PYTHONPATH=<root>/python_packages"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim-stats")
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python_packages" "installed: PYTHONPATH"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '2p')" "argv=sim_stats" "installed: module"
rm -rf "$root"
trap - EXIT

# --- Source wins when both are present ---
start_case "source layout wins when both are present"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" source installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim-stats")
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python" "both: source PYTHONPATH"
rm -rf "$root"
trap - EXIT

# --- Neither layout: exit 1 with both paths named ---
start_case "neither layout: exit 1 + names both probed paths"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root"
set +e
err=$(PYTHON=/bin/false PYTHONPATH="" "$root/bin/ttlang-sim-stats" 2>&1 >/dev/null)
rc=$?
set -e
assert_eq "$rc" "1" "no layout: exit 1"
assert_matches "$err" "python/sim_stats" "no layout: names source path"
assert_matches "$err" "python_packages/sim_stats" "no layout: names install path"
rm -rf "$root"
trap - EXIT

# --- Existing PYTHONPATH is preserved as a suffix ---
start_case "existing PYTHONPATH is preserved as a suffix"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="/elsewhere" "$root/bin/ttlang-sim-stats")
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python_packages:/elsewhere" "preserved"
rm -rf "$root"
trap - EXIT

# --- Child python exit code is propagated ---
start_case "child python exit code is propagated"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
cat > "$root/mock_python" <<'EOF'
#!/usr/bin/env bash
exit 7
EOF
chmod +x "$root/mock_python"
set +e
PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim-stats" >/dev/null 2>&1
rc=$?
set -e
assert_eq "$rc" "7" "child exit 7 propagates"
rm -rf "$root"
trap - EXIT

# --- Arguments pass through ---
start_case "arguments pass through to the module"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim-stats" /tmp/trace.jsonl --filter foo)
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '3p')" "argv=/tmp/trace.jsonl" "argv[2]"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '4p')" "argv=--filter" "argv[3]"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '5p')" "argv=foo" "argv[4]"
rm -rf "$root"
trap - EXIT

finish_tests
