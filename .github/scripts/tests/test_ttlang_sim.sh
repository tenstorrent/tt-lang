#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Dispatch tests for bin/ttlang-sim. The launcher's effect (what `python -m`
# would have been called with) is captured by a mock PYTHON that prints its
# argv and PYTHONPATH, instead of running real Python.

set -uo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

LAUNCHER="$(dirname "$SCRIPTS_DIR")/../bin/ttlang-sim"

# Mock-python script: prints PYTHONPATH and remaining args (one per line).
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

# Build a synthetic "_REPO_ROOT" with either or both layouts. Args after the
# first are layout markers: "source", "installed", or both.
make_layout() {
    local root="$1"
    shift
    mkdir -p "$root/bin"
    cp "$LAUNCHER" "$root/bin/ttlang-sim"
    for layout in "$@"; do
        case "$layout" in
            source)
                mkdir -p "$root/python/sim"
                : > "$root/python/sim/ttlang_sim.py"
                ;;
            installed)
                mkdir -p "$root/python_packages/ttl/sim"
                : > "$root/python_packages/ttl/sim/ttlang_sim.py"
                ;;
            *)
                echo "make_layout: unknown layout $layout" >&2
                return 1
                ;;
        esac
    done
}

# --- Case: source-tree layout dispatches via `sim.ttlang_sim` ---
start_case "source layout: sim.ttlang_sim + PYTHONPATH=<root>/python"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" source
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim" --help foo)
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python" "source: PYTHONPATH set"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '1p')" "argv=-m" "source: argv[0] = -m"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '2p')" "argv=sim.ttlang_sim" "source: argv[1] = sim.ttlang_sim"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '3p')" "argv=--help" "source: argv[2] = --help"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '4p')" "argv=foo" "source: argv[3] = foo"
rm -rf "$root"
trap - EXIT

# --- Case: installed layout dispatches via `ttl.sim.ttlang_sim` ---
start_case "installed layout: ttl.sim.ttlang_sim + PYTHONPATH=<root>/python_packages"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim" --help)
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python_packages" "installed: PYTHONPATH set"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '2p')" "argv=ttl.sim.ttlang_sim" "installed: module = ttl.sim.ttlang_sim"
rm -rf "$root"
trap - EXIT

# --- Case: source layout wins when both are present ---
start_case "source layout wins when both layouts coexist"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" source installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim")
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '2p')" "argv=sim.ttlang_sim" "both layouts -> source wins"
rm -rf "$root"
trap - EXIT

# --- Case: neither layout -> exit 1 with informative error ---
start_case "neither layout: exit 1 + names both probed paths"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root"
set +e
err=$(PYTHON=/bin/false PYTHONPATH="" "$root/bin/ttlang-sim" 2>&1 >/dev/null)
rc=$?
set -e
assert_eq "$rc" "1" "no layout: exit 1"
assert_matches "$err" "python/sim/ttlang_sim.py" "no layout: names source path"
assert_matches "$err" "python_packages/ttl/sim/ttlang_sim.py" "no layout: names install path"
rm -rf "$root"
trap - EXIT

# --- Case: existing PYTHONPATH is preserved (prepend, don't replace) ---
start_case "existing PYTHONPATH is preserved as a suffix"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="/preexisting/path" "$root/bin/ttlang-sim")
assert_eq "$(echo "$out" | grep '^PYTHONPATH=' | head -1)" "PYTHONPATH=$root/python_packages:/preexisting/path" "PYTHONPATH preserved"
rm -rf "$root"
trap - EXIT

# --- Case: PYTHON override is respected ---
start_case "PYTHON env override is honored over a PATH-resolved python"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
# Mock 'python' on PATH that exits 99 (would be visible if it ran).
mkdir -p "$root/path-shim"
cat > "$root/path-shim/python" <<'EOF'
#!/usr/bin/env bash
exit 99
EOF
chmod +x "$root/path-shim/python"
make_mock_python "$root/mock_python"
set +e
PATH="$root/path-shim:$PATH" PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim" >/dev/null
rc=$?
set -e
assert_eq "$rc" "0" "PYTHON override used (not the PATH shim that would exit 99)"
rm -rf "$root"
trap - EXIT

# --- Case: arguments with spaces / special chars pass through unmangled ---
start_case "arguments with spaces pass through"
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
make_layout "$root" installed
make_mock_python "$root/mock_python"
out=$(PYTHON="$root/mock_python" PYTHONPATH="" "$root/bin/ttlang-sim" "two words" "--opt=value with space")
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '3p')" "argv=two words" "argv preserved (positional)"
assert_eq "$(echo "$out" | grep '^argv=' | sed -n '4p')" "argv=--opt=value with space" "argv preserved (option)"
rm -rf "$root"
trap - EXIT

finish_tests
