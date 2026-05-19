#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Dispatch tests for bin/ttlang-sim-stats. Mirrors test_ttlang_sim.bats; the
# sim_stats package lives top-level (not under ttl/), so both layouts dispatch
# to `python -m sim_stats` — what differs is the PYTHONPATH that gets exported.

load test_helper

LAUNCHER="$BIN_DIR/ttlang-sim-stats"

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

setup() {
    ROOT="$BATS_TEST_TMPDIR/root"
    mkdir -p "$ROOT"
    MOCK_PY="$ROOT/mock_python"
}

@test "source layout: dispatches sim_stats with PYTHONPATH=<root>/python" {
    make_layout "$ROOT" source
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/ttlang-sim-stats" --version
    assert_line --index 0 "PYTHONPATH=$ROOT/python"
    assert_line --index 1 "argv=-m"
    assert_line --index 2 "argv=sim_stats"
    assert_line --index 3 "argv=--version"
}

@test "installed layout: dispatches sim_stats with PYTHONPATH=<root>/python_packages" {
    make_layout "$ROOT" installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/ttlang-sim-stats"
    assert_line --index 0 "PYTHONPATH=$ROOT/python_packages"
    assert_line --index 2 "argv=sim_stats"
}

@test "source layout wins when both are present" {
    make_layout "$ROOT" source installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/ttlang-sim-stats"
    assert_line --index 0 "PYTHONPATH=$ROOT/python"
}

@test "neither layout: exit 1 with both probed paths named in error" {
    make_layout "$ROOT"
    PYTHON=/bin/false PYTHONPATH="" run -1 "$ROOT/bin/ttlang-sim-stats"
    assert_output --partial "python/sim_stats"
    assert_output --partial "python_packages/sim_stats"
}

@test "existing PYTHONPATH is preserved as a suffix" {
    make_layout "$ROOT" installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="/elsewhere" run -0 "$ROOT/bin/ttlang-sim-stats"
    assert_line --index 0 "PYTHONPATH=$ROOT/python_packages:/elsewhere"
}

@test "child python exit code is propagated" {
    make_layout "$ROOT" installed
    cat > "$MOCK_PY" <<'EOF'
#!/usr/bin/env bash
exit 7
EOF
    chmod +x "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -7 "$ROOT/bin/ttlang-sim-stats"
}

@test "arguments pass through to the module" {
    make_layout "$ROOT" installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/ttlang-sim-stats" /tmp/trace.jsonl --filter foo
    assert_line --index 3 "argv=/tmp/trace.jsonl"
    assert_line --index 4 "argv=--filter"
    assert_line --index 5 "argv=foo"
}
