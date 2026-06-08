#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Dispatch tests for bin/tt-lang-sim. The launcher's effect (what `python -m`
# would have been called with) is captured by a mock PYTHON that prints its
# argv and PYTHONPATH, instead of running real Python.

load test_helper

LAUNCHER="$BIN_DIR/tt-lang-sim"

# Mock-python: prints PYTHONPATH and remaining args (one per line), exits 0.
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

# Build a synthetic root containing bin/tt-lang-sim plus optional layout
# markers. Args after $1 are one or more of: "source", "installed".
make_layout() {
    local root="$1"
    shift
    mkdir -p "$root/bin"
    cp "$LAUNCHER" "$root/bin/tt-lang-sim"
    for layout in "$@"; do
        case "$layout" in
            source)     mkdir -p "$root/python/sim"
                        : > "$root/python/sim/ttlang_sim.py" ;;
            installed)  mkdir -p "$root/python_packages/ttl/sim"
                        : > "$root/python_packages/ttl/sim/ttlang_sim.py" ;;
            *)          echo "make_layout: unknown layout $layout" >&2; return 1 ;;
        esac
    done
}

setup() {
    ROOT="$BATS_TEST_TMPDIR/root"
    mkdir -p "$ROOT"
    MOCK_PY="$ROOT/mock_python"
}

@test "source layout: dispatches sim.ttlang_sim with PYTHONPATH=<root>/python" {
    make_layout "$ROOT" source
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/tt-lang-sim" --help foo
    assert_line --index 0 "PYTHONPATH=$ROOT/python"
    assert_line --index 1 "argv=-m"
    assert_line --index 2 "argv=sim.ttlang_sim"
    assert_line --index 3 "argv=--help"
    assert_line --index 4 "argv=foo"
}

@test "installed layout: dispatches ttl.sim.ttlang_sim with PYTHONPATH=<root>/python_packages" {
    make_layout "$ROOT" installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/tt-lang-sim" --help
    assert_line --index 0 "PYTHONPATH=$ROOT/python_packages"
    assert_line --index 2 "argv=ttl.sim.ttlang_sim"
}

@test "source layout wins when both layouts coexist" {
    make_layout "$ROOT" source installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/tt-lang-sim"
    assert_line --index 2 "argv=sim.ttlang_sim"
}

@test "neither layout: exit 1 with both probed paths named in error" {
    make_layout "$ROOT"
    PYTHON=/bin/false PYTHONPATH="" run -1 "$ROOT/bin/tt-lang-sim"
    assert_output --partial "python/sim/ttlang_sim.py"
    assert_output --partial "python_packages/ttl/sim/ttlang_sim.py"
}

@test "existing PYTHONPATH is preserved as a suffix" {
    make_layout "$ROOT" installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="/preexisting/path" run -0 "$ROOT/bin/tt-lang-sim"
    assert_line --index 0 "PYTHONPATH=$ROOT/python_packages:/preexisting/path"
}

@test "PYTHON env override is honored over a PATH-resolved python" {
    make_layout "$ROOT" installed
    # PATH shim that would exit 99 if used.
    mkdir -p "$ROOT/path-shim"
    cat > "$ROOT/path-shim/python" <<'EOF'
#!/usr/bin/env bash
exit 99
EOF
    chmod +x "$ROOT/path-shim/python"
    make_mock_python "$MOCK_PY"
    PATH="$ROOT/path-shim:$PATH" PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/tt-lang-sim"
}

@test "child python exit code is propagated" {
    make_layout "$ROOT" installed
    cat > "$MOCK_PY" <<'EOF'
#!/usr/bin/env bash
exit 7
EOF
    chmod +x "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -7 "$ROOT/bin/tt-lang-sim"
}

@test "arguments with spaces pass through unmangled" {
    make_layout "$ROOT" installed
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/tt-lang-sim" "two words" "--opt=value with space"
    assert_line --index 3 "argv=two words"
    assert_line --index 4 "argv=--opt=value with space"
}
