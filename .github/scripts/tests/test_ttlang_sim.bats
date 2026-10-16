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

make_isolated_path() {
    mkdir -p "$ROOT/path"
    local tool
    for tool in bash cat dirname; do
        ln -s "$(command -v "$tool")" "$ROOT/path/$tool"
    done
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
    unset TTLANG_SIM_BACKEND TTLANG_EMULE_RUNNER
    ROOT="$BATS_TEST_TMPDIR/root"
    mkdir -p "$ROOT"
    MOCK_PY="$ROOT/mock_python"
}

make_mock_emule_runner() {
    local target="$1"
    cat > "$target" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do
    echo "argv=$a"
done
exit 0
EOF
    chmod +x "$target"
}

@test "action-looking script arguments after separator remain literal" {
    make_layout "$ROOT" source
    local runner="$ROOT/emule-runner"
    make_mock_emule_runner "$runner"
    TTLANG_EMULE_RUNNER="$runner" run -0 "$ROOT/bin/tt-lang-sim" \
        program.py --backend=emule -- --test --setup --examples --smoke-test
    assert_line --index 0 "argv=program.py"
    assert_line --index 1 "argv=--test"
    assert_line --index 2 "argv=--setup"
    assert_line --index 3 "argv=--examples"
    assert_line --index 4 "argv=--smoke-test"
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

@test "python3-only PATH dispatches the Python backend" {
    make_layout "$ROOT" installed
    make_isolated_path
    make_mock_python "$ROOT/path/python3"
    run -0 env -u PYTHON PATH="$ROOT/path" PYTHONPATH="" \
        "$ROOT/bin/tt-lang-sim" --backend=python program.py
    assert_line --index 0 "PYTHONPATH=$ROOT/python_packages"
    assert_line --index 2 "argv=ttl.sim.ttlang_sim"
    assert_line --index 3 "argv=program.py"
}

@test "empty PYTHON uses python3 from PATH in a source checkout" {
    make_layout "$ROOT" source
    make_isolated_path
    make_mock_python "$ROOT/path/python3"
    run -0 env PATH="$ROOT/path" PYTHON="" PYTHONPATH="" \
        "$ROOT/bin/tt-lang-sim" program.py
    assert_line --index 0 "PYTHONPATH=$ROOT/python"
    assert_line --index 2 "argv=sim.ttlang_sim"
    assert_line --index 3 "argv=program.py"
}

@test "PATH python is preferred over python3 for activated environments" {
    make_layout "$ROOT" source
    make_isolated_path
    make_mock_python "$ROOT/path/python"
    ln -s /bin/false "$ROOT/path/python3"
    run -0 env -u PYTHON PATH="$ROOT/path" PYTHONPATH="" \
        "$ROOT/bin/tt-lang-sim" program.py
    assert_line --index 2 "argv=sim.ttlang_sim"
}

@test "PYTHON env override is honored over PATH-resolved interpreters" {
    make_layout "$ROOT" installed
    make_isolated_path
    ln -s /bin/false "$ROOT/path/python"
    ln -s /bin/false "$ROOT/path/python3"
    MOCK_PY="$ROOT/custom python"
    make_mock_python "$MOCK_PY"
    run -0 env PATH="$ROOT/path" PYTHON="$MOCK_PY" PYTHONPATH="" \
        "$ROOT/bin/tt-lang-sim"
    assert_line --index 2 "argv=ttl.sim.ttlang_sim"
}

@test "missing PYTHON override does not fall back to PATH interpreters" {
    make_layout "$ROOT" source
    make_isolated_path
    make_mock_python "$ROOT/path/python3"
    run ! env PATH="$ROOT/path" PYTHON="$ROOT/missing-python" \
        "$ROOT/bin/tt-lang-sim" program.py
    assert_output --partial "$ROOT/missing-python"
}

@test "missing Python interpreters report how to set an override" {
    make_layout "$ROOT" source
    make_isolated_path
    run -1 env -u PYTHON PATH="$ROOT/path" "$ROOT/bin/tt-lang-sim" program.py
    assert_line "tt-lang-sim: cannot find python or python3 on PATH."
    assert_line "  Set PYTHON to the Python interpreter to use."
}

@test "emule dispatch, help and version work without Python interpreters on PATH" {
    make_layout "$ROOT" source
    make_isolated_path
    local runner="$ROOT/emule-runner"
    make_mock_emule_runner "$runner"
    run -0 env -u PYTHON PATH="$ROOT/path" TTLANG_EMULE_RUNNER="$runner" \
        "$ROOT/bin/tt-lang-sim" --backend=emule program.py
    assert_output "argv=program.py"
    run -0 env -u PYTHON PATH="$ROOT/path" \
        "$ROOT/bin/tt-lang-sim" --backend=emule --help
    assert_output --partial "Usage: tt-lang-sim --backend=emule SCRIPT.py"
    run -0 env -u PYTHON PATH="$ROOT/path" \
        "$ROOT/bin/tt-lang-sim" --backend=emule --version
    assert_output "tt-lang-sim emule (checkout unknown)"
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

@test "emule backend dispatches to its runner and removes backend option" {
    make_layout "$ROOT" source
    local runner="$ROOT/emule-runner"
    make_mock_emule_runner "$runner"
    TTLANG_EMULE_RUNNER="$runner" run -0 "$ROOT/bin/tt-lang-sim" \
        "two words.py" --backend emule --script-option
    assert_line --index 0 "argv=two words.py"
    assert_line --index 1 "argv=--script-option"
}

@test "emule backend accepts the environment default" {
    make_layout "$ROOT" source
    local runner="$ROOT/emule-runner"
    make_mock_emule_runner "$runner"
    TTLANG_SIM_BACKEND=emule TTLANG_EMULE_RUNNER="$runner" \
        run -0 "$ROOT/bin/tt-lang-sim" program.py
    assert_output "argv=program.py"
}

@test "emule environment help does not require host Python or a runner" {
    make_layout "$ROOT" source
    TTLANG_SIM_BACKEND=emule TTLANG_EMULE_RUNNER=/bin/false \
        PYTHON=/bin/false \
        run -0 "$ROOT/bin/tt-lang-sim" --help
    assert_output --partial "Usage: tt-lang-sim --backend=emule SCRIPT.py"
    assert_output --partial "./scripts/install-tt-lang-emule.sh"
}

@test "explicit emule help does not require host Python or a runner" {
    make_layout "$ROOT" source
    local option
    for option in -h --help; do
        TTLANG_EMULE_RUNNER=/bin/false PYTHON=/bin/false \
            run -0 "$ROOT/bin/tt-lang-sim" --backend=emule "$option"
        assert_output --partial "Usage: tt-lang-sim --backend=emule SCRIPT.py"
    done
}

@test "emule without a program reports usage without host Python" {
    make_layout "$ROOT" source
    TTLANG_EMULE_RUNNER=/bin/false PYTHON=/bin/false \
        run -1 "$ROOT/bin/tt-lang-sim" --backend emule
    assert_output --partial "Usage: tt-lang-sim --backend=emule SCRIPT.py"
}

@test "emule version reports the source checkout without host Python" {
    make_layout "$ROOT" source
    git -C "$ROOT" init -q
    git -C "$ROOT" add .
    git -C "$ROOT" -c user.name=Test -c user.email=test@example.com \
        commit -qm fixture
    local revision
    revision="$(git -C "$ROOT" rev-parse --short=12 HEAD)"
    TTLANG_EMULE_RUNNER=/bin/false PYTHON=/bin/false \
        run -0 "$ROOT/bin/tt-lang-sim" --backend=emule --version
    assert_output "tt-lang-sim emule (checkout $revision)"
    TTLANG_SIM_BACKEND=emule TTLANG_EMULE_RUNNER=/bin/false PYTHON=/bin/false \
        run -0 "$ROOT/bin/tt-lang-sim" --version
    assert_output "tt-lang-sim emule (checkout $revision)"
}

@test "emule preserves program help and version arguments" {
    make_layout "$ROOT" source
    local runner="$ROOT/emule-runner"
    make_mock_emule_runner "$runner"
    TTLANG_EMULE_RUNNER="$runner" PYTHON=/bin/false \
        run -0 "$ROOT/bin/tt-lang-sim" --backend=emule program.py -- --help --version
    assert_line --index 0 "argv=program.py"
    assert_line --index 1 "argv=--help"
    assert_line --index 2 "argv=--version"
}

@test "backend-looking script argument after separator is preserved" {
    make_layout "$ROOT" source
    make_mock_python "$MOCK_PY"
    PYTHON="$MOCK_PY" PYTHONPATH="" run -0 "$ROOT/bin/tt-lang-sim" \
        program.py -- --backend emule
    assert_line "argv=--"
    assert_line "argv=--backend"
    assert_line "argv=emule"
}

@test "emule removes the separator before passing script arguments" {
    make_layout "$ROOT" source
    local runner="$ROOT/emule-runner"
    make_mock_emule_runner "$runner"
    TTLANG_SIM_BACKEND=emule TTLANG_EMULE_RUNNER="$runner" \
        run -0 "$ROOT/bin/tt-lang-sim" program.py -- --backend emule
    assert_line --index 0 "argv=program.py"
    assert_line --index 1 "argv=--backend"
    assert_line --index 2 "argv=emule"
    refute_line "argv=--"
}

@test "unknown backend is rejected before dispatch" {
    make_layout "$ROOT" source
    make_isolated_path
    run -2 env -u PYTHON PATH="$ROOT/path" \
        "$ROOT/bin/tt-lang-sim" program.py --backend unknown
    assert_output --partial "unknown backend 'unknown'"
}

@test "backend without a value is rejected" {
    make_layout "$ROOT" source
    run -2 "$ROOT/bin/tt-lang-sim" program.py --backend
    assert_output --partial "--backend requires python or emule"
}
