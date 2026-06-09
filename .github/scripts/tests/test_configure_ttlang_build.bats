#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/configure-ttlang-build.sh.

load test_helper

# Install a fake `cmake` on PATH that records its full argv (one per line) to
# $FAKE_CMAKE_ARGS. Exits 0 unconditionally. Echoes the bindir.
make_cmake_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/cmake" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    printf '%s\n' "$arg" >> "$FAKE_CMAKE_ARGS"
done
exit 0
EOF
    chmod +x "$bindir/cmake"
    echo "$bindir"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/configure-ttlang-build.sh"
    FAKE_CMAKE_ARGS="$BATS_TEST_TMPDIR/cmake_args"
    : > "$FAKE_CMAKE_ARGS"
    export FAKE_CMAKE_ARGS
    BINDIR=$(make_cmake_mock)
    export PATH="$BINDIR:$PATH"

    unset TTLANG_EXTERNAL_TT_METAL_DIR
    unset TTLANG_EXTERNAL_TT_METAL_BUILD_DIR
    unset TTLANG_PYTHON_VENV
}

@test "no arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "two arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" build extra
}

@test "minimal args (no env overrides) -> only base cmake flags" {
    run -0 "$SCRIPT" build
    run cat "$FAKE_CMAKE_ARGS"
    assert_line "-G"
    assert_line "Ninja"
    assert_line "-B"
    assert_line "build"
    assert_line "-DCMAKE_BUILD_TYPE=Release"
    assert_line "-DTTLANG_USE_TOOLCHAIN=ON"
    refute_output --partial "TTLANG_EXTERNAL_TT_METAL_DIR"
    refute_output --partial "TTLANG_EXTERNAL_TT_METAL_BUILD_DIR"
    refute_output --partial "TTLANG_PYTHON_VENV"
}

@test "passes build dir argument through to -B" {
    run -0 "$SCRIPT" build-docker
    run cat "$FAKE_CMAKE_ARGS"
    assert_line "build-docker"
    refute_line "build"
}

@test "TTLANG_EXTERNAL_TT_METAL_DIR adds -DTTLANG_EXTERNAL_TT_METAL_DIR" {
    TTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-metal run -0 "$SCRIPT" build
    run cat "$FAKE_CMAKE_ARGS"
    assert_line "-DTTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-metal"
}

@test "TTLANG_EXTERNAL_TT_METAL_BUILD_DIR adds -DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR" {
    TTLANG_EXTERNAL_TT_METAL_BUILD_DIR=/opt/tt-metal/build run -0 "$SCRIPT" build
    run cat "$FAKE_CMAKE_ARGS"
    assert_line "-DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR=/opt/tt-metal/build"
}

@test "TTLANG_PYTHON_VENV adds -DTTLANG_PYTHON_VENV" {
    TTLANG_PYTHON_VENV=/opt/venv run -0 "$SCRIPT" build
    run cat "$FAKE_CMAKE_ARGS"
    assert_line "-DTTLANG_PYTHON_VENV=/opt/venv"
}

@test "empty env vars are not forwarded as empty flags" {
    TTLANG_EXTERNAL_TT_METAL_DIR="" \
    TTLANG_EXTERNAL_TT_METAL_BUILD_DIR="" \
    TTLANG_PYTHON_VENV="" \
        run -0 "$SCRIPT" build
    run cat "$FAKE_CMAKE_ARGS"
    refute_output --partial "TTLANG_EXTERNAL_TT_METAL_DIR="
    refute_output --partial "TTLANG_EXTERNAL_TT_METAL_BUILD_DIR="
    refute_output --partial "TTLANG_PYTHON_VENV="
}

@test "all overrides set -> all three flags present" {
    TTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-metal \
    TTLANG_EXTERNAL_TT_METAL_BUILD_DIR=/opt/tt-metal/build \
    TTLANG_PYTHON_VENV=/opt/venv \
        run -0 "$SCRIPT" build
    run cat "$FAKE_CMAKE_ARGS"
    assert_line "-DTTLANG_EXTERNAL_TT_METAL_DIR=/opt/tt-metal"
    assert_line "-DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR=/opt/tt-metal/build"
    assert_line "-DTTLANG_PYTHON_VENV=/opt/venv"
}

@test "cmake failure propagates" {
    cat > "$BINDIR/cmake" <<'EOF'
#!/usr/bin/env bash
exit 7
EOF
    chmod +x "$BINDIR/cmake"
    run "$SCRIPT" build
    assert_equal "$status" 7
}
