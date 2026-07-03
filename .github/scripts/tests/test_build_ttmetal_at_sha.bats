#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/build-ttmetal-at-sha.sh.
#
# The tt-metal build itself is heavy and CI-only, so `cmake` is mocked on PATH
# to fabricate the ttnn extensions install-ttmetal.sh expects. Everything else
# (clone/checkout/date/tag detection, idempotency, install layout, output
# emission) runs against the real script and the real install-ttmetal.sh.

load test_helper

# Build a synthetic tt-metal remote: a git repo whose working tree mirrors the
# minimal tt-metal source layout install-ttmetal.sh accepts. HEAD~1 is untagged;
# HEAD carries a v-tag. Echoes the repo path.
make_ttmetal_remote() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/ttmetal-remote.XXXXXX")
    (
        cd "$dir"
        git init -q -b main
        git config user.email t@t
        git config user.name t
        mkdir -p ttnn/ttnn ttnn/cpp tt_metal/api tools/tracy
        echo "from . import _ttnn" > ttnn/ttnn/__init__.py
        echo "version = '0.x'" > ttnn/ttnn/version.py
        echo "// cpp header" > ttnn/cpp/placeholder.h
        echo "// header" > tt_metal/api/sample.h
        echo "tracy_module" > tools/tracy/__init__.py
        git add -A
        git commit -q -m "untagged base"
        echo "second" >> tt_metal/api/sample.h
        git add -A
        git commit -q -m "release commit"
        git tag v99.88.77
    )
    echo "$dir"
}

# Install a fake `cmake` on PATH. On `--build ... --target ttnn ttnncpp` it
# fabricates the ttnn extensions and a shared library in the build dir; the
# configure and precompile-fw invocations are no-ops. Records each `--build`
# invocation to $FAKE_CMAKE_LOG. Echoes the bindir.
make_cmake_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/cmake" <<'EOF'
#!/usr/bin/env bash
build_dir=""
is_build=0
targets=""
prev=""
for arg in "$@"; do
    case "$arg" in
        --build) is_build=1 ;;
    esac
    if [[ "$prev" == "--build" || "$prev" == "-B" ]]; then
        build_dir="$arg"
    fi
    if [[ "$prev" == "--target" || -n "$targets" ]]; then
        targets="$targets $arg"
    fi
    prev="$arg"
done
if [[ "$is_build" -eq 1 ]]; then
    printf '%s\n' "$*" >> "$FAKE_CMAKE_LOG"
    if [[ "$targets" == *ttnn* ]]; then
        mkdir -p "$build_dir/ttnn" "$build_dir/lib"
        printf '\x7fELF fresh ttnn'    > "$build_dir/ttnn/_ttnn.so"
        printf '\x7fELF fresh ttnncpp' > "$build_dir/ttnn/_ttnncpp.so"
        printf '\x7fELF libdevice'     > "$build_dir/lib/libdevice.so"
    fi
else
    mkdir -p "$build_dir"
fi
exit 0
EOF
    chmod +x "$bindir/cmake"
    echo "$bindir"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/build-ttmetal-at-sha.sh"
    REMOTE=$(make_ttmetal_remote)
    export TTMETAL_REMOTE_URL="$REMOTE"
    SHA=$(git -C "$REMOTE" rev-parse HEAD)
    BASE_SHA=$(git -C "$REMOTE" rev-parse HEAD~1)
    SCRATCH="$BATS_TEST_TMPDIR/scratch"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"
    FAKE_CMAKE_LOG="$BATS_TEST_TMPDIR/cmake_calls"
    : > "$FAKE_CMAKE_LOG"
    export FAKE_CMAKE_LOG
    BINDIR=$(make_cmake_mock)
    export PATH="$BINDIR:$PATH"
    # No real toolchain venv in the test environment.
    export TTLANG_PYTHON_VENV="$BATS_TEST_TMPDIR/no-venv"
}

@test "missing --sha -> usage error (exit 2)" {
    run -2 "$SCRIPT" --scratch-dir "$SCRATCH"
}

@test "unknown argument -> usage error (exit 2)" {
    run -2 "$SCRIPT" --sha "$SHA" --bogus
}

@test "--no-build emits source_dir and ttmetal_date without building" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$SHA" --scratch-dir "$SCRATCH" --no-build

    run cat "$GH_OUT"
    assert_output --partial "source_dir=$SCRATCH/src"
    assert_output --partial "ttmetal_date="
    refute_output --partial "install_dir="

    # No build occurred and no install tree was produced.
    run cat "$FAKE_CMAKE_LOG"
    assert_output ""
    [ ! -d "$SCRATCH/install" ]
}

@test "--sha trims workflow-dispatch whitespace" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "   $SHA   " --scratch-dir "$SCRATCH" --no-build

    run cat "$GH_OUT"
    assert_output --partial "source_dir=$SCRATCH/src"
    assert_output --partial "ttmetal_date="
    # The emitted sha is the trimmed value, not the padded input.
    assert_output --partial "sha=$SHA"
    assert_output --partial "short=${SHA:0:7}"

    run git -C "$SCRATCH/src" rev-parse HEAD
    assert_output "$SHA"
}

@test "default mode builds and installs, emitting install_dir" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$SHA" --scratch-dir "$SCRATCH"

    run cat "$GH_OUT"
    assert_output --partial "install_dir=$SCRATCH/install"
    assert_output --partial "ttmetal_date="

    # Install produced the ttnn extension from the (mocked) build, not the src.
    [ -f "$SCRATCH/install/python_packages/ttnn/ttnn/_ttnn.so" ]
    run cat "$SCRATCH/install/python_packages/ttnn/ttnn/_ttnn.so"
    assert_output --partial "fresh ttnn"

    run cat "$SCRATCH/build/.ttmetal-source-sha"
    assert_output "$SHA"
}

@test "ttmetal_date is the SHA committer date" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$SHA" --scratch-dir "$SCRATCH" --no-build
    expected=$(git -C "$REMOTE" show -s --format=%cI "$SHA")
    run cat "$GH_OUT"
    assert_output --partial "ttmetal_date=$expected"
}

@test "a bare (untagged) SHA builds without a release tag" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$BASE_SHA" --scratch-dir "$SCRATCH"
    assert_success
    [ -f "$SCRATCH/install/python_packages/ttnn/ttnn/_ttnn.so" ]
}

@test "second run reuses the existing build (idempotent scratch)" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$SHA" --scratch-dir "$SCRATCH"
    run -0 "$SCRIPT" --sha "$SHA" --scratch-dir "$SCRATCH"
    assert_output --partial "Reusing existing tt-metal build"
}

@test "different SHA rebuilds an existing scratch build" {
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$SHA" --scratch-dir "$SCRATCH"
    GITHUB_OUTPUT="$GH_OUT" run -0 "$SCRIPT" --sha "$BASE_SHA" --scratch-dir "$SCRATCH"
    assert_output --partial "Discarding tt-metal build"

    run cat "$SCRATCH/build/.ttmetal-source-sha"
    assert_output "$BASE_SHA"

    build_call_count="$(wc -l < "$FAKE_CMAKE_LOG")"
    [ "$build_call_count" -eq 4 ]
}
