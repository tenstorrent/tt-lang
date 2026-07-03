#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/apply-wheel-patches.sh and the wheel patches it runs.

load test_helper

setup() {
    DRIVER="$SCRIPTS_DIR/apply-wheel-patches.sh"
    NUMPY_PATCH="$(dirname "$SCRIPTS_DIR")/wheel-patches/fix-numpy-requirement.sh"
    # A sandbox repo so the driver's repo-root resolution never touches the real
    # checkout (it derives repo_root from its own location).
    REPO="$BATS_TEST_TMPDIR/repo"
    mkdir -p "$REPO/.github/scripts" "$REPO/.github/wheel-patches"
    cp "$DRIVER" "$REPO/.github/scripts/apply-wheel-patches.sh"
    DRIVER_IN_REPO="$REPO/.github/scripts/apply-wheel-patches.sh"
}

@test "driver is a no-op when no patches directory exists" {
    rm -rf "$REPO/.github/wheel-patches"
    run -0 "$DRIVER_IN_REPO"
    assert_output --partial "nothing to apply"
}

@test "driver is a no-op when the patches directory is empty" {
    run -0 "$DRIVER_IN_REPO"
    assert_output --partial "nothing to apply"
}

@test "driver runs patches in sorted filename order" {
    export MARKER="$BATS_TEST_TMPDIR/order"
    cat > "$REPO/.github/wheel-patches/02-second.sh" <<'EOF'
#!/usr/bin/env bash
echo second >> "$MARKER"
EOF
    cat > "$REPO/.github/wheel-patches/01-first.sh" <<'EOF'
#!/usr/bin/env bash
echo first >> "$MARKER"
EOF
    run -0 "$DRIVER_IN_REPO"
    run cat "$MARKER"
    assert_line --index 0 "first"
    assert_line --index 1 "second"
}

@test "--target-dir patches a separate tree, not the runner's own" {
    # Patches live with the runner (REPO); the tree to patch is elsewhere and
    # need not contain the runner or patches -- the older-ref rebuild case.
    target="$BATS_TEST_TMPDIR/target"
    mkdir -p "$target"
    printf 'numpy==1.19.0\n' > "$target/requirements-runtime.txt"
    printf 'numpy==1.19.0\n' > "$REPO/requirements-runtime.txt"
    cp "$NUMPY_PATCH" "$REPO/.github/wheel-patches/fix-numpy-requirement.sh"

    run -0 "$DRIVER_IN_REPO" --target-dir "$target"

    run cat "$target/requirements-runtime.txt"
    assert_output --partial "numpy>=1.20.0"
    refute_output --partial "numpy==1.19.0"
    # The runner's own tree is untouched.
    run cat "$REPO/requirements-runtime.txt"
    assert_output --partial "numpy==1.19.0"
    refute_output --partial "numpy>=1.20.0"
}

@test "unknown argument -> exit 2" {
    run "$DRIVER_IN_REPO" --bogus
    assert_equal "$status" 2
}

@test "numpy patch rewrites a stale pin to the canonical lines" {
    cd "$BATS_TEST_TMPDIR"
    printf 'torch\nnumpy==1.19.0\nfoo>=1\n' > requirements-runtime.txt
    run -0 bash "$NUMPY_PATCH"
    run cat requirements-runtime.txt
    assert_output --partial "numpy>=1.20.0"
    assert_output --partial 'numpy<2; platform_system == "Darwin" and platform_machine == "x86_64"'
    refute_output --partial "numpy==1.19.0"
    # Non-numpy requirements are untouched.
    assert_output --partial "torch"
    assert_output --partial "foo>=1"
}

@test "numpy patch adds the canonical lines when numpy is absent" {
    cd "$BATS_TEST_TMPDIR"
    printf 'torch\n' > requirements-runtime.txt
    run -0 bash "$NUMPY_PATCH"
    run cat requirements-runtime.txt
    assert_output --partial "numpy>=1.20.0"
    assert_output --partial "torch"
}

@test "numpy patch fails when requirements-runtime.txt is absent" {
    cd "$BATS_TEST_TMPDIR"
    run bash "$NUMPY_PATCH"
    assert_failure
    assert_output --partial "not found"
}
