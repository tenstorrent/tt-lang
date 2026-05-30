#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/probe-docker-image.sh.

load test_helper

IRD_IMAGE_BASE="ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04"
BARE_TAG="v99.99.99"
UPLIFT_TAG="v99.99.99-uplift-abcd1234"
RC_TAG="v99.99.99-rc1"
DEV_TAG="v99.99.99-dev20260515"

# Install a fake `docker` on PATH whose `manifest inspect` exit status is
# controlled by FAKE_DOCKER_MISSING (1 -> exit 1, else exit 0). Records the
# full argv to $FAKE_DOCKER_ARGS so tests can assert on the image reference.
# Echoes the bindir.
make_docker_mock() {
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/docker" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${FAKE_DOCKER_ARGS:-}" ]]; then
    printf '%s\n' "$*" >> "$FAKE_DOCKER_ARGS"
fi
if [[ "$1" == "manifest" && "$2" == "inspect" ]]; then
    [[ "${FAKE_DOCKER_MISSING:-0}" == "1" ]] && exit 1
    exit 0
fi
exit 0
EOF
    chmod +x "$bindir/docker"
    echo "$bindir"
}

# Extract needs_rebuild from GITHUB_OUTPUT (empty string if absent).
read_needs_rebuild() {
    grep '^needs_rebuild=' "$1" 2>/dev/null | sed 's/^needs_rebuild=//' || true
}

setup() {
    SCRIPT="$SCRIPTS_DIR/probe-docker-image.sh"
    GH_OUT="$BATS_TEST_TMPDIR/gh_out"
    : > "$GH_OUT"
    BINDIR=$(make_docker_mock)
    export PATH="$BINDIR:$PATH"
    export GITHUB_OUTPUT="$GH_OUT"
}

# --- Image present -> needs_rebuild=false, exit 0 ---

@test "bare tag, image present -> needs_rebuild=false" {
    FAKE_DOCKER_MISSING=0 run -0 "$SCRIPT" "$BARE_TAG"
    assert_equal "$(read_needs_rebuild "$GH_OUT")" "false"
}

@test "uplift tag, image present -> needs_rebuild=false" {
    FAKE_DOCKER_MISSING=0 run -0 "$SCRIPT" "$UPLIFT_TAG"
    assert_equal "$(read_needs_rebuild "$GH_OUT")" "false"
}

# --- Image missing, uplift form -> needs_rebuild=true, exit 0 ---

@test "uplift tag, image missing -> needs_rebuild=true" {
    FAKE_DOCKER_MISSING=1 run -0 "$SCRIPT" "$UPLIFT_TAG"
    assert_equal "$(read_needs_rebuild "$GH_OUT")" "true"
}

# --- Image missing, bare release form -> refuse with exit 1 ---

@test "bare release tag, image missing -> refuse (exit 1)" {
    FAKE_DOCKER_MISSING=1 run -1 "$SCRIPT" "$BARE_TAG"
    assert_equal "$(read_needs_rebuild "$GH_OUT")" ""
}

@test "rc release tag, image missing -> refuse (exit 1)" {
    FAKE_DOCKER_MISSING=1 run -1 "$SCRIPT" "$RC_TAG"
}

@test "dev release tag, image missing -> refuse (exit 1)" {
    FAKE_DOCKER_MISSING=1 run -1 "$SCRIPT" "$DEV_TAG"
}

# --- Mock invoked with the expected image reference ---

@test "probes the ird image at the given tag" {
    ARGS_FILE="$BATS_TEST_TMPDIR/args"
    FAKE_DOCKER_MISSING=0 FAKE_DOCKER_ARGS="$ARGS_FILE" run -0 "$SCRIPT" "$BARE_TAG"
    run -0 cat "$ARGS_FILE"
    assert_output --partial "manifest inspect ${IRD_IMAGE_BASE}:${BARE_TAG}"
}

# --- Missing arg -> exit non-zero ---

@test "no tag argument -> non-zero exit" {
    run "$SCRIPT"
    assert_failure
}
