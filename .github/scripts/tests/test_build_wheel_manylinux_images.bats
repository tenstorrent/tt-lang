#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the skip-if-exists / :latest-retag logic in
# .github/containers/build-wheel-manylinux-images.sh.

load test_helper

# Fake `docker`, injected via the DOCKER env var the script honors. Records
# every invocation's argv to $FAKE_DOCKER_CALLS. `manifest inspect` exit is
# controlled by FAKE_IMAGE_EXISTS (1 -> image present, else absent); `buildx
# imagetools` exit by FAKE_BUILDX_FAIL. Everything else succeeds.
make_docker_mock() {
    local mock="$BATS_TEST_TMPDIR/fake-docker"
    cat > "$mock" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_DOCKER_CALLS"
case "$1 $2" in
    "manifest inspect")
        [ "${FAKE_IMAGE_EXISTS:-0}" = "1" ] && exit 0
        exit 1
        ;;
    "buildx imagetools")
        [ "${FAKE_BUILDX_FAIL:-0}" = "1" ] && exit 1
        exit 0
        ;;
esac
exit 0
EOF
    chmod +x "$mock"
    echo "$mock"
}

TAG="v99.99.99-abcd1234"
IMG="ghcr.io/tenstorrent/tt-lang/tt-lang-wheel-manylinux-2-34-cp312:${TAG}"
LATEST="ghcr.io/tenstorrent/tt-lang/tt-lang-wheel-manylinux-2-34-cp312:latest"

setup() {
    SCRIPT="$CONTAINERS_DIR/build-wheel-manylinux-images.sh"
    REPO="$(mkrepo)"
    DOCKER_MOCK="$(make_docker_mock)"
    export DOCKER="$DOCKER_MOCK"
    export FAKE_DOCKER_CALLS="$BATS_TEST_TMPDIR/docker.calls"
    : > "$FAKE_DOCKER_CALLS"
    cd "$REPO"
}

@test "push + image exists on main: skips build and retags :latest" {
    run env FAKE_IMAGE_EXISTS=1 GITHUB_REF=refs/heads/main \
        "$SCRIPT" --image-tag "$TAG" --python-tags cp312
    assert_success
    assert_output --partial "Image already exists, skipping build"
    run cat "$FAKE_DOCKER_CALLS"
    assert_line --partial "manifest inspect $IMG"
    assert_line --partial "buildx imagetools create -t $LATEST"
    refute_output --partial "build --progress"
    refute_output --partial "push"
}

@test "push + image exists off main: skips build, no :latest retag" {
    run env FAKE_IMAGE_EXISTS=1 GITHUB_REF=refs/heads/feature \
        "$SCRIPT" --image-tag "$TAG" --python-tags cp312
    assert_success
    run cat "$FAKE_DOCKER_CALLS"
    assert_line --partial "manifest inspect $IMG"
    refute_output --partial "buildx"
    refute_output --partial "build --progress"
}

@test "push + image missing: builds and pushes" {
    run env FAKE_IMAGE_EXISTS=0 GITHUB_REF=refs/heads/main \
        "$SCRIPT" --image-tag "$TAG" --python-tags cp312
    assert_success
    refute_output --partial "skipping build"
    run cat "$FAKE_DOCKER_CALLS"
    assert_line --partial "manifest inspect $IMG"
    assert_line --partial "build --progress"
    assert_line --partial "push $IMG"
}

@test "no-push: builds locally, never probes the registry" {
    run env FAKE_IMAGE_EXISTS=1 GITHUB_REF=refs/heads/main \
        "$SCRIPT" --no-push --image-tag "$TAG" --python-tags cp312
    assert_success
    run cat "$FAKE_DOCKER_CALLS"
    refute_output --partial "manifest inspect"
    refute_output --partial "buildx"
    assert_line --partial "build --progress"
    refute_output --partial "push"
}

@test "push + image exists on main but buildx unavailable: warns, still succeeds" {
    run env FAKE_IMAGE_EXISTS=1 FAKE_BUILDX_FAIL=1 GITHUB_REF=refs/heads/main \
        "$SCRIPT" --image-tag "$TAG" --python-tags cp312
    assert_success
    assert_output --partial "WARNING: could not retag"
    run cat "$FAKE_DOCKER_CALLS"
    refute_output --partial "build --progress"
}
