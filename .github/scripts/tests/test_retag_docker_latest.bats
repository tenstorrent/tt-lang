#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/retag-docker-latest.sh.

load test_helper

IRD_IMAGE_BASE="ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-24-04"
DIST_IMAGE_BASE="ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-24-04"
TAG="v99.99.99"

# `manifest inspect` fails when FAKE_DOCKER_MISSING=1; argv goes to
# $FAKE_DOCKER_ARGS.
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

setup() {
    SCRIPT="$BATS_TEST_DIRNAME/../retag-docker-latest.sh"
    FAKE_DOCKER_ARGS="$BATS_TEST_TMPDIR/docker-args"
    : > "$FAKE_DOCKER_ARGS"
    export FAKE_DOCKER_ARGS
    PATH="$(make_docker_mock):$PATH"
    export PATH
    export GITHUB_REPOSITORY="tenstorrent/tt-lang"
}

@test "points both ird and dist :latest at the given tag" {
    run -0 "$SCRIPT" "$TAG"
    assert_output --partial "Pointed $IRD_IMAGE_BASE:latest at $TAG"
    assert_output --partial "Pointed $DIST_IMAGE_BASE:latest at $TAG"
}

@test "copies the manifest instead of rebuilding" {
    run -0 "$SCRIPT" "$TAG"
    grep -q "buildx imagetools create --tag $IRD_IMAGE_BASE:latest $IRD_IMAGE_BASE:$TAG" \
        "$FAKE_DOCKER_ARGS"
    grep -q "buildx imagetools create --tag $DIST_IMAGE_BASE:latest $DIST_IMAGE_BASE:$TAG" \
        "$FAKE_DOCKER_ARGS"
}

@test "verifies the source image exists before moving the tag" {
    run -0 "$SCRIPT" "$TAG"
    grep -q "manifest inspect $DIST_IMAGE_BASE:$TAG" "$FAKE_DOCKER_ARGS"
}

@test "missing source image -> refuse without moving :latest" {
    FAKE_DOCKER_MISSING=1 run -1 "$SCRIPT" "$TAG"
    assert_output --partial "refusing to move :latest"
    run -1 grep -q "imagetools create" "$FAKE_DOCKER_ARGS"
}

@test "no tag argument -> non-zero exit" {
    run -1 "$SCRIPT"
}
