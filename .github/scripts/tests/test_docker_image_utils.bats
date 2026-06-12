#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/lib/docker-image-utils.sh.

load test_helper

setup() {
    LIB="$SCRIPTS_DIR/lib/docker-image-utils.sh"
    MOCK_DOCKER="$BATS_TEST_TMPDIR/docker"
}

write_mock_docker() {
    local exit_code="$1"
    cat > "$MOCK_DOCKER" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$BATS_TEST_TMPDIR/docker.calls"
exit $exit_code
EOF
    chmod +x "$MOCK_DOCKER"
    export DOCKER="$MOCK_DOCKER"
}

@test "ttlang_image_for_tag prefers a local image when it exists" {
    write_mock_docker 0

    run bash -c "source '$LIB'; ttlang_image_for_tag tt-lang-wheel-manylinux-2-34-cp312 local-tag"

    assert_success
    assert_output "tt-lang-wheel-manylinux-2-34-cp312:local-tag"
    run cat "$BATS_TEST_TMPDIR/docker.calls"
    assert_output "image inspect tt-lang-wheel-manylinux-2-34-cp312:local-tag"
}

@test "ttlang_image_for_tag falls back to GHCR when local image is absent" {
    write_mock_docker 1

    run bash -c "source '$LIB'; ttlang_image_for_tag tt-lang-wheel-manylinux-2-34-cp312 local-tag"

    assert_success
    assert_output "ghcr.io/tenstorrent/tt-lang/tt-lang-wheel-manylinux-2-34-cp312:local-tag"
}

@test "ttlang_image_for_tag accepts an explicit registry" {
    write_mock_docker 1

    run bash -c "source '$LIB'; ttlang_image_for_tag image-name image-tag registry.example.com/repo"

    assert_success
    assert_output "registry.example.com/repo/image-name:image-tag"
}

@test "ttlang_image_for_tag validates arguments" {
    run bash -c "source '$LIB'; ttlang_image_for_tag image-only"

    assert_failure
    assert_output --partial "Usage: ttlang_image_for_tag"
}
