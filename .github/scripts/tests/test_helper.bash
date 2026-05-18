# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared bats helpers for .github/scripts/tests/. Loaded with `load test_helper`
# from each *.bats file.
#
# Requires bats-support and bats-assert. The CI workflow installs both via
# bats-core/bats-action. Locally, install via your package manager and set
# BATS_LIB_PATH to the directory containing the bats-support and bats-assert
# install dirs.

bats_require_minimum_version 1.5.0
bats_load_library bats-support
bats_load_library bats-assert

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$TESTS_DIR")"
CONTAINERS_DIR="$(dirname "$SCRIPTS_DIR")/containers"
BIN_DIR="$(dirname "$SCRIPTS_DIR")/../bin"

# Build a synthetic git repo inside the bats per-test tmpdir (auto-cleaned).
# Commits a minimal initial set of files (the five uplift paths + a non-uplift
# sentinel under python/sim/ used by tests that need a non-uplift change).
# Echoes the repo path. Multiple calls within one @test get distinct subdirs.
mkrepo() {
    local tmpdir
    tmpdir=$(mktemp -d -p "${BATS_TEST_TMPDIR:-/tmp}")
    (
        cd "$tmpdir"
        git init -q -b main
        git config user.email t@t
        git config user.name t
        mkdir -p third-party/llvm-project third-party/tt-metal .github/containers python/sim
        echo v0.69.0 > third-party/tt-metal-version
        echo "llvm-content-v1" > third-party/llvm-project/sentinel
        echo "tt-metal-content-v1" > third-party/tt-metal/sentinel
        cat > .github/containers/Dockerfile.base <<'EOF'
FROM ubuntu:22.04
RUN echo "base v1"
EOF
        echo "greenlet>=3.0.0" > requirements-runtime.txt
        echo "// kernel placeholder" > python/sim/example.py
        git add -A
        git commit -q -m "initial"
    )
    echo "$tmpdir"
}

# Copy the scripts under test (and their dependencies) into the repo so they
# resolve their own paths correctly. The synthetic repo must look like a
# tt-lang checkout to the scripts.
install_scripts_in_repo() {
    local repo="$1"
    mkdir -p "$repo/.github/scripts" "$repo/.github/containers"
    cp "$SCRIPTS_DIR/uplift-paths.sh"          "$repo/.github/scripts/"
    cp "$SCRIPTS_DIR/detect-uplift.sh"         "$repo/.github/scripts/"
    cp "$SCRIPTS_DIR/require-release-tag.sh"   "$repo/.github/scripts/"
    cp "$SCRIPTS_DIR/verify-wheel-version.sh"  "$repo/.github/scripts/"
    cp "$CONTAINERS_DIR/get-version-tag.sh"    "$repo/.github/containers/"
}

# Stage all changes and commit with a message.
commit_all() {
    local repo="$1"
    local msg="$2"
    (cd "$repo" && git add -A && git commit -q -m "$msg")
}
