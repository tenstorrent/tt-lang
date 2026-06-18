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
# Real tt-lang repo root (parent of .github/). Lets tests reach
# scripts/ (top-level) without hard-coding a path.
TTLANG_REPO_ROOT="$(dirname "$(dirname "$SCRIPTS_DIR")")"
WHEEL_PYTAG="cp312-cp312-linux_x86_64"
TEST_TTNN_PYPI_VERSION="99.88.77"
TEST_TT_METAL_TAG="v99.88.77"
TEST_TT_METAL_RC1_TAG="v99.88.77-rc1"
TEST_TT_METAL_RC2_TAG="v99.88.77-rc2"
TEST_TT_METAL_NEXT_TAG="v99.88.78"

whl()       { printf 'tt_lang-%s-%s.whl' "$1" "$WHEEL_PYTAG"; }
whl_sim()   { printf 'tt_lang_sim-%s-py3-none-any.whl' "$1"; }
whl_light() { printf 'tt_lang_light-%s-py3-none-any.whl' "$1"; }
whl_build() { printf 'tt_lang-%s-%s-%s.whl' "$1" "$2" "$WHEEL_PYTAG"; }

make_wheel_dir() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/wheels.XXXXXX")
    for name in "$@"; do
        : > "$dir/$name"
    done
    echo "$dir"
}

write_tt_metal_version_file() {
    local version_file="$1"
    local ttnn_pypi="$2"
    local pypi_tag="$3"
    local tt_metal_tag="$4"
    cat > "$version_file" <<EOF
TTNN_PYPI="$ttnn_pypi"
TTNN_PYPI_TT_METAL_TAG="$pypi_tag"
TT_METAL_TAG="$tt_metal_tag"
EOF
}

make_tt_metal_version_file() {
    local pypi_tag="$1"
    local tt_metal_tag="$2"
    local ttnn_pypi="${3:-$TEST_TTNN_PYPI_VERSION}"
    local version_file="$BATS_TEST_TMPDIR/tt-metal-version.$pypi_tag.$tt_metal_tag"
    write_tt_metal_version_file "$version_file" "$ttnn_pypi" "$pypi_tag" "$tt_metal_tag"
    echo "$version_file"
}

# Build a synthetic git repo in $BATS_TEST_TMPDIR (auto-cleaned). Initialized
# with one file at each UPLIFT_PATHS location, plus python/sim/example.py for
# tests that need a non-uplift file to modify. Echoes the repo path.
mkrepo() {
    local tmpdir
    # `mktemp -d <template>` is portable across Linux and BSD/macOS;
    # `-p <dir>` is Linux-only (BSD `-p` is a prefix template).
    tmpdir=$(mktemp -d "${BATS_TEST_TMPDIR:-/tmp}/repo.XXXXXX")
    (
        cd "$tmpdir"
        git init -q -b main
        git config user.email t@t
        git config user.name t
        mkdir -p third-party/llvm-project third-party/tt-metal .github/containers python/sim
        # Sourceable shell snippet matching the real third-party/tt-metal-version
        # schema.
        write_tt_metal_version_file third-party/tt-metal-version \
            "$TEST_TTNN_PYPI_VERSION" \
            "$TEST_TT_METAL_TAG" \
            "$TEST_TT_METAL_TAG"
        echo "llvm-content-v1" > third-party/llvm-project/sentinel
        echo "tt-metal-content-v1" > third-party/tt-metal/sentinel
        cat > .github/containers/Dockerfile.base <<'EOF'
FROM ubuntu:24.04
RUN echo "base v1"
EOF
        echo "greenlet>=3.0.0" > requirements-runtime.txt
        echo "// kernel placeholder" > python/sim/example.py
        git add -A
        git commit -q -m "initial"
    )
    echo "$tmpdir"
}

# Copy .github/scripts/ (except tests/) and .github/containers/ from the
# real tt-lang checkout into the synthetic repo so the scripts under test
# find their own dependencies via the usual relative paths, then commit.
# Commit is required because the real .github/containers/Dockerfile.base
# overwrites the placeholder mkrepo wrote; without committing here, that
# overwrite would appear in every later test's diff and break uplift checks.
install_scripts_in_repo() {
    local repo="$1"
    mkdir -p "$repo/.github/scripts" "$repo/.github/containers"
    find "$SCRIPTS_DIR" -maxdepth 1 -mindepth 1 -not -name tests \
        -exec cp -r {} "$repo/.github/scripts/" \;
    find "$CONTAINERS_DIR" -maxdepth 1 -mindepth 1 \
        -exec cp -r {} "$repo/.github/containers/" \;
    (cd "$repo" && git add -A && git commit -q -m "install scripts under test")
}

commit_all() {
    local repo="$1"
    local msg="$2"
    (cd "$repo" && git add -A && git commit -q -m "$msg")
}
