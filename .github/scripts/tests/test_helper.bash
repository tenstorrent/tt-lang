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
LIGHT_CP310_PYTAG="cp310-cp310-manylinux_2_34_x86_64"
LIGHT_CP312_PYTAG="cp312-cp312-manylinux_2_34_x86_64"
TEST_TTNN_PYPI_VERSION="99.88.77"
TEST_TT_METAL_TAG="v99.88.77"
TEST_TT_METAL_RC1_TAG="v99.88.77-rc1"
TEST_TT_METAL_RC2_TAG="v99.88.77-rc2"
TEST_TT_METAL_NEXT_TAG="v99.88.78"

whl()       { printf 'tt_lang-%s-%s.whl' "$1" "$WHEEL_PYTAG"; }
whl_sim()   { printf 'tt_lang_sim-%s-py3-none-any.whl' "$1"; }
whl_light() { printf 'tt_lang_light-%s-py3-none-any.whl' "$1"; }
whl_build() { printf 'tt_lang-%s-%s-%s.whl' "$1" "$2" "$WHEEL_PYTAG"; }
whl_light_core_cp310() { printf 'tt_lang-%s+light-%s.whl' "$1" "$LIGHT_CP310_PYTAG"; }
whl_light_core_cp312() { printf 'tt_lang-%s+light-%s.whl' "$1" "$LIGHT_CP312_PYTAG"; }
whl_light_core_tagged() { printf 'tt_lang-%s+light-%s.whl' "$1" "$2"; }

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
        mkdir -p \
            third-party/patches \
            third-party/llvm-project \
            third-party/tt-metal \
            .github/containers \
            .github/scripts \
            bin \
            cmake/modules \
            docs \
            python/sim \
            scripts
        # Sourceable shell snippet matching the real third-party/tt-metal-version
        # schema.
        write_tt_metal_version_file third-party/tt-metal-version \
            "$TEST_TTNN_PYPI_VERSION" \
            "$TEST_TT_METAL_TAG" \
            "$TEST_TT_METAL_TAG"
        echo "llvm-content-v1" > third-party/llvm-project/sentinel
        echo "tt-metal-content-v1" > third-party/tt-metal/sentinel
        echo "patch-content-v1" > third-party/patches/sentinel
        echo "cmake_minimum_required(VERSION 3.28)" > CMakeLists.txt
        echo "build/" > .dockerignore
        cat > .github/containers/Dockerfile.base <<'EOF'
FROM ubuntu:24.04
RUN echo "base v1"
EOF
        cat > .github/containers/CMakeLists.wheel-toolchain <<'EOF'
cmake_minimum_required(VERSION 3.28)
project(test-wheel-toolchain)
EOF
        echo "build manylinux images" > .github/containers/build-wheel-manylinux-images.sh
        echo "cache manylinux component" > .github/containers/cache-wheel-manylinux-component.sh
        echo "cleanup toolchain" > .github/containers/cleanup-toolchain.sh
        echo "normalize toolchain" > .github/scripts/normalize-toolchain-install.sh
        echo "tt-triage launcher" > bin/tt-triage
        echo "build llvm" > cmake/modules/BuildLLVM.cmake
        echo "build tt-metal" > cmake/modules/BuildTTMetal.cmake
        echo "version from git" > cmake/modules/GetVersionFromGit.cmake
        echo "compiler setup" > cmake/modules/TTLangCompilerSetup.cmake
        echo "python setup" > cmake/modules/TTLangPython.cmake
        echo "toolchain component" > cmake/modules/TTLangToolchainComponent.cmake
        echo "toolchain options" > cmake/modules/TTLangToolchainOptions.cmake
        echo "cmake helpers" > cmake/modules/TTLangUtils.cmake
        echo "pytest" > dev-requirements.txt
        echo "sphinx" > docs/requirements.txt
        echo "-r requirements-runtime.txt" > requirements.txt
        echo "greenlet>=3.0.0" > requirements-runtime.txt
        echo "copy runtime artifacts" > scripts/copy-ttmetal-runtime-artifacts.sh
        echo "install tt-metal" > scripts/install-ttmetal.sh
        echo "verify sha" > scripts/verify-sha.sh
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

list_uplift_paths() {
    bash -c 'source "$1"; printf "%s\n" "${UPLIFT_PATHS[@]}"' \
        _ "$1"
}

modify_repo_path() {
    local repo="$1"
    local path_to_change="$2"
    local target="$repo/$path_to_change"

    if [ -d "$target" ]; then
        echo "modified" >> "$target/sentinel"
    else
        mkdir -p "$(dirname "$target")"
        echo "modified" >> "$target"
    fi
}
