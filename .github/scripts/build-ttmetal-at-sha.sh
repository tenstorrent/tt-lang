#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build tt-metal at an arbitrary commit SHA into an isolated install tree that
# can be consumed as TTLANG_EXTERNAL_TT_METAL_DIR (install layout (a) in
# cmake/modules/BuildTTMetal.cmake). Prints, to $GITHUB_OUTPUT (or stdout):
#   install_dir=<path>   (default mode)  -- pass to TTLANG_EXTERNAL_TT_METAL_DIR
#   source_dir=<path>    (--no-build)    -- checked-out tt-metal source tree
#   ttmetal_date=<iso>                   -- the SHA's committer date (%cI)
#   sha=<sha>                            -- the full checked-out SHA built
#   short=<sha7>                         -- the SHA's 7-char prefix
#
# The tt-metal build is standalone (its own cmake config, mirrored from
# BuildTTMetal.cmake) rather than tt-lang's build-and-install.sh so that a
# per-SHA install lands in scratch without touching the shared toolchain's
# tt-metal, and so LLVM is never involved (tt-metal does not depend on it).
#
# Idempotent on the scratch dir: an existing clone is reused and re-checked-out;
# an existing build is reused only when its recorded source SHA matches the
# checked-out source tree.
#
# Usage: build-ttmetal-at-sha.sh --sha <sha> [--scratch-dir <dir>] [--no-build]
#
# Env:
#   TTMETAL_REMOTE_URL  tt-metal git remote (default github.com/tenstorrent/tt-metal)
#   TTLANG_PYTHON_VENV  venv to activate for the build (default /opt/ttlang-toolchain/venv)
#   CPM_SOURCE_CACHE    tt-metal CPM dependency cache (default <scratch>/.cpmcache)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

REMOTE_URL="${TTMETAL_REMOTE_URL:-https://github.com/tenstorrent/tt-metal.git}"
PYTHON_VENV="${TTLANG_PYTHON_VENV:-/opt/ttlang-toolchain/venv}"

usage() {
    echo "Usage: $0 --sha <sha> [--scratch-dir <dir>] [--no-build]" >&2
    exit 2
}

sha=""
scratch_dir="${TTLANG_TTMETAL_SCRATCH:-/tmp/ttmetal-at-sha}"
no_build=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sha)
            [[ $# -ge 2 ]] || usage
            sha="$2"
            shift 2
            ;;
        --scratch-dir)
            [[ $# -ge 2 ]] || usage
            scratch_dir="$2"
            shift 2
            ;;
        --no-build)
            no_build=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

sha="$(printf '%s' "$sha" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

if [[ -z "$sha" ]]; then
    echo "--sha is required" >&2
    usage
fi

src_dir="$scratch_dir/src"
build_dir="$scratch_dir/build"
install_dir="$scratch_dir/install"
cpm_cache="${CPM_SOURCE_CACHE:-$scratch_dir/.cpmcache}"
build_stamp="$build_dir/.ttmetal-source-sha"

emit() {
    printf '%s\n' "$1" >> "${GITHUB_OUTPUT:-/dev/stdout}"
}

# Clone (once) and check out $sha. Robust to shallow clones and bare SHAs.
ensure_source() {
    mkdir -p "$scratch_dir"
    if [[ ! -d "$src_dir/.git" ]]; then
        git clone --no-checkout "$REMOTE_URL" "$src_dir"
    fi
    if ! git -C "$src_dir" cat-file -e "${sha}^{commit}" 2>/dev/null; then
        git -C "$src_dir" fetch --tags origin "$sha" \
            || git -C "$src_dir" fetch --tags origin
    fi
    git -C "$src_dir" checkout --quiet --detach "$sha"
    git -C "$src_dir" submodule update --init --recursive --depth 1
}

commit_date() {
    git -C "$src_dir" show -s --format=%cI "$1"
}

# Log whether $sha carries a release tag. The published ttnn wheel for a tagged
# release could skip a source build, but its payload does not include the JIT
# source tree (tt_metal/) that the external-tt-metal install layout requires, so
# the source build below is used regardless.
log_release_tag() {
    local tags
    tags="$(git -C "$src_dir" tag --points-at "$sha" 2>/dev/null | grep -E '^v[0-9]' || true)"
    if [[ -n "$tags" ]]; then
        echo "Release tag(s) at $sha: $(echo "$tags" | tr '\n' ' ')" >&2
    fi
}

configure_ttmetal() {
    # Mirrors the curated tt-metal configure in BuildTTMetal.cmake: minimal
    # runtime build, Python bindings against the venv interpreter, no tests.
    cmake -G Ninja -S "$src_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$build_dir" \
        -DCMAKE_INSTALL_MESSAGE=NEVER \
        -DCPM_SOURCE_CACHE="$cpm_cache" \
        -DPython3_FIND_VIRTUALENV=ONLY \
        -DWITH_PYTHON_BINDINGS=ON \
        -DTT_UNITY_BUILDS=ON \
        -DENABLE_CCACHE=OFF \
        -DENABLE_TRACY=ON \
        -DENABLE_DISTRIBUTED=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_PROGRAMMING_EXAMPLES=OFF \
        -DTT_METAL_BUILD_TESTS=OFF \
        -DTTNN_BUILD_TESTS=OFF \
        -DBUILD_TT_TRAIN=OFF \
        -DBUILD_TELEMETRY=OFF \
        -DENABLE_TTNN_SHARED_SUBLIBS=OFF \
        -DTT_ENABLE_LIGHT_METAL_TRACE=OFF \
        -DENABLE_LIBCXX=OFF
}

build_ttmetal() {
    # Build only the ttnn runtime targets (BuildTTMetal.cmake avoids `all` to
    # skip gtest), then precompile firmware. Firmware precompile is serial.
    local build_env=(
        "TT_METAL_RUNTIME_ROOT=$src_dir"
        "TT_METAL_HOME=$src_dir"
        "TT_METAL_CACHE=$build_dir/tt-metal-cache"
    )
    env "${build_env[@]}" cmake --build "$build_dir" --target ttnn ttnncpp
    env "${build_env[@]}" cmake --build "$build_dir" --target precompile-fw --parallel 1
}

source_sha() {
    git -C "$src_dir" rev-parse HEAD
}

build_matches_source() {
    local expected_sha="$1"
    [[ -f "$build_dir/ttnn/_ttnn.so" ]] || return 1
    [[ -f "$build_stamp" ]] || return 1
    [[ "$(cat "$build_stamp")" == "$expected_sha" ]]
}

build_and_install() {
    if [[ -f "$PYTHON_VENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        . "$PYTHON_VENV/bin/activate"
    fi
    mkdir -p "$cpm_cache"
    local checked_out_sha
    checked_out_sha="$(source_sha)"
    if build_matches_source "$checked_out_sha"; then
        echo "Reusing existing tt-metal build at $build_dir for $checked_out_sha" >&2
    else
        if [[ -d "$build_dir" ]]; then
            echo "Discarding tt-metal build at $build_dir; source SHA is $checked_out_sha" >&2
            rm -rf "$build_dir"
        fi
        configure_ttmetal
        build_ttmetal
        printf '%s\n' "$checked_out_sha" > "$build_stamp"
    fi
    rm -rf "$install_dir"
    bash "$repo_root/scripts/install-ttmetal.sh" "$src_dir" "$build_dir" "$install_dir"
}

ensure_source
log_release_tag
checked_out_sha="$(source_sha)"
ttmetal_date="$(commit_date "$checked_out_sha")"
emit "sha=$checked_out_sha"
emit "short=${checked_out_sha:0:7}"

if [[ "$no_build" -eq 1 ]]; then
    emit "source_dir=$src_dir"
    emit "ttmetal_date=$ttmetal_date"
    exit 0
fi

build_and_install
emit "install_dir=$install_dir"
emit "ttmetal_date=$ttmetal_date"
