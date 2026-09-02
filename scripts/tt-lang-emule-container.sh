#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run a TT-Lang program through the compiler and an emule-enabled tt-metal
# runtime. macOS uses Docker's linux/amd64 virtualization because tt-emule and
# tt-metal are Linux/x86-64 components.

set -euo pipefail

readonly _DEFAULT_TT_EMULE_COMMIT="07f1bd8301544403c8bc1faa4038f6cbf69909f1"
readonly _DEFAULT_TT_METAL_COMMIT="d48d09dee19de51f694a52fdf75d569950d38ceb"
readonly _TT_EMULE_COMMIT="${TTLANG_EMULE_RUNTIME_COMMIT:-$_DEFAULT_TT_EMULE_COMMIT}"
readonly _TT_METAL_COMMIT="${TTLANG_EMULE_RUNTIME_METAL_COMMIT:-$_DEFAULT_TT_METAL_COMMIT}"
readonly _TT_EMULE_SOURCE_URL="${TTLANG_EMULE_RUNTIME_SOURCE_URL:-https://github.com/tenstorrent/tt-emule.git}"

for _COMMIT in "$_TT_EMULE_COMMIT" "$_TT_METAL_COMMIT"; do
    if [ "${#_COMMIT}" -ne 40 ] || [[ "$_COMMIT" == *[!0-9a-f]* ]]; then
        echo "tt-lang-sim: emulator revisions must be full lowercase commit SHAs." >&2
        exit 2
    fi
done

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_REPO_ROOT="$(dirname "$_SCRIPT_DIR")"
_IMAGE_INPUT_ID="$(
    cksum "${_REPO_ROOT}/.github/containers/Dockerfile.emule" \
        "${_SCRIPT_DIR}/tt-lang-emule-entrypoint.sh" |
        awk '{print $1, $2}' |
        cksum |
        awk '{print $1}'
)"
readonly _IMAGE_INPUT_ID
readonly _RUNTIME_ID="${_TT_EMULE_COMMIT:0:8}-${_TT_METAL_COMMIT:0:8}-r${_IMAGE_INPUT_ID}"
_DOCKER="${TTLANG_EMULE_DOCKER:-docker}"
_PLATFORM="${TTLANG_EMULE_PLATFORM:-linux/amd64}"
_IMAGE="${TTLANG_EMULE_IMAGE:-tt-lang-emule:${_RUNTIME_ID}}"
_SOURCE_ID="$(printf '%s' "$_REPO_ROOT" | cksum | awk '{print $1}')"
_BUILD_VOLUME="${TTLANG_EMULE_BUILD_VOLUME:-tt-lang-emule-build-${_RUNTIME_ID}-${_SOURCE_ID}}"
_CACHE_VOLUME="${TTLANG_EMULE_CACHE_VOLUME:-tt-lang-emule-cache-${_RUNTIME_ID}}"
_TEMP_EMULE_SOURCE=""

cleanup() {
    if [ -n "$_TEMP_EMULE_SOURCE" ] && [ -d "$_TEMP_EMULE_SOURCE" ]; then
        rm -rf -- "$_TEMP_EMULE_SOURCE"
    fi
}

trap cleanup EXIT HUP INT TERM

usage() {
    cat >&2 <<'EOF'
Usage: tt-lang-sim SCRIPT.py [arguments] --backend emule

Runs SCRIPT.py unchanged with the TT-Lang compiler and tt-emule. A working
Docker-compatible daemon is required. On Apple Silicon the image runs as
linux/amd64 through the container runtime's x86 virtualization.
EOF
}

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi

_SCRIPT_ARGUMENT="$1"
shift
if [ ! -f "$_SCRIPT_ARGUMENT" ]; then
    echo "tt-lang-sim: script not found: ${_SCRIPT_ARGUMENT}" >&2
    exit 2
fi

if ! command -v "$_DOCKER" >/dev/null 2>&1; then
    echo "tt-lang-sim: Docker CLI not found: ${_DOCKER}" >&2
    echo "Install Docker Desktop or Colima, then retry." >&2
    exit 1
fi

if ! "$_DOCKER" info >/dev/null 2>&1; then
    echo "tt-lang-sim: cannot connect to the Docker daemon." >&2
    echo "Start Docker Desktop or a Colima VM with amd64 emulation, then retry." >&2
    exit 1
fi

_HOST_CWD="$(pwd -P)"
_SCRIPT_ABSOLUTE="$(realpath "$_SCRIPT_ARGUMENT")"
_SCRIPT_DIR_HOST="$(dirname "$_SCRIPT_ABSOLUTE")"
_SCRIPT_BASENAME="$(basename "$_SCRIPT_ABSOLUTE")"

_RUN_ARGS=(
    run
    --rm
    --platform "$_PLATFORM"
    --mount "type=bind,src=${_REPO_ROOT},dst=/workspace"
    --mount "type=volume,src=${_BUILD_VOLUME},dst=/ttlang-build"
    --mount "type=volume,src=${_CACHE_VOLUME},dst=/tt-metal-cache"
)

case "${_HOST_CWD}/" in
    "${_REPO_ROOT}/"*)
        _CONTAINER_CWD="/workspace${_HOST_CWD#"$_REPO_ROOT"}"
        ;;
    *)
        _CONTAINER_CWD="/workdir"
        _RUN_ARGS+=(--mount "type=bind,src=${_HOST_CWD},dst=/workdir")
        ;;
esac

case "${_SCRIPT_ABSOLUTE}" in
    "${_REPO_ROOT}/"*)
        _CONTAINER_SCRIPT="/workspace${_SCRIPT_ABSOLUTE#"$_REPO_ROOT"}"
        ;;
    "${_HOST_CWD}/"*)
        _CONTAINER_SCRIPT="${_CONTAINER_CWD}${_SCRIPT_ABSOLUTE#"$_HOST_CWD"}"
        ;;
    *)
        _CONTAINER_SCRIPT="/ttlang-script/${_SCRIPT_BASENAME}"
        _RUN_ARGS+=(--mount "type=bind,src=${_SCRIPT_DIR_HOST},dst=/ttlang-script")
        ;;
esac

_RUN_ARGS+=(--workdir "$_CONTAINER_CWD")

# Docker detaches stdin without -i, including when the caller supplies a pipe.
_RUN_ARGS+=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    _RUN_ARGS+=(-t)
fi

for _ENV_NAME in \
    TTLANG_EMULE_JOBS \
    TTLANG_KEEP_GENERATED_KERNELS \
    TT_METAL_DPRINT_CHIPS \
    TT_METAL_DPRINT_CORES \
    TT_METAL_LOGGER_LEVEL; do
    if [ -n "${!_ENV_NAME:-}" ]; then
        _RUN_ARGS+=(-e "${_ENV_NAME}")
    fi
done

if [ "${TTLANG_EMULE_REBUILD:-0}" = "1" ] || \
   ! "$_DOCKER" image inspect "$_IMAGE" >/dev/null 2>&1; then
    _EMULE_SOURCE="${TTLANG_EMULE_RUNTIME_SOURCE_DIR:-}"
    if [ -z "$_EMULE_SOURCE" ]; then
        if ! command -v git >/dev/null 2>&1; then
            echo "tt-lang-sim: git is required to fetch the emulator source." >&2
            exit 1
        fi
        _TEMP_EMULE_SOURCE="$(mktemp -d "${TMPDIR:-/tmp}/tt-lang-emule.XXXXXX")"
        git init "$_TEMP_EMULE_SOURCE"
        git -C "$_TEMP_EMULE_SOURCE" remote add origin "$_TT_EMULE_SOURCE_URL"
        git -C "$_TEMP_EMULE_SOURCE" fetch --depth 1 origin "$_TT_EMULE_COMMIT"
        git -C "$_TEMP_EMULE_SOURCE" checkout --detach FETCH_HEAD
        _EMULE_SOURCE="$_TEMP_EMULE_SOURCE"
    fi
    if [ ! -d "$_EMULE_SOURCE" ]; then
        echo "tt-lang-sim: emulator source directory not found: ${_EMULE_SOURCE}" >&2
        exit 1
    fi
    _EMULE_SOURCE="$(cd "$_EMULE_SOURCE" && pwd -P)"
    _EMULE_SOURCE_COMMIT="$(git -C "$_EMULE_SOURCE" rev-parse HEAD 2>/dev/null || true)"
    if [ "$_EMULE_SOURCE_COMMIT" != "$_TT_EMULE_COMMIT" ]; then
        echo "tt-lang-sim: emulator source must be at ${_TT_EMULE_COMMIT}." >&2
        echo "  source: ${_EMULE_SOURCE}" >&2
        echo "  found:  ${_EMULE_SOURCE_COMMIT:-not a Git checkout}" >&2
        exit 1
    fi
    echo "tt-lang-sim: building compiler + tt-emule image ${_IMAGE}" >&2
    "$_DOCKER" build \
        --platform "$_PLATFORM" \
        --build-context "tt-emule-source=${_EMULE_SOURCE}" \
        --file "${_REPO_ROOT}/.github/containers/Dockerfile.emule" \
        --build-arg "TT_EMULE_COMMIT=${_TT_EMULE_COMMIT}" \
        --build-arg "TT_METAL_COMMIT=${_TT_METAL_COMMIT}" \
        --tag "$_IMAGE" \
        "${_REPO_ROOT}/scripts"
fi

exec "$_DOCKER" "${_RUN_ARGS[@]}" "$_IMAGE" "$_CONTAINER_SCRIPT" "$@"
