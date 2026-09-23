#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run a TT-Lang program through the compiler and an emule-enabled tt-metal
# runtime. macOS uses Docker's linux/amd64 virtualization because tt-emule and
# tt-metal are Linux/x86-64 components.

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_REPO_ROOT="$(dirname "$_SCRIPT_DIR")"
readonly _SCRIPT_DIR _REPO_ROOT
readonly _STACK_MANIFEST="${_REPO_ROOT}/config/tt-lang-emule-stack.json"
readonly _STACK_TOOL="${_SCRIPT_DIR}/tt-lang-emule-stack.py"
readonly _PYTHON="${TTLANG_EMULE_HOST_PYTHON:-python3}"

if ! command -v "$_PYTHON" >/dev/null 2>&1; then
    echo "tt-lang-sim: Python 3 is required to read the emulator stack manifest." >&2
    exit 1
fi

_STACK_OUTPUT="$("$_PYTHON" "$_STACK_TOOL" --manifest "$_STACK_MANIFEST" emit)"
while IFS=$'\t' read -r _STACK_KEY _STACK_VALUE; do
    case "$_STACK_KEY" in
        TTLANG_EMULE_STACK_MANIFEST_SHA256) _MANIFEST_SHA256="$_STACK_VALUE" ;;
        TTLANG_COMPILER_REPOSITORY) _MANIFEST_COMPILER_REPOSITORY="$_STACK_VALUE" ;;
        TTLANG_COMPILER_BASE_COMMIT) _MANIFEST_COMPILER_BASE_COMMIT="$_STACK_VALUE" ;;
        TTLANG_EMULE_COMMIT) _MANIFEST_EMULE_COMMIT="$_STACK_VALUE" ;;
        TTLANG_METAL_REPOSITORY) _MANIFEST_METAL_REPOSITORY="$_STACK_VALUE" ;;
        TTLANG_METAL_COMMIT) _MANIFEST_METAL_COMMIT="$_STACK_VALUE" ;;
        TTLANG_EMULE_BASE_IMAGE) _MANIFEST_BASE_IMAGE="$_STACK_VALUE" ;;
        TTLANG_EMULE_TARGET) _MANIFEST_TARGET="$_STACK_VALUE" ;;
        TTLANG_EMULE_CLUSTER_DESCRIPTOR) _MANIFEST_CLUSTER_DESCRIPTOR="$_STACK_VALUE" ;;
        TTLANG_EMULE_MESH_DEVICE) _MANIFEST_MESH_DEVICE="$_STACK_VALUE" ;;
    esac
done <<< "$_STACK_OUTPUT"

readonly _TT_EMULE_COMMIT="$_MANIFEST_EMULE_COMMIT"
readonly _TT_METAL_COMMIT="$_MANIFEST_METAL_COMMIT"
readonly _TT_EMULE_SOURCE_URL="${TTLANG_EMULE_RUNTIME_SOURCE_URL:-}"
readonly _TT_METAL_SOURCE_URL="$_MANIFEST_METAL_REPOSITORY"
readonly _BASE_IMAGE="$_MANIFEST_BASE_IMAGE"
readonly _REQUIRED_EMULE_FILE="$_MANIFEST_CLUSTER_DESCRIPTOR"
readonly _PLATFORM="linux/amd64"

_IMAGE_INPUT_ID="$(
    {
        cksum "$_STACK_MANIFEST" \
            "${_REPO_ROOT}/.github/containers/Dockerfile.emule" |
            awk '{print $1, $2}'
        printf '%s\n' "$_BASE_IMAGE" "$_PLATFORM"
    } |
        cksum |
        awk '{print $1}'
)"
readonly _IMAGE_INPUT_ID
readonly _RUNTIME_ID="${_TT_EMULE_COMMIT:0:8}-${_TT_METAL_COMMIT:0:8}-r${_IMAGE_INPUT_ID}"
_DOCKER="${TTLANG_EMULE_DOCKER:-docker}"
_IMAGE="${TTLANG_EMULE_IMAGE:-tt-lang-emule:${_RUNTIME_ID}}"
_SOURCE_ID="$(printf '%s' "$_REPO_ROOT" | cksum | awk '{print $1}')"
_BUILD_VOLUME="${TTLANG_EMULE_BUILD_VOLUME:-tt-lang-emule-build-${_RUNTIME_ID}-${_SOURCE_ID}}"
_CACHE_VOLUME="${TTLANG_EMULE_CACHE_VOLUME:-tt-lang-emule-cache-${_RUNTIME_ID}}"
_TEMP_EMULE_SOURCE=""
_TEMP_EMULE_CONTEXT=""
_TEMP_STACK_CONTEXT=""

"$_PYTHON" "$_STACK_TOOL" --manifest "$_STACK_MANIFEST" validate \
    --compiler-source "$_REPO_ROOT" --quiet

_COMPILER_SHA="$(git -C "$_REPO_ROOT" rev-parse HEAD)"
# Workloads and their output do not change the installed compiler. Include
# build inputs so edits to compiler sources still require reinstallation.
_COMPILER_INPUTS=(
    CMakeLists.txt .gitmodules
    cmake env include lib python tools third-party
    setup.py pyproject.toml packaging
    requirements.txt requirements-runtime.txt dev-requirements.txt
    docs/requirements.txt scripts config
    .github/containers/Dockerfile.emule
    test/CMakeLists.txt test/lib test/pytest.ini.in
    test/lit.cfg.py test/lit.site.cfg.py.in
)
readonly _COMPILER_INPUTS
_COMPILER_SOURCE_FINGERPRINT="$(
    {
        printf '%s\n' "$_COMPILER_SHA"
        git -C "$_REPO_ROOT" diff --no-ext-diff --binary HEAD -- \
            "${_COMPILER_INPUTS[@]}"
        while IFS= read -r -d '' _UNTRACKED; do
            [ -f "${_REPO_ROOT}/${_UNTRACKED}" ] || continue
            printf 'untracked:%s\n' "$_UNTRACKED"
            cksum "${_REPO_ROOT}/${_UNTRACKED}"
        done < <(git -C "$_REPO_ROOT" ls-files --others --exclude-standard -z -- \
            "${_COMPILER_INPUTS[@]}")
    } |
        cksum |
        awk '{print $1, $2}'
)"
readonly _COMPILER_SHA _COMPILER_SOURCE_FINGERPRINT

_EXPECTED_LLVM_SHA="$(
    git -C "$_REPO_ROOT" ls-tree HEAD -- third-party/llvm-project |
        awk '$1 == "160000" && $2 == "commit" {print $3}'
)"
if [ "${#_EXPECTED_LLVM_SHA}" -ne 40 ] || \
   [[ "$_EXPECTED_LLVM_SHA" == *[!0-9a-f]* ]]; then
    echo "tt-lang-sim: cannot read the compiler's LLVM gitlink from HEAD." >&2
    exit 1
fi
readonly _EXPECTED_LLVM_SHA

cleanup() {
    for _TEMP_DIR in \
        "$_TEMP_EMULE_SOURCE" \
        "$_TEMP_EMULE_CONTEXT" \
        "$_TEMP_STACK_CONTEXT"; do
        if [ -n "$_TEMP_DIR" ] && [ -d "$_TEMP_DIR" ]; then
            rm -rf -- "$_TEMP_DIR"
        fi
    done
    _TEMP_EMULE_SOURCE=""
    _TEMP_EMULE_CONTEXT=""
    _TEMP_STACK_CONTEXT=""
}

trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
    cat >&2 <<'EOF'
Usage: ./bin/tt-lang-sim --backend=emule SCRIPT.py [arguments]

Runs SCRIPT.py unchanged with the TT-Lang compiler and tt-emule. A working
Docker-compatible daemon is required. On Apple Silicon the image runs as
linux/amd64 through the container runtime's x86 virtualization.
EOF
}

if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ] && \
   [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    echo "tt-lang-sim: installation and shell modes are mutually exclusive." >&2
    exit 2
fi
if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ] || \
   [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    if [ "$#" -ne 0 ]; then
        echo "tt-lang-sim: installation and shell modes do not accept script arguments." >&2
        exit 2
    fi
else
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

_RUN_ARGS=(
    run
    --rm
    --platform "$_PLATFORM"
    --entrypoint /workspace/scripts/tt-lang-emule-entrypoint.sh
    --mount "type=bind,src=${_REPO_ROOT},dst=/workspace"
    --mount "type=volume,src=${_BUILD_VOLUME},dst=/ttlang-build"
    --mount "type=volume,src=${_CACHE_VOLUME},dst=/tt-metal-cache"
    -e "TTLANG_EMULE_TARGET_NAME=${_MANIFEST_TARGET}"
    -e "TTLANG_EMULE_EXPECTED_LLVM_SHA=${_EXPECTED_LLVM_SHA}"
    -e "TT_METAL_MOCK_CLUSTER_DESC_PATH=/opt/tt-emule/${_REQUIRED_EMULE_FILE}"
    -e "TT_METAL_ALLOCATOR_MODE_HYBRID=1"
    -e "MESH_DEVICE=${_MANIFEST_MESH_DEVICE}"
    -e "TTLANG_EMULE_COMPILER_SHA=${_COMPILER_SHA}"
    -e "TTLANG_EMULE_SOURCE_FINGERPRINT=${_COMPILER_SOURCE_FINGERPRINT}"
)

# A linked Git worktree stores only a .git pointer inside the checkout. Mount
# its external common directory at the same absolute path so packaging and
# versioning code inside the container can still resolve the checkout's HEAD.
_GIT_COMMON_DIR="$(
    git -C "$_REPO_ROOT" rev-parse --path-format=absolute --git-common-dir
)"
readonly _GIT_COMMON_DIR
case "${_GIT_COMMON_DIR}/" in
    "${_REPO_ROOT}/"*) ;;
    *)
        _RUN_ARGS+=(
            --mount
            "type=bind,src=${_GIT_COMMON_DIR},dst=${_GIT_COMMON_DIR},readonly"
        )
        ;;
esac

if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ]; then
    _RUN_ARGS+=(-e TTLANG_EMULE_INSTALL=1)
elif [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    _RUN_ARGS+=(-e TTLANG_EMULE_SHELL=1)
fi

case "${_HOST_CWD}/" in
    "${_REPO_ROOT}/"*)
        _CONTAINER_CWD="/workspace${_HOST_CWD#"$_REPO_ROOT"}"
        ;;
    *)
        _CONTAINER_CWD="/workdir"
        _RUN_ARGS+=(--mount "type=bind,src=${_HOST_CWD},dst=/workdir")
        ;;
esac

if [ "${TTLANG_EMULE_INSTALL:-0}" != "1" ] && \
   [ "${TTLANG_EMULE_SHELL:-0}" != "1" ]; then
    _SCRIPT_ABSOLUTE="$(realpath "$_SCRIPT_ARGUMENT")"
    _SCRIPT_DIR_HOST="$(dirname "$_SCRIPT_ABSOLUTE")"
    _SCRIPT_BASENAME="$(basename "$_SCRIPT_ABSOLUTE")"
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
fi

_RUN_ARGS+=(--workdir "$_CONTAINER_CWD")

# Docker detaches stdin without -i, including when the caller supplies a pipe.
_RUN_ARGS+=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    _RUN_ARGS+=(-t)
fi

for _ENV_NAME in \
    TTLANG_KEEP_GENERATED_KERNELS \
    TT_METAL_DPRINT_CHIPS \
    TT_METAL_DPRINT_CORES \
    TT_METAL_LOGGER_LEVEL; do
    if [ -n "${!_ENV_NAME:-}" ]; then
        _RUN_ARGS+=(-e "${_ENV_NAME}")
    fi
done

_BUILD_IMAGE=0
if [ "${TTLANG_EMULE_REBUILD:-0}" = "1" ]; then
    if [ "${TTLANG_EMULE_INSTALL:-0}" != "1" ]; then
        echo "tt-lang-sim: runtime rebuilding is only available during installation." >&2
        echo "Run scripts/install-tt-lang-emule.sh instead." >&2
        exit 1
    fi
    _BUILD_IMAGE=1
elif _IMAGE_INSPECT_ERROR="$("$_DOCKER" image inspect -- "$_IMAGE" 2>&1 >/dev/null)"; then
    :
else
    _IMAGE_INSPECT_STATUS=$?
    _IMAGE_IS_MISSING=0
    # Listing matches name patterns, not IDs or digests. Restrict the missing
    # image check to literal repository tags.
    if [ "$_IMAGE_INSPECT_STATUS" -eq 1 ] && \
        [[ "$_IMAGE" != sha256:* && ! "$_IMAGE" =~ ^[0-9a-f]{1,64}$ && \
            "$_IMAGE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$ ]]; then
        _IMAGE_TAG="$_IMAGE"
        case "$_IMAGE_TAG" in
            docker.io/*|index.docker.io/*) _IMAGE_TAG="${_IMAGE_TAG#*/}" ;;
        esac
        _IMAGE_TAG="${_IMAGE_TAG#library/}"
        case "${_IMAGE_TAG##*/}" in
            *:*) ;;
            *) _IMAGE_TAG="${_IMAGE_TAG}:latest" ;;
        esac
        if _IMAGE_IDS="$("$_DOCKER" images -q -- "$_IMAGE_TAG")"; then
            if [ -z "$_IMAGE_IDS" ]; then
                _IMAGE_IS_MISSING=1
            fi
        else
            _IMAGE_LIST_STATUS=$?
            printf 'tt-lang-sim: Docker could not list image %s (exit %s).\n' \
                "$_IMAGE" "$_IMAGE_LIST_STATUS" >&2
            echo "Check the Docker daemon and selected context, then retry." >&2
            exit "$_IMAGE_LIST_STATUS"
        fi
    fi
    if [ "$_IMAGE_IS_MISSING" -eq 1 ]; then
        if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ]; then
            _BUILD_IMAGE=1
        else
            echo "tt-lang-sim: the compiler-backed emule environment is not installed." >&2
            echo "Run scripts/install-tt-lang-emule.sh, then retry." >&2
            exit 1
        fi
    else
        printf 'tt-lang-sim: Docker could not inspect image %s (exit %s).\n' \
            "$_IMAGE" "$_IMAGE_INSPECT_STATUS" >&2
        if [ -n "$_IMAGE_INSPECT_ERROR" ]; then
            printf '%s\n' "$_IMAGE_INSPECT_ERROR" >&2
        fi
        echo "Check the Docker daemon and selected context, then retry." >&2
        exit "$_IMAGE_INSPECT_STATUS"
    fi
fi

if [ "$_BUILD_IMAGE" -eq 1 ]; then
    _EMULE_SOURCE="${TTLANG_EMULE_RUNTIME_SOURCE_DIR:-}"
    if [ -z "$_EMULE_SOURCE" ]; then
        if [ -z "$_TT_EMULE_SOURCE_URL" ]; then
            echo "tt-lang-sim: no emulator source was configured." >&2
            echo "Set TTLANG_EMULE_RUNTIME_SOURCE_URL to the approved repository" >&2
            echo "before running scripts/install-tt-lang-emule.sh." >&2
            exit 1
        fi
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
    "$_PYTHON" "$_STACK_TOOL" --manifest "$_STACK_MANIFEST" validate \
        --compiler-source "$_REPO_ROOT" --emulator-source "$_EMULE_SOURCE" \
        --quiet
    _TEMP_EMULE_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/tt-lang-emule-context.XXXXXX")"
    git -C "$_EMULE_SOURCE" archive --format=tar "$_TT_EMULE_COMMIT" |
        tar -xf - -C "$_TEMP_EMULE_CONTEXT"
    if [ ! -f "${_TEMP_EMULE_CONTEXT}/${_REQUIRED_EMULE_FILE}" ]; then
        echo "tt-lang-sim: selected emulator does not provide the required P150 descriptor." >&2
        echo "  missing: ${_REQUIRED_EMULE_FILE}" >&2
        echo "  Update the supported stack manifest before installing another runtime." >&2
        exit 1
    fi
    _TEMP_STACK_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/tt-lang-stack-context.XXXXXX")"
    cp -- "$_STACK_MANIFEST" \
        "${_TEMP_STACK_CONTEXT}/tt-lang-emule-stack.json"
    echo "tt-lang-sim: building compiler + tt-emule image ${_IMAGE}" >&2
    "$_DOCKER" build \
        --platform "$_PLATFORM" \
        --build-context "tt-emule-source=${_TEMP_EMULE_CONTEXT}" \
        --build-context "tt-lang-stack=${_TEMP_STACK_CONTEXT}" \
        --file "${_REPO_ROOT}/.github/containers/Dockerfile.emule" \
        --build-arg "STACK_MANIFEST_SHA256=${_MANIFEST_SHA256}" \
        --build-arg "TT_LANG_COMPILER_REPOSITORY=${_MANIFEST_COMPILER_REPOSITORY}" \
        --build-arg "TT_LANG_COMPILER_BASE_COMMIT=${_MANIFEST_COMPILER_BASE_COMMIT}" \
        --build-arg "TT_EMULE_COMMIT=${_TT_EMULE_COMMIT}" \
        --build-arg "TT_METAL_COMMIT=${_TT_METAL_COMMIT}" \
        --build-arg "TT_METAL_SOURCE_URL=${_TT_METAL_SOURCE_URL}" \
        --build-arg "BASE_IMAGE=${_BASE_IMAGE}" \
        --build-arg "RUNTIME_PLATFORM=${_PLATFORM}" \
        --build-arg "TARGET_NAME=${_MANIFEST_TARGET}" \
        --build-arg "TARGET_CLUSTER_DESCRIPTOR=${_MANIFEST_CLUSTER_DESCRIPTOR}" \
        --build-arg "TARGET_MESH_DEVICE=${_MANIFEST_MESH_DEVICE}" \
        --tag "$_IMAGE" \
        "${_REPO_ROOT}/scripts"
fi

cleanup
if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ]; then
    printf 'Runtime image: %s\n' "$_IMAGE"
    printf 'Compiler build volume: %s\n' "$_BUILD_VOLUME"
    printf 'Runtime cache volume: %s\n' "$_CACHE_VOLUME"
fi
if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ] || \
   [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    exec "$_DOCKER" "${_RUN_ARGS[@]}" "$_IMAGE"
fi
exec "$_DOCKER" "${_RUN_ARGS[@]}" "$_IMAGE" "$_CONTAINER_SCRIPT" "$@"
