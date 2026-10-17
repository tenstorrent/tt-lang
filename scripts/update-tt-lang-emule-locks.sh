#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Regenerate the lock files that pin the tt-emule runtime image's downloads:
# .github/containers/emule-requirements.lock from requirements-test.txt and
# .github/containers/emule-metal-deps.lock from the manifest's tt-metal
# commit. The lock stages never receive the emulator source.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: scripts/update-tt-lang-emule-locks.sh" >&2
    exit 2
fi

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_REPO_ROOT="$(dirname "$_SCRIPT_DIR")"
readonly _SCRIPT_DIR _REPO_ROOT
readonly _CONTAINERS="${_REPO_ROOT}/.github/containers"
readonly _STACK_MANIFEST="${_REPO_ROOT}/config/tt-lang-emule-stack.json"
readonly _PYTHON="${TTLANG_EMULE_HOST_PYTHON:-python3}"
readonly _DOCKER="${TTLANG_EMULE_DOCKER:-docker}"
readonly _LOCKS=(emule-requirements.lock emule-metal-deps.lock)

_STACK_OUTPUT="$("$_PYTHON" "${_SCRIPT_DIR}/tt-lang-emule-stack.py" \
    --manifest "$_STACK_MANIFEST" emit)"
while IFS=$'\t' read -r _STACK_KEY _STACK_VALUE; do
    case "$_STACK_KEY" in
        TTLANG_METAL_REPOSITORY) _METAL_REPOSITORY="$_STACK_VALUE" ;;
        TTLANG_METAL_COMMIT) _METAL_COMMIT="$_STACK_VALUE" ;;
        TTLANG_EMULE_BASE_IMAGE) _BASE_IMAGE="$_STACK_VALUE" ;;
    esac
done <<< "$_STACK_OUTPUT"

_WORK="$(mktemp -d "${TMPDIR:-/tmp}/tt-lang-emule-locks.XXXXXX")"
trap 'rm -rf -- "$_WORK"' EXIT
mkdir "${_WORK}/stack" "${_WORK}/output"
cp -- "${_REPO_ROOT}/requirements.txt" "${_REPO_ROOT}/requirements-runtime.txt" \
    "${_REPO_ROOT}/requirements-test.txt" "${_WORK}/stack/"

# Cached lock stages would hide upstream changes since they ran, so the locks
# always record what a cold image build downloads now.
"$_DOCKER" build \
    --platform linux/amd64 \
    --target emule-locks \
    --no-cache-filter metal-fetch \
    --no-cache-filter venv-resolve \
    --build-context "tt-lang-stack=${_WORK}/stack" \
    --file "${_CONTAINERS}/Dockerfile.emule" \
    --build-arg "BASE_IMAGE=${_BASE_IMAGE}" \
    --build-arg "TT_METAL_COMMIT=${_METAL_COMMIT}" \
    --build-arg "TT_METAL_SOURCE_URL=${_METAL_REPOSITORY}" \
    --output "type=local,dest=${_WORK}/output" \
    "$_CONTAINERS"

for _LOCK in "${_LOCKS[@]}"; do
    if [ ! -s "${_WORK}/output/${_LOCK}" ]; then
        echo "update-tt-lang-emule-locks: the build produced no ${_LOCK}." >&2
        exit 1
    fi
done
for _LOCK in "${_LOCKS[@]}"; do
    cp -- "${_WORK}/output/${_LOCK}" "${_CONTAINERS}/${_LOCK}"
    printf 'Updated .github/containers/%s\n' "$_LOCK"
done
