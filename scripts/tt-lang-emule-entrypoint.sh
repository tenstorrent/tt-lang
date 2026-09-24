#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly TTLANG_SOURCE_DIR="${TTLANG_EMULE_SOURCE_DIR:-/workspace}"
readonly TTLANG_BUILD_DIR="${TTLANG_EMULE_BUILD_DIR:-/ttlang-build}"
readonly TT_METAL_SOURCE_DIR="/opt/tt-emule-runtime/tt-metal"
readonly TT_METAL_BUILD_DIR="${TT_METAL_SOURCE_DIR}/build_emule"

if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ] && \
   [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    echo "tt-lang emule container: installation and shell modes are mutually exclusive." >&2
    exit 2
fi
if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ] || \
   [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    if [ "$#" -ne 0 ]; then
        echo "tt-lang emule container: installation and shell modes do not accept script arguments." >&2
        exit 2
    fi
else
    if [ "$#" -eq 0 ]; then
        echo "tt-lang emule container: no Python script was provided." >&2
        exit 2
    fi
    if [ ! -f "$1" ]; then
        echo "tt-lang emule container: script not found: $1" >&2
        exit 2
    fi
fi

for _TARGET_SETTING in TT_METAL_MOCK_CLUSTER_DESC_PATH \
    TT_METAL_ALLOCATOR_MODE_HYBRID MESH_DEVICE; do
    if [ -z "${!_TARGET_SETTING:-}" ]; then
        echo "tt-lang emule container: required target setting ${_TARGET_SETTING} is missing; use the host launcher or installer." >&2
        exit 1
    fi
done

export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH TT_METAL_ALLOCATOR_MODE_HYBRID MESH_DEVICE
export EMULE_FABRIC8="${EMULE_FABRIC8:-1}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/tt-metal-cache}"
export TT_EMULE_JIT_CACHE_DIR="${TT_EMULE_JIT_CACHE_DIR:-${TT_METAL_CACHE}/emule-jit}"
unset TTLANG_COMPILE_ONLY TTLANG_SIM_ONLY

if [ ! -f "${TT_METAL_MOCK_CLUSTER_DESC_PATH}" ]; then
    echo "tt-lang emule container: cluster descriptor not found: ${TT_METAL_MOCK_CLUSTER_DESC_PATH}" >&2
    exit 1
fi

_EXPECTED_LLVM_SHA="${TTLANG_EMULE_EXPECTED_LLVM_SHA:-}"
_LLVM_REVISION_HEADER=/opt/ttlang-toolchain/include/llvm/Support/VCSRevision.h
_ACTUAL_LLVM_SHA=""
if [ -f "$_LLVM_REVISION_HEADER" ]; then
    _ACTUAL_LLVM_SHA="$(sed -nE \
        's/^[[:space:]]*#define[[:space:]]+LLVM_REVISION[[:space:]]+(R)?"\(?([0-9a-f]{7,40})\)?"[[:space:]]*$/\2/p' \
        "$_LLVM_REVISION_HEADER")"
fi
if [ "${#_EXPECTED_LLVM_SHA}" -ne 40 ] || \
   [[ "$_EXPECTED_LLVM_SHA" == *[!0-9a-f]* ]]; then
    echo "tt-lang emule container: expected LLVM revision is missing or invalid; use ./bin/tt-lang-sim --backend=emule." >&2
    exit 1
fi
if [ -z "$_ACTUAL_LLVM_SHA" ] || \
   [[ "$_EXPECTED_LLVM_SHA" != "$_ACTUAL_LLVM_SHA"* ]]; then
    echo "tt-lang emule container: runtime LLVM revision does not match the compiler checkout." >&2
    echo "  expected: $_EXPECTED_LLVM_SHA" >&2
    echo "  installed: ${_ACTUAL_LLVM_SHA:-unknown ($_LLVM_REVISION_HEADER)}" >&2
    echo "Reinstall the compiler-backed environment with scripts/install-tt-lang-emule.sh." >&2
    exit 1
fi

readonly _COMPILER_MARKER="${TTLANG_BUILD_DIR}/.ttlang-emule-source-fingerprint"
if [ "${TTLANG_EMULE_INSTALL:-0}" = "1" ]; then
    if [ -z "${TTLANG_EMULE_SOURCE_FINGERPRINT:-}" ] || \
       [ -z "${TTLANG_EMULE_COMPILER_SHA:-}" ]; then
        echo "tt-lang emule container: installer source identity is missing." >&2
        exit 1
    fi
    _COMPILER_MARKER_TEMP="${_COMPILER_MARKER}.tmp"
    rm -f -- "$_COMPILER_MARKER" "$_COMPILER_MARKER_TEMP"
    cmake -G Ninja -S "$TTLANG_SOURCE_DIR" -B "$TTLANG_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang-20 \
        -DCMAKE_CXX_COMPILER=clang++-20 \
        -DLLVM_USE_LINKER=lld-20 \
        -DTTLANG_USE_TOOLCHAIN=ON \
        -DTTLANG_USE_TOOLCHAIN_TTMETAL=OFF \
        -DTTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain \
        -DTTLANG_EXTERNAL_TT_METAL_DIR="$TT_METAL_SOURCE_DIR" \
        -DTTLANG_EXTERNAL_TT_METAL_BUILD_DIR="$TT_METAL_BUILD_DIR" \
        -DTTLANG_ENABLE_PERF_TRACE=OFF
    _BUILD_JOBS="$(nproc)"
    _MEMORY_MAX_PATH="${_TTLANG_EMULE_CGROUP_MEMORY_MAX_PATH:-/sys/fs/cgroup/memory.max}"
    _MEMINFO_PATH="${_TTLANG_EMULE_MEMINFO_PATH:-/proc/meminfo}"
    _MEMORY_BYTES=""
    if [ -r "$_MEMORY_MAX_PATH" ]; then
        _MEMORY_MAX="$(cat "$_MEMORY_MAX_PATH")"
        if [[ "$_MEMORY_MAX" =~ ^[0-9]+$ ]]; then
            _MEMORY_BYTES="$_MEMORY_MAX"
        fi
    fi
    if [ -r "$_MEMINFO_PATH" ]; then
        _MEMTOTAL_KIB="$(awk '$1 == "MemTotal:" { print $2; exit }' "$_MEMINFO_PATH")"
        if [[ "$_MEMTOTAL_KIB" =~ ^[0-9]+$ ]]; then
            _MEMTOTAL_BYTES="$((_MEMTOTAL_KIB * 1024))"
            if [ -z "$_MEMORY_BYTES" ] || \
               [ "$_MEMTOTAL_BYTES" -lt "$_MEMORY_BYTES" ]; then
                _MEMORY_BYTES="$_MEMTOTAL_BYTES"
            fi
        fi
    fi
    if [ -n "$_MEMORY_BYTES" ]; then
        # MLIR translation units can use close to 2 GiB each.  Docker can
        # expose every host CPU while granting the VM or container much less
        # memory, so CPU-count parallelism alone can OOM without a useful
        # compiler diagnostic.
        _MEMORY_JOBS="$((_MEMORY_BYTES / (2 * 1024 * 1024 * 1024)))"
        if [ "$_MEMORY_JOBS" -lt 1 ]; then
            _MEMORY_JOBS=1
        fi
        if [ "$_MEMORY_JOBS" -lt "$_BUILD_JOBS" ]; then
            _BUILD_JOBS="$_MEMORY_JOBS"
        fi
    fi
    echo "Building TT-Lang with ${_BUILD_JOBS} parallel job(s)."
    cmake --build "$TTLANG_BUILD_DIR" --parallel "$_BUILD_JOBS"
    printf '%s\n' "${TTLANG_EMULE_SOURCE_FINGERPRINT}" > \
        "$_COMPILER_MARKER_TEMP"
    mv -- "$_COMPILER_MARKER_TEMP" "$_COMPILER_MARKER"
    echo "Installed compiler-backed emule environment for ${TTLANG_EMULE_COMPILER_SHA}."
    exit 0
fi

if [ ! -f "$_COMPILER_MARKER" ]; then
    echo "tt-lang emule container: the compiler environment is not installed." >&2
    echo "Run scripts/install-tt-lang-emule.sh on the host, then retry." >&2
    exit 1
fi
_INSTALLED_SOURCE_FINGERPRINT="$(cat "$_COMPILER_MARKER")"
if [ "$_INSTALLED_SOURCE_FINGERPRINT" != "${TTLANG_EMULE_SOURCE_FINGERPRINT:-}" ]; then
    echo "tt-lang emule container: the installed compiler does not match this checkout." >&2
    echo "Reinstall with scripts/install-tt-lang-emule.sh." >&2
    exit 1
fi
if [ ! -f "${TTLANG_BUILD_DIR}/env/activate" ]; then
    echo "tt-lang emule container: the installed compiler environment is incomplete." >&2
    echo "Reinstall with scripts/install-tt-lang-emule.sh." >&2
    exit 1
fi

# activate is generated for bash and may legitimately reference variables that
# are absent from a non-interactive container shell.
set +u
source "${TTLANG_BUILD_DIR}/env/activate"
set -u

unset TTLANG_COMPILE_ONLY TTLANG_SIM_ONLY

if [ "${TTLANG_EMULE_SHELL:-0}" = "1" ]; then
    exec /bin/bash --noprofile --norc
fi
exec python "$@"
