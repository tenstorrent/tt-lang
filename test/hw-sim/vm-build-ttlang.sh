#!/bin/bash
# Build tt-lang against the prebuilt VM toolchain (LLVM + tt-metal at
# TTLANG_TOOLCHAIN_DIR). Run inside the VM after vm-build-toolchain.sh. Safe to
# run detached (setsid); writes its exit code to $VM_LOCAL/ttlang-build.exit.
set -euo pipefail
VM_LOCAL="${VM_LOCAL:-/var/tmp}"
TOOLCHAIN="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"
LOG="$VM_LOCAL/ttlang-build.log"
cd "$VM_LOCAL/tt-lang"
{
  cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTTLANG_USE_TOOLCHAIN=ON \
    -DTTLANG_TOOLCHAIN_DIR="$TOOLCHAIN" &&
  cmake --build build
} > "$LOG" 2>&1
echo $? > "$VM_LOCAL/ttlang-build.exit"
