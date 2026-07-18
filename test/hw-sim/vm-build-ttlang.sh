#!/bin/bash
# Build tt-lang from the mounted checkout into build-lima/, against the prebuilt
# toolchain (LLVM + tt-metal at TTLANG_TOOLCHAIN_DIR). tt-lang builds fine over
# virtiofs, so this uses your real source tree -- no copy. Run in the VM:
#   limactl shell <vm> -- bash <mounted>/test/hw-sim/vm-build-ttlang.sh
set -euo pipefail
TOOLCHAIN="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"

# Mounted TT root (the dir holding tt-lang). Auto-detect, or set SRC_HOST.
if [ -z "${SRC_HOST:-}" ]; then
  while IFS= read -r _m; do
    [ -d "$_m/tt-lang" ] && { SRC_HOST="$_m"; break; }
  done < <(findmnt -nrt virtiofs -o TARGET 2>/dev/null)
fi
: "${SRC_HOST:?set SRC_HOST to the mounted TT root (the dir holding tt-lang)}"

cd "$SRC_HOST/tt-lang"
cmake -G Ninja -B build-lima \
  -DCMAKE_BUILD_TYPE=Release \
  -DTTLANG_USE_TOOLCHAIN=ON \
  -DTTLANG_TOOLCHAIN_DIR="$TOOLCHAIN"
cmake --build build-lima
