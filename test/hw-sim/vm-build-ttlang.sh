#!/bin/bash
# Build tt-lang from the mounted checkout into build-lima/, against the prebuilt
# toolchain (LLVM + tt-metal at TTLANG_TOOLCHAIN_DIR). tt-lang builds fine over
# virtiofs, so this uses your real source tree -- no copy. Run in the VM:
#   limactl shell <vm> -- bash <mounted>/test/hw-sim/vm-build-ttlang.sh
set -euo pipefail
TOOLCHAIN="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"

# check-ttlang-me2e/-pytest parallelize only when TT_METAL_SIMULATOR is defined
# at configure time. This script builds for the sim, so define it here (honoring
# an existing value) pointing at the staged simulator.
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-$TOOLCHAIN/sim/libttsim.so}"

# pytest-xdist worker count for those runs, written to build-lima's CMake cache.
# Each worker starts its own libttsim.so device (~2-3 GiB), so the default of 2
# suits the 16 GiB VM; raise it with TTLANG_SIM_PYTEST_JOBS=<N> if it has more.
SIM_PYTEST_JOBS="${TTLANG_SIM_PYTEST_JOBS:-2}"

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
  -DTTLANG_TOOLCHAIN_DIR="$TOOLCHAIN" \
  -DTTLANG_SIM_PYTEST_JOBS="$SIM_PYTEST_JOBS"
cmake --build build-lima
