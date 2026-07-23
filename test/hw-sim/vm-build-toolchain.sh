#!/bin/bash
# One-shot (run inside the Lima VM): provision + build the current tt-lang
# toolchain (LLVM + tt-metal from the submodules) from a VM-local copy of the
# mounted source. Reproducible new-Mac / post-uplift setup.
#
#   limactl shell ttlang-ttsim -- bash <mounted>/test/hw-sim/vm-build-toolchain.sh
#
# Notes captured from aarch64 bring-up:
#  - tt-metal's install_dependencies.sh downloads an amd64-only OpenMPI-ULFM
#    .deb; --no-distributed skips it (not needed for single-device sim).
#  - A wrong-arch (x86-64) cmake can shadow the apt arm64 cmake in /usr/local/bin.
#  - Clang comes from apt.llvm.org as a versioned package (clang-N); register the
#    newest installed one as bare clang/clang++ for tt-lang's CMake.
#  - tt-metal writes into its own source tree (CPM cache, firmware ELFs); those
#    writes fail over the virtiofs mount, so build from a VM-local ext4 copy.
set -euo pipefail

# Host TT root (the dir holding the tt-lang clone), as mounted in
# the guest. Auto-detect the virtiofs mount that contains tt-lang, or set SRC_HOST.
if [ -z "${SRC_HOST:-}" ]; then
  while IFS= read -r _m; do
    [ -d "$_m/tt-lang" ] && { SRC_HOST="$_m"; break; }
  done < <(findmnt -nrt virtiofs -o TARGET 2>/dev/null)
fi
: "${SRC_HOST:?set SRC_HOST to the mounted TT root (the dir holding tt-lang)}"
VM_LOCAL="${VM_LOCAL:-/var/tmp}"
TOOLCHAIN="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"

echo "=== stop unattended-upgrades (avoid apt-lock races on a fresh VM) ==="
sudo systemctl stop unattended-upgrades apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
for _ in $(seq 1 90); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 2; done

echo "=== tt-metal system deps (skip amd64-only OpenMPI-ULFM on aarch64) ==="
sudo DEBIAN_FRONTEND=noninteractive bash "$SRC_HOST/tt-lang/third-party/tt-metal/install_dependencies.sh" --no-distributed

echo "=== drop wrong-arch cmake shadowing apt arm64 cmake ==="
for t in cmake ctest cpack ccmake; do
  if [ -f "/usr/local/bin/$t" ] && file "/usr/local/bin/$t" | grep -q "x86-64"; then sudo rm -f "/usr/local/bin/$t"; fi
done

echo "=== register unversioned clang/clang++ -> newest installed clang-N ==="
clang_ver=$(ls -1 /usr/bin/clang-[0-9]* 2>/dev/null | sed -E 's#.*/clang-([0-9]+)$#\1#' | sort -n | tail -1)
if [ -n "${clang_ver:-}" ]; then
  sudo update-alternatives --install /usr/bin/clang   clang   "/usr/bin/clang-$clang_ver"   100
  sudo update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-$clang_ver" 100
fi

echo "=== clean slate ==="
rm -rf "$VM_LOCAL/build-toolchain" "$VM_LOCAL/cpmcache" "$VM_LOCAL/tt-lang"
sudo rm -rf "$TOOLCHAIN"; sudo mkdir -p "$TOOLCHAIN"; sudo chown "$(id -un)" "$TOOLCHAIN"

echo "=== copy source to VM-local ext4 (exclude build outputs) ==="
# tar over the live mount can hit "file changed as we read it" (host IDE/indexer)
# -> exit 1; benign for a source copy, but the extractor must succeed (0).
copy_tree() {
  local src="$1" dst="$2"; shift 2
  mkdir -p "$dst"
  set +e
  ( cd "$src" && tar --warning=no-file-changed "$@" -cf - . ) | ( cd "$dst" && tar -xf - )
  local rc_c=${PIPESTATUS[0]} rc_x=${PIPESTATUS[1]}
  set -e
  [ "$rc_x" -eq 0 ] || { echo "extract into $dst failed (rc=$rc_x)"; exit "$rc_x"; }
  [ "$rc_c" -le 1 ] || { echo "read of $src failed (tar rc=$rc_c)"; exit "$rc_c"; }
}
copy_tree "$SRC_HOST/tt-lang" "$VM_LOCAL/tt-lang" --exclude=./build --exclude=./build-toolchain --exclude=.cpmcache --exclude=pre-compiled

echo "=== build toolchain (LLVM + tt-metal from submodules) ==="
cd "$VM_LOCAL/tt-lang"
CMAKE_BINARY_DIR="$VM_LOCAL/build-toolchain" \
TTLANG_TOOLCHAIN_DIR="$TOOLCHAIN" \
CPM_SOURCE_CACHE="$VM_LOCAL/cpmcache" \
  ./scripts/build-and-install.sh --toolchain-only

echo "=== install tt-metal into the toolchain (_ttnn.so etc.) ==="
CMAKE_BINARY_DIR="$VM_LOCAL/build-toolchain" TTLANG_TOOLCHAIN_DIR="$TOOLCHAIN" \
  ./scripts/build-and-install.sh --install-ttmetal

# Reclaim the build scratch (best-effort: the build already succeeded, so a
# cleanup hiccup must not fail the run). The toolchain now lives at $TOOLCHAIN,
# and tt-lang itself is built from your checkout (build-lima), not this copy.
cd /
rm -rf "$VM_LOCAL/tt-lang" "$VM_LOCAL/build-toolchain" "$VM_LOCAL/cpmcache" || true

echo "=== DONE: toolchain at $TOOLCHAIN ==="
