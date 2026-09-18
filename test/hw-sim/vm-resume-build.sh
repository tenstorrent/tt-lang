#!/bin/bash
# Resume ONLY the toolchain build in the VM (no clean/re-copy), from the existing
# VM-local source + build dir. Use this if a detached build was orphaned (e.g. a
# dropped SSH session) -- ninja + ccache make it incremental. Run detached:
#   limactl shell <vm> -- bash -c 'setsid bash <mounted>/test/hw-sim/vm-resume-build.sh </dev/null >/dev/null 2>&1 & disown'
# Writes its exit code to $VM_LOCAL/toolchain-build.exit on completion.
set -uo pipefail
VM_LOCAL="${VM_LOCAL:-/var/tmp}"
MARKER="$VM_LOCAL/toolchain-build.exit"
cd "$VM_LOCAL/tt-lang" || { echo 1 > "$MARKER"; exit 1; }
rc=0
CMAKE_BINARY_DIR="$VM_LOCAL/build-toolchain" \
TTLANG_TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}" \
CPM_SOURCE_CACHE="$VM_LOCAL/cpmcache" \
  ./scripts/build-and-install.sh --toolchain-only > "$VM_LOCAL/toolchain-build.log" 2>&1 || rc=$?
echo "$rc" > "$MARKER"
exit "$rc"
