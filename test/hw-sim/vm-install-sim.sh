#!/bin/bash
# Install a Tenstorrent hardware simulator (libttsim.so) into the toolchain so
# tt-lang runtime tests can run without silicon. Run inside the Lima VM:
#   limactl shell <vm> -- bash <mounted>/test/hw-sim/vm-install-sim.sh
#
# By default it downloads a pinned public ttsim release (tenstorrent/ttsim) for
# the target chip and host architecture -- no build required. To use a custom
# simulator instead (e.g. an internal fork), set one of:
#   TTLANG_SIM_SO=<path>    use this prebuilt libttsim.so as-is
#   TTLANG_SIM_SRC=<dir>    build libttsim.so from this simulator source tree
#
# Other knobs:
#   TTSIM_VERSION   public ttsim release tag to download (default below)
#   TTSIM_CHIP      target chip: wh (wormhole) | bh (blackhole)   (default wh)
#   TTLANG_TOOLCHAIN_DIR   install prefix (default /opt/ttlang-toolchain)
#
# The simulator is staged at $TTLANG_TOOLCHAIN_DIR/sim/{libttsim.so,
# soc_descriptor.yaml}; point TT_METAL_SIMULATOR at that libttsim.so to use it.
set -euo pipefail

TOOLCHAIN="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"
# Pinned to a release ABI-compatible with the tt-metal built into the toolchain;
# bump alongside tt-metal submodule uplifts.
TTSIM_VERSION="${TTSIM_VERSION:-v1.9.6}"
TTSIM_CHIP="${TTSIM_CHIP:-wh}"
SIM_DIR="$TOOLCHAIN/sim"
mkdir -p "$SIM_DIR"

# SOC descriptor for the target chip, taken from the installed tt-metal.
case "$TTSIM_CHIP" in
  wh) soc_yaml="wormhole_b0_80_arch.yaml" ;;
  bh) soc_yaml="blackhole_140_arch.yaml" ;;
  *) echo "unknown TTSIM_CHIP=$TTSIM_CHIP (expected wh or bh)" >&2; exit 1 ;;
esac
soc_src="$TOOLCHAIN/tt-metal/tt_metal/soc_descriptors/$soc_yaml"
[ -f "$soc_src" ] || { echo "SOC descriptor not found: $soc_src (is tt-metal installed?)" >&2; exit 1; }

# Resolve libttsim.so by precedence: prebuilt override, custom source build, then
# the default public ttsim release download.
if [ -n "${TTLANG_SIM_SO:-}" ]; then
  echo "=== using prebuilt simulator: $TTLANG_SIM_SO ==="
  cp "$TTLANG_SIM_SO" "$SIM_DIR/libttsim.so"
elif [ -n "${TTLANG_SIM_SRC:-}" ]; then
  echo "=== building simulator from source: $TTLANG_SIM_SRC (chip=$TTSIM_CHIP) ==="
  ( cd "$TTLANG_SIM_SRC" && ./make.py "src/_out/release_${TTSIM_CHIP}/libttsim.so" )
  cp "$TTLANG_SIM_SRC/src/_out/release_${TTSIM_CHIP}/libttsim.so" "$SIM_DIR/libttsim.so"
else
  case "$(uname -m)" in
    aarch64|arm64) asset="libttsim_${TTSIM_CHIP}_aarch64.so" ;;
    x86_64)        asset="libttsim_${TTSIM_CHIP}.so" ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  url="https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/${asset}"
  echo "=== downloading public ttsim ${TTSIM_VERSION}: ${asset} ==="
  wget -q "$url" -O "$SIM_DIR/libttsim.so" || { echo "download failed: $url" >&2; exit 1; }
fi

cp "$soc_src" "$SIM_DIR/soc_descriptor.yaml"
echo "=== simulator staged at $SIM_DIR ==="
file "$SIM_DIR/libttsim.so"
ls -la "$SIM_DIR"
