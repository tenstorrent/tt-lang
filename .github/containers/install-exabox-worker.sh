#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Install the account and services required by Exabox worker pods.

set -euo pipefail

TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"
OMPI_TAG="${OMPI_TAG:-v5.0.7}"
OMPI_PREFIX="${OMPI_PREFIX:-/opt/openmpi-${OMPI_TAG}-ulfm}"

apt-get update
apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    flex \
    libtool \
    openssh-server \
    sudo \
    libatomic1
rm -rf /var/lib/apt/lists/*

if [ ! -x "$OMPI_PREFIX/bin/prted" ]; then
    ompi_source_dir="$(mktemp -d)"
    trap 'rm -rf "$ompi_source_dir"' EXIT
    git clone --branch "$OMPI_TAG" --depth 1 \
        https://github.com/open-mpi/ompi.git "$ompi_source_dir"
    cd "$ompi_source_dir"
    git submodule update --init --recursive
    grep -rl 'req_complete = false;' --include='*.h' --include='*.c' . | \
        xargs -r sed -i \
            's/req_complete = false;/req_complete = REQUEST_PENDING;/g'
    ./autogen.pl
    ./configure \
        --prefix="$OMPI_PREFIX" \
        --with-ft=ulfm \
        --enable-wrapper-rpath \
        --enable-mpirun-prefix-by-default \
        --disable-mca-dso \
        --disable-dlopen
    make -j"$(nproc)"
    make install
    cd /
    rm -rf "$ompi_source_dir"
    trap - EXIT
fi

if id -u user > /dev/null 2>&1; then
    [ "$(id -u user)" = "1001" ] || {
        echo "Existing user account must have UID 1001." >&2
        exit 1
    }
elif getent passwd 1001 > /dev/null; then
    echo "UID 1001 is already assigned to another account." >&2
    exit 1
else
    useradd --uid 1001 --create-home --shell /bin/bash user
fi

usermod -aG sudo user
echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user
chmod 0440 /etc/sudoers.d/user
mkdir -p /run/sshd
grep -q '^StrictModes no$' /etc/ssh/sshd_config || \
    echo 'StrictModes no' >> /etc/ssh/sshd_config

ln -sf "$TOOLCHAIN_DIR/venv/bin/tt-smi" /usr/local/bin/tt-smi

install -d -m 0777 \
    "$TOOLCHAIN_DIR/tt-metal/build" \
    "$TOOLCHAIN_DIR/tt-metal/build/profiler" \
    "$TOOLCHAIN_DIR/tt-metal/build/profiler/build_wasm" \
    "$TOOLCHAIN_DIR/tt-metal/build/profiler/build_wasm/traces" \
    "$TOOLCHAIN_DIR/tt-metal/generated" \
    "$TOOLCHAIN_DIR/tt-metal/generated/profiler"

test -x /usr/sbin/sshd
test -x /usr/local/bin/tt-smi
test -x "$OMPI_PREFIX/bin/prted"
