# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang distribution image: Multi-stage build with dist/ird targets
#
# Targets:
#   ird  - Development image with toolchain only (no tt-lang; developers build their own)
#   dist - Pre-built tt-lang for end users (ready to import ttl)
#
# Build stages (LLVM + tt-metal are built exactly once):
#   configure       - cmake configure (builds LLVM + tt-metal from submodules)
#   build-toolchain - Finalize toolchain without tt-lang (copy libs, cleanup). Used by ird.
#   build           - Build tt-lang, install, finalize. Used by dist.
#
# Both build-toolchain and build extend the configure stage, so LLVM + tt-metal
# are never built twice regardless of which target is requested.
#
# Cache injection (optional):
#   Pass --build-context llvm-cache=./path and/or --build-context ttmetal-cache=./path
#   to inject pre-built LLVM and tt-metal artifacts. cmake's existing skip-if-exists
#   logic avoids rebuilding when caches are present.

# Cache stages: empty by default, overridden by --build-context when caches are available.
FROM scratch AS llvm-cache
FROM scratch AS ttmetal-cache

# =============================================================================
# Configure: cmake configure (builds LLVM + tt-metal, keeps build dirs)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:latest AS configure
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

ADD . /home/build/tt-lang
WORKDIR /home/build/tt-lang

RUN git log -1 || echo "Git info not available"

COPY .github/scripts/normalize-ttmlir-install.sh /tmp/normalize-ttmlir-install.sh
COPY .github/containers/cleanup-toolchain.sh /tmp/cleanup-toolchain.sh
COPY .github/containers/build-and-install.sh /tmp/build-and-install.sh
RUN --mount=from=llvm-cache,target=/tmp/llvm-cache \
    --mount=from=ttmetal-cache,target=/tmp/ttmetal-cache \
    chmod +x /tmp/build-and-install.sh && \
    /tmp/build-and-install.sh --configure-only \
        --llvm-cache /tmp/llvm-cache \
        --ttmetal-cache /tmp/ttmetal-cache

# =============================================================================
# Build toolchain only (for ird): finalize without building tt-lang
# =============================================================================
FROM configure AS build-toolchain
RUN chmod +x /tmp/build-and-install.sh && \
    /tmp/build-and-install.sh --finalize-only

# =============================================================================
# Build tt-lang (for dist): build + install + finalize
# =============================================================================
FROM configure AS build
RUN chmod +x /tmp/build-and-install.sh && \
    /tmp/build-and-install.sh --build-and-finalize

# =============================================================================
# IRD (for developers who will clone and build their own tt-lang)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:latest AS ird
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
ENV DEBIAN_FRONTEND=noninteractive

# Copy toolchain (LLVM + tt-metal + tt-mlir only; tt-lang was never built)
COPY --from=build-toolchain $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Set write permissions
RUN chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Install Python dev dependencies (black, sphinx, etc.) into the toolchain venv
COPY requirements.txt /tmp/requirements.txt
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN $TTMLIR_TOOLCHAIN_DIR/venv/bin/python -m pip install -r /tmp/dev-requirements.txt tt-smi --no-cache-dir && \
    rm /tmp/requirements.txt /tmp/dev-requirements.txt && \
    apt-get update && apt-get install -y \
    ssh \
    sudo \
    vim \
    nano \
    tmux \
    psmisc \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Setup entrypoint
COPY .github/containers/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /root
COPY .github/containers/IRD_README.md /root/README.md

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

# =============================================================================
# Dist (pre-built tt-lang for users)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:latest AS dist
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Copy full toolchain with LLVM + tt-metal + tt-mlir AND tt-lang installed
COPY --from=build $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Set write permissions
RUN chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Install text editors, SSH, and tt-smi
RUN apt-get update && apt-get install -y vim nano openssh-server && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /run/sshd && \
    $TTMLIR_TOOLCHAIN_DIR/venv/bin/python -m pip install tt-smi --no-cache-dir

# Auto-activate tt-lang environment on login (works even when entrypoint is overridden)
# Add to /etc/profile.d for login shells and /etc/bash.bashrc for interactive shells
RUN printf '%s\n' \
      '# Auto-activate tt-lang environment' \
      'if [ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ]; then' \
      '    export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"' \
      '    if [ -f "$TTMLIR_TOOLCHAIN_DIR/env/activate" ]; then' \
      '        echo "Activating tt-lang environment (profile.d)..."' \
      '        source "$TTMLIR_TOOLCHAIN_DIR/env/activate"' \
      '    fi' \
      'fi' \
      > /etc/profile.d/ttlang.sh && \
    chmod +x /etc/profile.d/ttlang.sh && \
    echo '[ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ] && export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}" && [ -f "$TTMLIR_TOOLCHAIN_DIR/env/activate" ] && source "$TTMLIR_TOOLCHAIN_DIR/env/activate"' >> /etc/bash.bashrc

# Setup entrypoint
COPY .github/containers/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy examples to /root
WORKDIR /root
RUN cp -r $TTMLIR_TOOLCHAIN_DIR/examples .
COPY .github/containers/CONTAINER_README.md /root/README.md

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
