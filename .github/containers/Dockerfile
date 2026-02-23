# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang distribution image: Multi-stage build with dist/ird targets
#
# Targets:
#   ird  - Development image with toolchain only (no tt-lang; developers build their own)
#   dist - Pre-built tt-lang for end users (ready to import ttl)
#
# Build stages:
#   build-toolchain - Configure-only (builds tt-mlir via FetchContent, skips tt-lang). Used by ird.
#   build           - Full tt-lang build (configure + build + install). Used by dist.
#
# Docker only runs stages in the dependency chain of the requested target,
# so `--target ird` never builds tt-lang and `--target dist` never runs build-toolchain.

ARG FROM_TAG=latest
ARG MLIR_TAG=latest

# =============================================================================
# Build toolchain only (for ird)
# CMake configure triggers FetchContent which builds and installs tt-mlir,
# but tt-lang itself is not built or installed.
# =============================================================================
FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:${MLIR_TAG} AS build-toolchain
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

ADD . /home/build/tt-lang
WORKDIR /home/build/tt-lang

RUN git log -1 || echo "Git info not available"

COPY .github/scripts/normalize-ttmlir-install.sh /tmp/normalize-ttmlir-install.sh
COPY .github/containers/cleanup-toolchain.sh /tmp/cleanup-toolchain.sh
COPY .github/containers/build-and-install.sh /tmp/build-and-install.sh
RUN chmod +x /tmp/build-and-install.sh && /tmp/build-and-install.sh --toolchain-only && rm /tmp/build-and-install.sh

# =============================================================================
# Build tt-lang (for dist)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:${MLIR_TAG} AS build
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

ADD . /home/build/tt-lang
WORKDIR /home/build/tt-lang

# Show commit info for traceability
RUN git log -1 || echo "Git info not available"

# Configure, build, install, and cleanup in ONE RUN command to avoid Docker layer bloat
# Docker keeps all layers during build - by combining these, the huge build
# directory (50-80GB) never gets committed to a layer
COPY .github/scripts/normalize-ttmlir-install.sh /tmp/normalize-ttmlir-install.sh
COPY .github/containers/cleanup-toolchain.sh /tmp/cleanup-toolchain.sh
COPY .github/containers/build-and-install.sh /tmp/build-and-install.sh
RUN chmod +x /tmp/build-and-install.sh && /tmp/build-and-install.sh && rm /tmp/build-and-install.sh

# =============================================================================
# IRD (for developers who will clone and build their own tt-lang)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:${FROM_TAG} AS ird
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
ENV DEBIAN_FRONTEND=noninteractive

# Copy toolchain (tt-mlir only; tt-lang was never built)
COPY --from=build-toolchain $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Set write permissions
RUN chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Install Python dev dependencies (black, sphinx, etc.) into the toolchain venv
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN $TTMLIR_TOOLCHAIN_DIR/venv/bin/python -m pip install -r /tmp/dev-requirements.txt tt-smi --no-cache-dir && \
    rm /tmp/dev-requirements.txt && \
    (rm -f /etc/apt/sources.list.d/kitware*.list 2>/dev/null || true) && \
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
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:${FROM_TAG} AS dist
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Copy full toolchain with tt-mlir AND tt-lang installed
COPY --from=build $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Set write permissions
RUN chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Install text editors, SSH, and tt-smi
RUN rm -f /etc/apt/sources.list.d/kitware*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y vim nano openssh-server && rm -rf /var/lib/apt/lists/* && \
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
