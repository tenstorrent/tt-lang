# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang packaging Dockerfile: packages pre-built toolchains into Docker images.
#
# LLVM, tt-metal, and tt-lang are built OUTSIDE Docker (on the host or CI runner)
# and injected via --build-context. This Dockerfile only handles packaging.
#
# Targets:
#   ird  - Development image with toolchain only (no tt-lang)
#   dist - Pre-built tt-lang for end users
#
# Required build contexts:
#   --build-context ird-toolchain=./path/to/ird-toolchain
#   --build-context dist-toolchain=./path/to/dist-toolchain

# Base image: parameterized so local builds don't shadow registry tags.
ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:latest

# Toolchain build contexts: overridden by --build-context at build time.
FROM scratch AS ird-toolchain
FROM scratch AS dist-toolchain

# =============================================================================
# IRD (for developers who will clone and build their own tt-lang)
# =============================================================================
FROM ${BASE_IMAGE} AS ird
SHELL ["/bin/bash", "-c"]

ENV TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain
ENV DEBIAN_FRONTEND=noninteractive

# Use --chmod to avoid a separate chmod layer that would duplicate the entire
# toolchain in the image.
COPY --from=ird-toolchain --chmod=777 / $TTLANG_TOOLCHAIN_DIR/

# Install Python dev dependencies (black, sphinx, etc.) into the toolchain venv
COPY requirements.txt /tmp/requirements.txt
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN $TTLANG_TOOLCHAIN_DIR/venv/bin/python -m pip install -r /tmp/dev-requirements.txt tt-smi --no-cache-dir && \
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
FROM ${BASE_IMAGE} AS dist
SHELL ["/bin/bash", "-c"]

ENV TTLANG_TOOLCHAIN_DIR=/opt/ttlang-toolchain

COPY --from=dist-toolchain --chmod=777 / $TTLANG_TOOLCHAIN_DIR/

# Install text editors, SSH, and tt-smi
RUN apt-get update && apt-get install -y vim nano openssh-server && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /run/sshd && \
    $TTLANG_TOOLCHAIN_DIR/venv/bin/python -m pip install tt-smi --no-cache-dir

# Auto-activate tt-lang environment on login
RUN printf '%s\n' \
      '# Auto-activate tt-lang environment' \
      'if [ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ]; then' \
      '    export TTLANG_TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}"' \
      '    if [ -f "$TTLANG_TOOLCHAIN_DIR/env/activate" ]; then' \
      '        echo "Activating tt-lang environment (profile.d)..."' \
      '        source "$TTLANG_TOOLCHAIN_DIR/env/activate"' \
      '    fi' \
      'fi' \
      > /etc/profile.d/ttlang.sh && \
    chmod +x /etc/profile.d/ttlang.sh && \
    echo '[ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ] && export TTLANG_TOOLCHAIN_DIR="${TTLANG_TOOLCHAIN_DIR:-/opt/ttlang-toolchain}" && [ -f "$TTLANG_TOOLCHAIN_DIR/env/activate" ] && source "$TTLANG_TOOLCHAIN_DIR/env/activate"' >> /etc/bash.bashrc

# Setup entrypoint
COPY .github/containers/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy examples to /root
WORKDIR /root
RUN cp -r $TTLANG_TOOLCHAIN_DIR/examples .
COPY .github/containers/CONTAINER_README.md /root/README.md

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
