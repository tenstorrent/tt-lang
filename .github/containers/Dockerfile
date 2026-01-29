# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang container: Self-contained multi-stage build
#
# Targets:
#   dev  - Interactive Research & Development: tt-mlir toolchain + dev tools
#   user - Distribution: dev + pre-built tt-lang (ready to import ttl)
#
# Build context requirements:
#   ttmlir-toolchain - Pre-built LLVM + tt-mlir toolchain from CI cache
#                      (passed via --build-context ttmlir-toolchain=<dir>)
#
# Layer optimization:
#   - Dependencies ordered from least to most frequently changing
#   - apt-get and pip installs combined into single layers
#   - user extends dev to reuse layers

# =============================================================================
# Stage 0: Base with build dependencies
# =============================================================================
FROM ubuntu:22.04 AS base
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies in single layer (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        wget \
        gnupg \
        git \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" > /etc/apt/sources.list.d/llvm.list \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        lld \
        clang-17 \
        clang++-17 \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3.11 -m pip install --upgrade pip --no-cache-dir && \
    python3.11 -m pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    python3.11 -m pip install -r /tmp/requirements.txt --no-cache-dir && \
    rm /tmp/requirements.txt

# =============================================================================
# Stage 1: Build tt-lang (intermediate stage, not exported)
# =============================================================================
FROM base AS build

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Copy pre-built toolchain from build context
COPY --from=ttmlir-toolchain / $TTMLIR_TOOLCHAIN_DIR

# Copy tt-lang source
ADD . /home/build/tt-lang
WORKDIR /home/build/tt-lang

# Show commit info for traceability
RUN git log -1 || echo "Git info not available"

# Build and install tt-lang
COPY .github/containers/build-and-install.sh /tmp/build-and-install.sh
RUN chmod +x /tmp/build-and-install.sh && /tmp/build-and-install.sh && rm /tmp/build-and-install.sh

# =============================================================================
# Stage 2: IRD (Interactive Research & Development)
# Complete toolchain for developers to build tt-lang from source
# =============================================================================
FROM base AS dev

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Copy toolchain (tt-mlir only, no tt-lang - developers build it themselves)
COPY --from=ttmlir-toolchain / $TTMLIR_TOOLCHAIN_DIR

# Install dev tools
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN python3.11 -m pip install -r /tmp/dev-requirements.txt --no-cache-dir \
    && rm /tmp/dev-requirements.txt \
    && apt-get update && apt-get install -y --no-install-recommends \
        ssh \
        openssh-server \
        sudo \
        vim \
        nano \
        tmux \
        psmisc \
        bash-completion \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Auto-activate environment on login
RUN printf '%s\n' \
      '# Auto-activate tt-lang environment' \
      'if [ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ]; then' \
      '    export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"' \
      '    if [ -f "$TTMLIR_TOOLCHAIN_DIR/venv/bin/activate" ]; then' \
      '        source "$TTMLIR_TOOLCHAIN_DIR/venv/bin/activate"' \
      '    fi' \
      'fi' \
      > /etc/profile.d/ttlang.sh \
    && chmod +x /etc/profile.d/ttlang.sh \
    && echo '[ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ] && export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}" && [ -f "$TTMLIR_TOOLCHAIN_DIR/venv/bin/activate" ] && source "$TTMLIR_TOOLCHAIN_DIR/venv/bin/activate"' >> /etc/bash.bashrc

COPY .github/containers/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /root

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

# =============================================================================
# Stage 3: Dist (Distribution image for end users)
# IRD + pre-built tt-lang, ready to import ttl
# =============================================================================
FROM dev AS user

# Copy tt-lang from build stage (overwrites toolchain-only copy from dev)
COPY --from=build $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Update activation to use tt-lang's env/activate
RUN printf '%s\n' \
      '# Auto-activate tt-lang environment' \
      'if [ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ]; then' \
      '    export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"' \
      '    if [ -f "$TTMLIR_TOOLCHAIN_DIR/env/activate" ]; then' \
      '        echo "Activating tt-lang environment..."' \
      '        source "$TTMLIR_TOOLCHAIN_DIR/env/activate"' \
      '    fi' \
      'fi' \
      > /etc/profile.d/ttlang.sh \
    && chmod +x /etc/profile.d/ttlang.sh \
    && echo '[ "${TTLANG_ENV_ACTIVATED:-0}" != "1" ] && export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}" && [ -f "$TTMLIR_TOOLCHAIN_DIR/env/activate" ] && source "$TTMLIR_TOOLCHAIN_DIR/env/activate"' >> /etc/bash.bashrc \
    && chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Copy examples to /root
RUN cp -r $TTMLIR_TOOLCHAIN_DIR/examples . 2>/dev/null || true
COPY .github/containers/CONTAINER_README.md /root/README.md
