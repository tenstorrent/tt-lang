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
ARG BASE_IMAGE

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

# Split toolchain into multiple layers for parallel pulls.
# Largest directories get their own layer; the rest go in one layer.
COPY --from=ird-toolchain --chmod=777 /bin $TTLANG_TOOLCHAIN_DIR/bin/
COPY --from=ird-toolchain --chmod=777 /venv $TTLANG_TOOLCHAIN_DIR/venv/
COPY --from=ird-toolchain --chmod=777 /lib $TTLANG_TOOLCHAIN_DIR/lib/
COPY --from=ird-toolchain --chmod=777 /tt-metal $TTLANG_TOOLCHAIN_DIR/tt-metal/
# Remaining small directories (~445MB combined).
COPY --from=ird-toolchain --chmod=777 /python_packages $TTLANG_TOOLCHAIN_DIR/python_packages/
COPY --from=ird-toolchain --chmod=777 /include $TTLANG_TOOLCHAIN_DIR/include/
COPY --from=ird-toolchain --chmod=777 /share $TTLANG_TOOLCHAIN_DIR/share/
COPY --from=ird-toolchain --chmod=777 /src $TTLANG_TOOLCHAIN_DIR/src/

# Install Python dev dependencies (black, sphinx, etc.) into the toolchain venv
COPY requirements.txt /tmp/requirements.txt
COPY docs/requirements.txt /tmp/docs/requirements.txt
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN $TTLANG_TOOLCHAIN_DIR/venv/bin/python -m pip install -r /tmp/dev-requirements.txt tt-smi --no-cache-dir && \
    rm -rf /tmp/requirements.txt /tmp/docs /tmp/dev-requirements.txt && \
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

# Split toolchain into multiple layers for parallel pulls.
COPY --from=dist-toolchain --chmod=777 /bin $TTLANG_TOOLCHAIN_DIR/bin/
COPY --from=dist-toolchain --chmod=777 /venv $TTLANG_TOOLCHAIN_DIR/venv/
COPY --from=dist-toolchain --chmod=777 /lib $TTLANG_TOOLCHAIN_DIR/lib/
COPY --from=dist-toolchain --chmod=777 /tt-metal $TTLANG_TOOLCHAIN_DIR/tt-metal/
RUN mkdir -p $TTLANG_TOOLCHAIN_DIR/tt-metal/generated && chmod 777 $TTLANG_TOOLCHAIN_DIR/tt-metal/generated
COPY --from=dist-toolchain --chmod=777 /python_packages $TTLANG_TOOLCHAIN_DIR/python_packages/
COPY --from=dist-toolchain --chmod=777 /include $TTLANG_TOOLCHAIN_DIR/include/
COPY --from=dist-toolchain --chmod=777 /share $TTLANG_TOOLCHAIN_DIR/share/
# Dist-only: installed tt-lang (examples, tests, env).
COPY --from=dist-toolchain --chmod=777 /examples $TTLANG_TOOLCHAIN_DIR/examples/
COPY --from=dist-toolchain --chmod=777 /test $TTLANG_TOOLCHAIN_DIR/test/
COPY --from=dist-toolchain --chmod=777 /env $TTLANG_TOOLCHAIN_DIR/env/

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
