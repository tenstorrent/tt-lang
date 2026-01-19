# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# tt-lang distribution image: Multi-stage build with dist/dev targets
#
# Targets:
#   dist - Pre-built tt-lang for end users (ready to import ttl)
#   dev  - Development image with toolchain + extra tools (vim, tmux, nano)

ARG FROM_TAG=latest
ARG MLIR_TAG=latest

# =============================================================================
# Stage 1: Build tt-lang
# =============================================================================
FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:${MLIR_TAG} AS build
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

ADD . /home/build/tt-lang
WORKDIR /home/build/tt-lang

# Show commit info for traceability
RUN git log -1 || echo "Git info not available"

# Build tt-lang using FetchContent (builds tt-mlir from pinned commit)
# Install tt-mlir directly to /opt/ttmlir-toolchain
RUN TTMLIR_COMMIT=$(cat third-party/tt-mlir.commit | tr -d '[:space:]') && \
    cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_C_COMPILER=clang-17 \
    -DTTMLIR_CMAKE_BUILD_TYPE=Release \
    -DTTMLIR_INSTALL_PREFIX=$TTMLIR_TOOLCHAIN_DIR \
    -DTTMLIR_GIT_TAG=$TTMLIR_COMMIT \
    -DTTLANG_ENABLE_BINDINGS_PYTHON=ON

# Build and install tt-lang to toolchain
RUN source build/env/activate && \
    cmake --build build && \
    cmake --install build --prefix $TTMLIR_TOOLCHAIN_DIR

# Copy tt-mlir Python packages that aren't included in install
# Use -L to dereference symlinks (copy actual files, not symlink paths)
# TODO: Remove when tt-mlir install is fixed
RUN mkdir -p $TTMLIR_TOOLCHAIN_DIR/python_packages/ttrt/runtime && \
    cp -prL build/_deps/tt-mlir-build/python_packages/ttrt/runtime/* $TTMLIR_TOOLCHAIN_DIR/python_packages/ttrt/runtime/ 2>/dev/null || true

# Copy ttmlir Python bindings (dereference symlinks with -L)
RUN if [ -d "build/_deps/tt-mlir-build/python_packages/ttmlir" ]; then \
      cp -prL build/_deps/tt-mlir-build/python_packages/ttmlir/* $TTMLIR_TOOLCHAIN_DIR/python_packages/ttmlir/ 2>/dev/null || true; \
    fi

# Normalize tt-mlir installation (replace any remaining symlinks with actual files)
COPY .github/scripts/normalize-ttmlir-install.sh /tmp/normalize-ttmlir-install.sh
RUN bash /tmp/normalize-ttmlir-install.sh $TTMLIR_TOOLCHAIN_DIR && rm /tmp/normalize-ttmlir-install.sh

# Remove unnecessary LLVM/MLIR tools to reduce image size
COPY .github/containers/cleanup-toolchain.sh /tmp/cleanup-toolchain.sh
RUN bash /tmp/cleanup-toolchain.sh $TTMLIR_TOOLCHAIN_DIR && rm /tmp/cleanup-toolchain.sh

# Keep build directory - dist stage needs it for installation

# =============================================================================
# Stage 2: CI (tt-mlir toolchain for CI workflows)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:${FROM_TAG} AS ci
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Copy toolchain (has both tt-mlir and tt-lang)
COPY --from=build $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Remove tt-lang files (CI only needs tt-mlir toolchain)
RUN rm -rf $TTMLIR_TOOLCHAIN_DIR/bin/ttlang-* \
           $TTMLIR_TOOLCHAIN_DIR/python_packages/ttl \
           $TTMLIR_TOOLCHAIN_DIR/python_packages/pykernel \
           $TTMLIR_TOOLCHAIN_DIR/python_packages/sim \
           $TTMLIR_TOOLCHAIN_DIR/examples \
           $TTMLIR_TOOLCHAIN_DIR/test \
           $TTMLIR_TOOLCHAIN_DIR/env 2>/dev/null || true

# Set write permissions
RUN chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

WORKDIR /root

# =============================================================================
# Stage 3: Dist (pre-built tt-lang for users)
# =============================================================================
FROM ghcr.io/tenstorrent/tt-lang/tt-lang-base-ubuntu-22-04:${FROM_TAG} AS dist
SHELL ["/bin/bash", "-c"]

ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# Copy full toolchain with tt-mlir AND tt-lang installed
COPY --from=build $TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR

# Set write permissions
RUN chmod -R 777 $TTMLIR_TOOLCHAIN_DIR

# Install text editors
RUN rm -f /etc/apt/sources.list.d/kitware*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y vim nano && rm -rf /var/lib/apt/lists/*

# Setup entrypoint
COPY .github/containers/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy examples to /root
WORKDIR /root
RUN cp -r $TTMLIR_TOOLCHAIN_DIR/examples .
COPY .github/containers/CONTAINER_README.md /root/README.md

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

# =============================================================================
# Stage 4: IRD (primarily for developmers who will clone and build their own tt-lang)
# =============================================================================
FROM ci AS ird

ENV DEBIAN_FRONTEND=noninteractive

# Install Python dev dependencies (black, sphinx, etc.)
COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN python3.11 -m pip install -r /tmp/dev-requirements.txt --no-cache-dir && \
    rm /tmp/dev-requirements.txt

# Install additional development tools
RUN rm -f /etc/apt/sources.list.d/kitware*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y \
    ssh \
    sudo \
    vim \
    nano \
    tmux \
    psmisc \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Setup entrypoint (same as dist)
COPY .github/containers/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
