#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Paths whose change can affect manylinux wheel-builder image content or
# assembly. This extends the shared IRD/dist image input list with the driver
# scripts that control the manylinux builder Docker invocation.

wheel_builder_path_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=uplift-paths.sh
source "$wheel_builder_path_script_dir/uplift-paths.sh"

WHEEL_BUILDER_TAG_PATHS=(
    "${UPLIFT_PATHS[@]}"
    .github/containers/Dockerfile.wheel-manylinux-2-34
    .github/containers/CMakeLists.wheel-toolchain
    .github/containers/build-wheel-manylinux-images.sh
    .github/containers/cache-wheel-manylinux-component.sh
)
