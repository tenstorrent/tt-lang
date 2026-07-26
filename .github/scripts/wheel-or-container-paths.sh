#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Paths whose change should trigger PR-time wheel build + dist-container
# tutorial coverage. Every container input requires that coverage, with
# additional wheel-only inputs appended here.

wheel_path_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=uplift-paths.sh
source "$wheel_path_script_dir/uplift-paths.sh"

WHEEL_OR_CONTAINER_PATHS=(
    "${UPLIFT_PATHS[@]}"
    .github/actions/setup-wheel-image-build
    .github/containers/build-wheel-manylinux-images.sh
    .github/containers/cache-wheel-manylinux-component.sh
    .github/scripts/build-manylinux-core-wheel.sh
    .github/scripts/build-manylinux-wheel-set-member.sh
    .github/scripts/build-s3-light-core-wheel.sh
    .github/scripts/build-s3-light-metapackage-wheel.sh
    .github/scripts/check-installed-ttnn.py
    .github/scripts/check-wheel-ttnn-metadata.py
    .github/scripts/lib/docker-image-utils.sh
    .github/scripts/resolve-wheel-builder-images.sh
    .github/scripts/resolve-wheel-versions.sh
    .github/scripts/run-tutorials.sh
    .github/scripts/smoke-test-wheel.py
    .github/scripts/test-manylinux-wheel.sh
    .github/scripts/test-s3-light-wheels.sh
    .github/scripts/validate-manylinux-wheel-inputs.sh
    .github/scripts/verify-manylinux-wheel-set.sh
    .github/workflows/call-build-manylinux-wheels.yml
    .github/workflows/call-build-wheel-images.yml
    .github/workflows/call-test-manylinux-wheels.yml
    scripts/build-s3-light-wheels-local.sh
    bin
    examples
    packaging
    pyproject.toml
    setup.py
    python/CMakeLists.txt
    python/setup.py
)
