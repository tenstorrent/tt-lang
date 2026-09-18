#!/bin/sh
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Compatibility entry point for S3 and per-tt-metal light-wheel workflows.

set -eu

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec "$script_dir/build-manylinux-core-wheel.sh" \
    --ttnn-dep-mode external \
    "$@"
