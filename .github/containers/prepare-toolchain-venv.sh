#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

toolchain_dir="${1:?Usage: prepare-toolchain-venv.sh TOOLCHAIN_DIR}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_python="$toolchain_dir/venv/bin/python"

# Seed the packaging venv before configure installs the general requirements.
python3.12 -m venv "$toolchain_dir/venv"
"$venv_python" -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch
"$venv_python" "$script_dir/check-cpu-torch.py"
