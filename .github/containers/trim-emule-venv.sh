#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <venv-python>" >&2
    exit 2
fi
readonly VENV_PYTHON="$1"

"$VENV_PYTHON" -c '
import sys
if sys.prefix == sys.base_prefix:
    raise SystemExit("expected a virtual environment")
'

# Compiler build and test packages, including the donor's Torch variant, remain
# installed. Remove only documentation, lint, and hardware-management tools.
"$VENV_PYTHON" -m pip uninstall --yes \
    black pre-commit pyright \
    sphinx myst-parser sphinx-rtd-theme sphinx-reredirects sphinxcontrib-mermaid \
    sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp \
    sphinxcontrib-jquery sphinxcontrib-jsmath sphinxcontrib-qthelp \
    sphinxcontrib-serializinghtml \
    alabaster babel docutils imagesize roman-numerals snowballstemmer \
    tt-smi tt-tools-common tt-umd pyluwen

"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" -c 'import torch, numpy, ml_dtypes, nanobind, pybind11, lit, pytest, packaging'
