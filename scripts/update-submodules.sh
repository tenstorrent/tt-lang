#!/usr/bin/env bash
# Update all submodules to the commits recorded in the current branch.
# --force is needed because CMake applies patches to submodule working trees;
# patches are tracked in third-party/patches/ and re-applied on next configure.
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Update top-level submodules (llvm-project, tt-mlir, tt-metal).
# Do NOT use --recursive here: llvm-project's submodules are huge and unneeded.
git submodule update --init --force --depth 1

# tt-metal has nested submodules (tracy, tt_llk, umd) that must be initialized.
git -C third-party/tt-metal submodule update --init --force --depth 1
