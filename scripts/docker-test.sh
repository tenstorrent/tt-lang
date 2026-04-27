#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run tests inside a Docker container using the local build.
# Mounts the source tree at its original path so build/env/activate
# works without modification.
#
# Usage:
#   scripts/docker-test.sh [OPTIONS] [TARGET | -- COMMAND...]
#
# Options:
#   --build-dir DIR   Build directory (default: build-dev)
#   --image IMAGE     Docker image to use
#
# Targets:
#   mlir        MLIR dialect lit tests
#   python-lit  Python lit tests
#   me2e        Metal end-to-end tests (pytest)
#   sim         Simulator tests (pytest)
#   pytest      Python unit tests (pytest test/python/)
#   all         Run all test suites
#   (none)      Interactive shell
#
# Examples:
#   scripts/docker-test.sh                      # interactive shell
#   scripts/docker-test.sh all                  # run all tests
#   scripts/docker-test.sh mlir                 # MLIR dialect tests
#   scripts/docker-test.sh me2e                 # metal end-to-end tests
#   scripts/docker-test.sh -- pytest test/me2e/ # arbitrary command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
BUILD_DIR="build-dev"
DOCKER_TAG="${DOCKER_TAG:-latest}"
IMAGE="${DOCKER_IMAGE:-ghcr.io/tenstorrent/tt-lang/tt-lang-ird-ubuntu-22-04:$DOCKER_TAG}"
CONTAINER_NAME="${USER}-test-$$"
TARGET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option '$1'"
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            break
            ;;
    esac
done

ABS_BUILD="$REPO_DIR/$BUILD_DIR"

# Verify build directory exists
if [[ ! -f "$ABS_BUILD/env/activate" ]]; then
    echo "Error: $ABS_BUILD/env/activate not found."
    echo "Build tt-lang first, or specify --build-dir."
    exit 1
fi

PREAMBLE="source '$ABS_BUILD/env/activate' && cd '$REPO_DIR'"

# Map target names to test commands.
# Cannot use ninja targets because the container has different cmake/compilers
# than the host, which triggers an unwanted cmake reconfigure.
TEST_DIR="$ABS_BUILD/test"
case "$TARGET" in
    mlir)
        USER_CMD="llvm-lit -sv -j1 '$TEST_DIR/ttlang/'"
        ;;
    python-lit)
        USER_CMD="llvm-lit -sv -j1 '$TEST_DIR/python/'"
        ;;
    me2e)
        USER_CMD="pytest test/me2e/ -v"
        ;;
    sim)
        USER_CMD="pytest test/sim/ -v --run-slow"
        ;;
    pytest)
        USER_CMD="pytest test/python/ -v --tb=short"
        ;;
    all)
        USER_CMD="echo '=== mlir ===' && llvm-lit -sv -j1 '$TEST_DIR/ttlang/'"
        USER_CMD="$USER_CMD && echo '=== me2e ===' && pytest test/me2e/ -v --tb=short"
        USER_CMD="$USER_CMD && echo '=== pytest ===' && pytest test/python/ -v --tb=short"
        USER_CMD="$USER_CMD && echo '=== python-lit ===' && llvm-lit -sv -j1 '$TEST_DIR/python/'"
        USER_CMD="$USER_CMD && echo '=== All tests completed ==='"
        ;;
    "")
        # No target specified — interactive shell or custom command
        ;;
    *)
        echo "Error: unknown target '$TARGET'"
        echo "Available targets: mlir, python-lit, me2e, sim, pytest, all"
        exit 1
        ;;
esac

# Build the command to run inside the container
if [[ -n "$TARGET" ]]; then
    DOCKER_CMD=(bash -c "$PREAMBLE && $USER_CMD")
elif [[ $# -eq 0 ]]; then
    DOCKER_CMD=(bash -c "$PREAMBLE && exec bash")
else
    DOCKER_CMD=(bash -c "$PREAMBLE && $*")
fi

# Use -it only when stdin is a terminal
DOCKER_TTY_FLAGS=()
if [[ -t 0 ]]; then
    DOCKER_TTY_FLAGS=(-it)
fi

exec sudo docker run "${DOCKER_TTY_FLAGS[@]}" --rm \
    --name "$CONTAINER_NAME" \
    -v "$REPO_DIR:$REPO_DIR" \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    --device=/dev/tenstorrent/0:/dev/tenstorrent/0 \
    -w "$REPO_DIR" \
    "$IMAGE" \
    "${DOCKER_CMD[@]}"
