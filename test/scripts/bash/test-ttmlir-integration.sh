#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test script for tt-mlir integration scenarios
# Tests all three priority scenarios in order

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_BUILD_DIR="${PROJECT_ROOT}/test-build"

echo "=== Testing tt-mlir Integration Scenarios ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Test build directory: ${TEST_BUILD_DIR}"
echo ""

# Clean up from previous test runs
rm -rf "${TEST_BUILD_DIR}"

# Test Scenario 1: Environment activated - uses build tree
echo "=== Scenario 1: Environment activated (uses build tree) ==="
if [ -z "${TTMLIR_ENV_ACTIVATED:-}" ] || [ -z "${TT_MLIR_HOME:-}" ]; then
    echo "SKIP: TTMLIR_ENV_ACTIVATED or TT_MLIR_HOME not set"
    echo "      This scenario requires tt-mlir environment to be activated"
    echo ""
else
    echo "TTMLIR_ENV_ACTIVATED=${TTMLIR_ENV_ACTIVATED}"
    echo "TT_MLIR_HOME=${TT_MLIR_HOME}"
    BUILD_DIR="${TEST_BUILD_DIR}/scenario1"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    echo "Configuring CMake..."
    if cmake -GNinja "${PROJECT_ROOT}" 2>&1 | tee configure.log; then
        echo "✓ Configuration successful"
        if grep -q "Using pre-built tt-mlir from:" configure.log; then
            echo "✓ Found tt-mlir via build tree"
        else
            echo "✗ Failed to find tt-mlir via build tree"
            exit 1
        fi
    else
        echo "✗ Configuration failed"
        exit 1
    fi
    echo ""
fi

# Test Scenario 2: No environment but tt-mlir installed - uses installation
echo "=== Scenario 2: No environment, tt-mlir installed at toolchain path ==="
BUILD_DIR="${TEST_BUILD_DIR}/scenario2"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Unset environment variables to simulate no environment
unset TTMLIR_ENV_ACTIVATED
unset TT_MLIR_HOME
unset TTMLIR_TOOLCHAIN_DIR
unset TTMLIR_VENV_DIR

echo "Testing with default /opt/ttmlir-toolchain..."
if [ -d "/opt/ttmlir-toolchain/lib/cmake/ttmlir" ]; then
    echo "Configuring CMake (no environment vars)..."
    if cmake -GNinja "${PROJECT_ROOT}" 2>&1 | tee configure.log; then
        echo "✓ Configuration successful"
        if grep -q "Using pre-built tt-mlir from:" configure.log; then
            echo "✓ Found tt-mlir at /opt/ttmlir-toolchain"
        else
            echo "✗ Failed to find tt-mlir at /opt/ttmlir-toolchain"
            exit 1
        fi
    else
        echo "✗ Configuration failed"
        exit 1
    fi
else
    echo "SKIP: /opt/ttmlir-toolchain not found"
    echo "      Install tt-mlir to /opt/ttmlir-toolchain to test this scenario"
fi
echo ""

# Test Scenario 3: Neither - builds private copy with FetchContent
echo "=== Scenario 3: Neither found - builds private copy with FetchContent ==="
BUILD_DIR="${TEST_BUILD_DIR}/scenario3"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Unset environment variables
unset TTMLIR_ENV_ACTIVATED
unset TT_MLIR_HOME
unset TTMLIR_TOOLCHAIN_DIR
unset TTMLIR_VENV_DIR

# Use a custom toolchain dir that doesn't exist
TEST_TOOLCHAIN_DIR="${TEST_BUILD_DIR}/test-toolchain"
rm -rf "${TEST_TOOLCHAIN_DIR}"

echo "Configuring CMake with non-existent toolchain dir..."
echo "TTMLIR_TOOLCHAIN_DIR=${TEST_TOOLCHAIN_DIR}"
if cmake -GNinja -DTTMLIR_TOOLCHAIN_DIR="${TEST_TOOLCHAIN_DIR}" "${PROJECT_ROOT}" 2>&1 | tee configure.log; then
    echo "✓ Configuration successful"
    if grep -q "tt-mlir not found. Building private copy" configure.log; then
        echo "✓ FetchContent fallback triggered"
        if grep -q "Built and using private tt-mlir installation" configure.log; then
            echo "✓ Private tt-mlir build completed"
        else
            echo "✗ Private tt-mlir build failed or not found"
            exit 1
        fi
    else
        echo "✗ FetchContent fallback not triggered (may have found existing installation)"
        echo "  This is OK if tt-mlir was found elsewhere"
    fi
else
    echo "✗ Configuration failed"
    echo "  Note: This may take a long time as it builds tt-mlir from source"
    exit 1
fi
echo ""

echo "=== All scenario tests completed ==="
echo "Test build directories: ${TEST_BUILD_DIR}"
echo "To clean up: rm -rf ${TEST_BUILD_DIR}"

