#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Hardware test runner for tt-lang
# This script sets up the environment and runs hardware tests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTLANG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
HARDWARE_TYPE="${RUNS_ON:-n150}"
BUILD_DIR="${BUILD_DIR:-${TTLANG_ROOT}/build}"
INSTALL_DIR="${INSTALL_DIR:-${TTLANG_ROOT}/tt-mlir-install}"
TEST_FILTER="${TEST_FILTER:-}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${TTLANG_ROOT}/test_reports}"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run hardware tests for tt-lang

OPTIONS:
    -h, --help              Show this help message
    -t, --type HARDWARE     Hardware type (n150, n300, tg, p150) [default: n150]
    -b, --build-dir DIR     Build directory [default: ./build]
    -i, --install-dir DIR   tt-mlir installation directory [default: ./tt-mlir-install]
    -f, --filter PATTERN    Test filter pattern for llvm-lit
    -o, --output-dir DIR    Test output directory [default: ./test_reports]
    -v, --verbose           Enable verbose test output

ENVIRONMENT VARIABLES:
    RUNS_ON                 Hardware type (same as --type)
    BUILD_DIR               Build directory path
    INSTALL_DIR             Installation directory path
    SYSTEM_DESC_PATH        Path to system descriptor .ttsys file
    TEST_FILTER             Test filter pattern
    TTLANG_VERBOSE_PASSES   Set to 1 to enable verbose pass output

EXAMPLES:
    # Run all tests on N150
    $0 --type n150

    # Run specific test
    $0 --filter test_operator_add.py

    # Run with verbose output
    $0 --verbose --type n150

EOF
}

# Parse command line arguments
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--type)
            HARDWARE_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -i|--install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -f|--filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        -o|--output-dir)
            TEST_OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

echo "========================================"
echo "tt-lang Hardware Test Runner"
echo "========================================"
echo "Hardware type: ${HARDWARE_TYPE}"
echo "Build directory: ${BUILD_DIR}"
echo "Install directory: ${INSTALL_DIR}"
echo "Test output directory: ${TEST_OUTPUT_DIR}"
echo ""

# Validate directories
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found: ${BUILD_DIR}"
    exit 1
fi

if [ ! -d "${INSTALL_DIR}" ]; then
    echo "Error: Install directory not found: ${INSTALL_DIR}"
    exit 1
fi

# Set up environment variables
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:/opt/ttmlir-toolchain/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${BUILD_DIR}/python_packages:${PYTHONPATH:-}"

# Generate system descriptor if not provided
if [ -z "${SYSTEM_DESC_PATH}" ] || [ ! -f "${SYSTEM_DESC_PATH}" ]; then
    echo "Generating system descriptor for ${HARDWARE_TYPE}..."

    TTRT_ARTIFACTS_DIR="${TTLANG_ROOT}/ttrt-artifacts"
    mkdir -p "${TTRT_ARTIFACTS_DIR}"
    cd "${TTRT_ARTIFACTS_DIR}"

    if command -v ttrt &> /dev/null; then
        # Generate system descriptor based on hardware type
        case "${HARDWARE_TYPE}" in
            tg|p150)
                ttrt query --save-artifacts --disable-eth-dispatch
                ;;
            *)
                ttrt query --save-artifacts
                ;;
        esac

        # Find the generated system descriptor
        SYSTEM_DESC_PATH=$(find . -name "*.ttsys" -type f | head -1)
        if [ -n "${SYSTEM_DESC_PATH}" ]; then
            SYSTEM_DESC_PATH="${TTRT_ARTIFACTS_DIR}/${SYSTEM_DESC_PATH}"
            export SYSTEM_DESC_PATH
            echo "Generated system descriptor: ${SYSTEM_DESC_PATH}"
        else
            echo "Warning: Failed to generate system descriptor"
        fi
    else
        echo "Warning: ttrt command not found, cannot generate system descriptor"
        echo "Tests requiring system descriptor will be skipped"
    fi

    cd "${TTLANG_ROOT}"
else
    echo "Using provided system descriptor: ${SYSTEM_DESC_PATH}"
fi

# Activate tt-lang environment
if [ -f "${BUILD_DIR}/env/activate" ]; then
    echo "Activating tt-lang environment..."
    source "${BUILD_DIR}/env/activate"
else
    echo "Warning: Environment activation script not found"
fi

# Create test output directory
mkdir -p "${TEST_OUTPUT_DIR}"

# Build llvm-lit command
LIT_CMD="llvm-lit"
LIT_ARGS=()

if [ ${VERBOSE} -eq 1 ]; then
    LIT_ARGS+=("-v")
else
    LIT_ARGS+=("-s")
fi

# Add filter if specified
if [ -n "${TEST_FILTER}" ]; then
    LIT_ARGS+=("--filter=${TEST_FILTER}")
fi

# Add test directory
LIT_ARGS+=("${BUILD_DIR}/test/python/")

# Add JUnit XML output
LIT_ARGS+=("--xunit-xml-output" "${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml")

# Run tests
echo ""
echo "Running hardware tests..."
echo "Command: ${LIT_CMD} ${LIT_ARGS[@]}"
echo ""

# Run tests and capture exit code
set +e
"${LIT_CMD}" "${LIT_ARGS[@]}"
TEST_EXIT_CODE=$?
set -e

# Display results
echo ""
echo "========================================"
echo "Test Results"
echo "========================================"

if [ -f "${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml" ]; then
    echo "Test report generated: ${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml"

    # Extract summary from XML
    if command -v xmllint &> /dev/null; then
        TOTAL=$(xmllint --xpath "string(//testsuite/@tests)" "${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml" 2>/dev/null || echo "?")
        FAILURES=$(xmllint --xpath "string(//testsuite/@failures)" "${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml" 2>/dev/null || echo "?")
        ERRORS=$(xmllint --xpath "string(//testsuite/@errors)" "${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml" 2>/dev/null || echo "?")
        SKIPPED=$(xmllint --xpath "string(//testsuite/@skipped)" "${TEST_OUTPUT_DIR}/report_${HARDWARE_TYPE}.xml" 2>/dev/null || echo "?")

        echo "Total tests: ${TOTAL}"
        echo "Failures: ${FAILURES}"
        echo "Errors: ${ERRORS}"
        echo "Skipped: ${SKIPPED}"
    fi
else
    echo "Warning: Test report not generated"
fi

echo ""
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "✗ Some tests failed (exit code: ${TEST_EXIT_CODE})"
fi
echo "========================================"

exit ${TEST_EXIT_CODE}
