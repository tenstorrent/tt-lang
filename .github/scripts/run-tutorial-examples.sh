#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run all tutorial examples to verify they compile and execute successfully.
# This script should be run with hardware access (requires ttnn and device).

set -euo pipefail

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

EXAMPLES_DIR="examples/tutorial"
FAILED_EXAMPLES=()
PASSED_EXAMPLES=()

echo "========================================"
echo "Running Tutorial Examples"
echo "========================================"

# Activate environment
# Note: Temporarily disable unbound variable check because activate script checks $1
if [ -f "build/env/activate" ]; then
    set +u
    source build/env/activate
    set -u
    echo "✓ Activated virtual environment"
else
    echo "Error: build/env/activate not found. Please build the project first."
    exit 1
fi

# Verify ttnn is available
if ! python3 -c "import ttnn" 2>/dev/null; then
    echo "Error: ttnn not available. These tests require ttnn and hardware access."
    exit 1
fi

# Find all Python files in tutorial directory
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "Error: Tutorial directory '$EXAMPLES_DIR' not found"
    exit 1
fi

# Exclude ttnn_base.py (reference example showing vanilla ttnn, not ttl)
TUTORIAL_FILES=$(find "$EXAMPLES_DIR" -maxdepth 1 -name "*.py" -type f ! -name "ttnn_base.py" | sort)

if [ -z "$TUTORIAL_FILES" ]; then
    echo "Error: No tutorial examples found in $EXAMPLES_DIR"
    exit 1
fi

echo "Found $(echo "$TUTORIAL_FILES" | wc -l) tutorial examples"
echo ""

# Run each example
for example in $TUTORIAL_FILES; do
    example_name=$(basename "$example")
    echo "----------------------------------------"
    echo "Running: $example_name"
    echo "----------------------------------------"

    if python3 "$example" 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}: $example_name"
        PASSED_EXAMPLES+=("$example_name")
    else
        echo -e "${RED}✗ FAILED${NC}: $example_name"
        FAILED_EXAMPLES+=("$example_name")
    fi
    echo ""
done

# Print summary
echo "========================================"
echo "Summary"
echo "========================================"
echo -e "${GREEN}Passed${NC}: ${#PASSED_EXAMPLES[@]}"
echo -e "${RED}Failed${NC}: ${#FAILED_EXAMPLES[@]}"
echo ""

if [ ${#PASSED_EXAMPLES[@]} -gt 0 ]; then
    echo "Passed examples:"
    for example in "${PASSED_EXAMPLES[@]}"; do
        echo -e "  ${GREEN}✓${NC} $example"
    done
    echo ""
fi

if [ ${#FAILED_EXAMPLES[@]} -gt 0 ]; then
    echo "Failed examples:"
    for example in "${FAILED_EXAMPLES[@]}"; do
        echo -e "  ${RED}✗${NC} $example"
    done
    echo ""
    exit 1
fi

echo -e "${GREEN}All tutorial examples passed!${NC}"
exit 0
