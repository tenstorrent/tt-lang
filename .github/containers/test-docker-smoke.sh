#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Quick smoke test for tt-lang Docker containers

# Track test results
PASSED=0
FAILED=0

# Helper: run_test "pass message" "fail message" command args...
run_test() {
    local pass_msg="$1"
    local fail_msg="$2"
    shift 2

    if "$@" > /dev/null 2>&1; then
        echo "✓ PASS: $pass_msg"
        ((PASSED++))
    else
        echo "✗ FAIL: $fail_msg"
        ((FAILED++))
    fi
}

# Accept an optional tag argument; default to the nearest version tag.
DOCKER_TAG="${1:-$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null)}"
if [ -z "$DOCKER_TAG" ]; then
    echo "ERROR: Could not determine Docker tag from git tags and no tag argument provided."
    exit 1
fi
DIST_IMAGE="tt-lang-dist-ubuntu-22-04:$DOCKER_TAG"

echo "=== tt-lang Docker Smoke Test ==="
echo "Image: $DIST_IMAGE"
echo ""

# Test 1: Basic imports
echo "Test 1: Basic imports"
run_test "All imports work" "Import error" \
    sudo docker run --rm "$DIST_IMAGE" python -c "
import pykernel; import sim; import ttl
from ttmlir.dialects import ttkernel
"
echo ""

# Test 2: ttl module
echo "Test 2: ttl module"
run_test "ttl.ttl works" "ttl.ttl import failed" \
    sudo docker run --rm $DIST_IMAGE python -c "import ttl"
echo ""

# Test 3: Hardware example (if available)
if [ -e /dev/tenstorrent/0 ]; then
    echo "Test 3: Hardware example"
    run_test "Example ran on hardware" "Hardware example failed" \
        sudo docker run --rm \
            --device=/dev/tenstorrent/0 \
            -v /dev/hugepages:/dev/hugepages \
            -v /dev/hugepages-1G:/dev/hugepages-1G \
            $DIST_IMAGE python /opt/ttlang-toolchain/examples/elementwise-tutorial/step_4_multinode_grid_auto.py
else
    echo "Test 3: SKIPPED (no hardware)"
fi
echo ""

# Test 4: Editors
echo "Test 4: Editors available"
run_test "vim/nano available" "Editors missing" \
    sudo docker run --rm $DIST_IMAGE bash -c "which vim && which nano"
echo ""

# Test 5: Examples in /root
echo "Test 5: Examples in /root"
run_test "Examples in /root" "Examples missing" \
    sudo docker run --rm $DIST_IMAGE ls /root/examples/elementwise-tutorial/step_4_multinode_grid_auto.py
echo ""

echo "=== Smoke Test Complete ==="
echo ""
echo "Summary: $PASSED passed, $FAILED failed"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "❌ Some tests failed"
    exit 1
else
    echo "✅ All tests passed!"
    exit 0
fi
