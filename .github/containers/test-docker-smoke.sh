#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Quick smoke test for tt-lang Docker containers
# Tests basic functionality and hardware access

set -e

echo "=== tt-lang Docker Smoke Test ==="
echo ""

# Test 1: Basic import test
echo "Test 1: Basic imports (no hardware)"
sudo docker run --rm tt-lang-dist:local python -c "
import pykernel
import ttlang
from ttmlir.dialects import ttkernel
print('✓ All imports successful')
" && echo "✓ PASS: Imports work" || echo "✗ FAIL: Import error"
echo ""

# Test 2: Example without hardware
echo "Test 2: Run example (no hardware, will fail at runtime but imports should work)"
sudo docker run --rm tt-lang-dist:local python -c "
from ttlang import ttl
print('✓ ttlang.ttl imported successfully')
" && echo "✓ PASS: ttl module works" || echo "✗ FAIL: ttl import failed"
echo ""

# Test 3: With hardware (if available)
if [ -e /dev/tenstorrent/0 ]; then
    echo "Test 3: Run example with hardware (card 0)"
    sudo docker run --rm \
      --device=/dev/tenstorrent/0 \
      -v /dev/hugepages:/dev/hugepages \
      -v /dev/hugepages-1G:/dev/hugepages-1G \
      tt-lang-dist:local python /opt/ttmlir-toolchain/examples/demo_one.py \
      && echo "✓ PASS: Example ran successfully" || echo "✗ FAIL: Example failed"
else
    echo "Test 3: SKIPPED (no Tenstorrent hardware detected)"
fi
echo ""

# Test 4: Check tools are available
echo "Test 4: Check vim/nano are available"
sudo docker run --rm tt-lang-dist:local bash -c "which vim && which nano" > /dev/null \
  && echo "✓ PASS: Editors available" || echo "✗ FAIL: Editors missing"
echo ""

# Test 5: Check examples are in /root
echo "Test 5: Check examples in /root"
sudo docker run --rm tt-lang-dist:local bash -c "ls /root/examples/demo_one.py" > /dev/null \
  && echo "✓ PASS: Examples copied to /root" || echo "✗ FAIL: Examples missing"
echo ""

echo "=== Smoke Test Complete ==="
