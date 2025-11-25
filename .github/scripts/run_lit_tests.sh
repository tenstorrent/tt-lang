#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Source the environment
source build/env/activate

# Create test output directory
mkdir -p test_reports

# Run lit tests on hardware
echo "Running hardware tests on ${RUNS_ON}..."

# Run tests using cmake target
cmake --build build --target check-ttlang

# Display test results summary (assuming XML is generated)
if [ -f test_reports/report_${RUNS_ON}.xml ]; then
  echo "Test report generated successfully"
  grep -E "testsuite|testcase" test_reports/report_${RUNS_ON}.xml | head -20 || true
fi
