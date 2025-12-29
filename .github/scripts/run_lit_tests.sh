#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

HARDWARE_TYPE="${1:-n150}"

source build/env/activate
echo "Activated virtual environment"

# Set SYSTEM_DESC_PATH to the generated system descriptor
if [ -f "ttrt-artifacts/system_desc.ttsys" ]; then
  export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"
  echo "Using system descriptor: ${SYSTEM_DESC_PATH}"
else
  echo "Warning: system_desc.ttsys not found in ttrt-artifacts/"
fi

mkdir -p test_reports

echo "Running hardware tests on ${HARDWARE_TYPE}..."

llvm-lit -sv test/python/ \
  --xunit-xml-output test_reports/report_${HARDWARE_TYPE}.xml

if [ -f test_reports/report_${HARDWARE_TYPE}.xml ]; then
  echo "Test report generated successfully"
  grep -E "testsuite|testcase" test_reports/report_${HARDWARE_TYPE}.xml | head -20 || true
fi
