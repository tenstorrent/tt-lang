#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

HARDWARE_TYPE="${1:-n150}"

source build/env/activate
echo "Activated virtual environment"
env

mkdir -p test_reports

echo "Running hardware tests on ${HARDWARE_TYPE}..."

llvm-lit -sv test/python/ \
  --xunit-xml-output test_reports/report_${HARDWARE_TYPE}.xml

if [ -f test_reports/report_${HARDWARE_TYPE}.xml ]; then
  echo "Test report generated successfully"
  grep -E "testsuite|testcase" test_reports/report_${HARDWARE_TYPE}.xml | head -20 || true
fi
