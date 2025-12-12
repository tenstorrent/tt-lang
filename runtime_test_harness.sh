#!/bin/bash

for test in $(find test/python -name "*runtime*.py" -type f | sort); do echo "=== $test ==="; if /Users/zcarver/Developer/tt-lang-hw-sim/run-test.sh "$test" 2>&1 | grep -q "PASS"; then echo "✓ PASSED"; else echo "✗ FAILED"; fi; echo ""; done
