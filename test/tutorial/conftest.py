# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the tutorial example suite.

This suite lives in its own directory, a sibling of test/python and test/me2e,
so those end-to-end suites do not autocollect these device-heavy example
scripts. Worker-to-chip pinning matches them so the tutorials participate in the
run-hardware-pytests.sh parallel (per-chip) and serial (multi_device) phases.
"""

import os
import sys

# Shared test utilities live at the test/ root, one level up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttlang_test_utils import pin_xdist_worker_to_device

pin_xdist_worker_to_device()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "multi_device: opens a device mesh; excluded from the per-chip parallel "
        "run and executed serially",
    )
