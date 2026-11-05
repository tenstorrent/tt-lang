# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for the example script suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "multi_device: needs a device mesh; skipped on a single device"
    )
