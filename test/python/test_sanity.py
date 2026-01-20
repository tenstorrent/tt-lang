# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests that run on all platforms without hardware dependencies."""

import pytest


def test_ttl_import():
    """Verify ttl module can be imported."""
    import ttl


def test_ttl_api_basic():
    """Verify basic ttl API is available."""
    import ttl

    # Basic API should be available even without TTNN
    assert hasattr(ttl, "kernel")
    assert hasattr(ttl, "compute")
    assert hasattr(ttl, "datamovement")
    assert hasattr(ttl, "Program")


def test_python_environment():
    """Verify Python environment is set up correctly."""
    import sys

    assert sys.version_info >= (3, 10), "Python 3.10+ required"
