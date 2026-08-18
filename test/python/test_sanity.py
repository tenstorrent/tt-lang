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
    assert hasattr(ttl, "operation")
    assert hasattr(ttl, "compute")
    assert hasattr(ttl, "datamovement")


def test_ttl_program_entry_point_is_gone():
    """``Program`` is not part of the API the specification describes.

    Specification 0.2 replaced it with ``@ttl.operation``, and the simulator
    never exposed it.  Keeping it exported would let a program compile against a
    name the simulator cannot run.
    """
    import ttl

    assert not hasattr(ttl, "Program")


def test_ttl_version():
    """Verify ttl version is available and valid."""
    import ttl

    assert hasattr(ttl, "__version__")
    assert isinstance(ttl.__version__, str)
    assert ttl.__version__ != ""
    # Should be a real version, not an unsubstituted CMake variable
    assert not ttl.__version__.startswith("@")


def test_python_environment():
    """Verify Python environment is set up correctly."""
    import sys

    assert sys.version_info >= (3, 10), "Python 3.10+ required"
