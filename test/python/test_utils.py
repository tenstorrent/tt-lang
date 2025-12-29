# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Common test utilities for tt-lang Python tests.

Provides consistent TTNN import handling and device availability checking.
When no hardware is available, TTLANG_COMPILE_ONLY is set automatically.

Device availability is determined at CMake configure time by checking for
/dev/tenstorrent* files, avoiding the slow ttnn.GetNumAvailableDevices() call.
"""

import os
import sys

# Check device availability from CMake-generated config (fast path)
# Falls back to checking environment if config not available
_hardware_available = False

try:
    # Try to import CMake-generated config first (fast - no ttnn import needed)
    from test_config import HAS_TT_DEVICE

    _hardware_available = HAS_TT_DEVICE
except ImportError:
    # Config not available (running outside build dir) - check env or device files
    import glob

    if os.environ.get("TTLANG_HAS_DEVICE") == "1":
        _hardware_available = True
    elif glob.glob("/dev/tenstorrent*"):
        _hardware_available = True

# Set compile-only mode if no hardware
if not _hardware_available:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

# Try to import TTNN
ttnn = None
_ttnn_available = False

try:
    import ttnn as _ttnn

    ttnn = _ttnn
    _ttnn_available = True
except ImportError:
    pass


def require_ttnn():
    """Exit test if TTNN is not available."""
    if not _ttnn_available:
        print("TTNN not available - exiting")
        sys.exit(0)


def is_hardware_available():
    """Check if Tenstorrent hardware is available for running kernels."""
    return _hardware_available


def is_ttnn_available():
    """Check if TTNN library is available."""
    return _ttnn_available


def skip_without_hardware(message: str = "Skipping - no hardware available"):
    """Print message and exit if no hardware is available.

    Use this at the start of `if __name__ == "__main__":` blocks in tests
    that need to run kernels on actual hardware.
    """
    if not _hardware_available:
        print(message)
        sys.exit(0)
