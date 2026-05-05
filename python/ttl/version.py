# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt-lang version string, read from CMake-generated config module."""

try:
    from ttl.config import VERSION as _VERSION  # type: ignore[reportMissingTypeStubs]
except ImportError:
    _VERSION = "0.0.0"

__version__: str = str(_VERSION)
