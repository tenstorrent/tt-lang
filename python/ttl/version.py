# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt-lang version string, read from CMake-generated config module."""

try:
    from ttl.config import VERSION as _VERSION  # type: ignore[reportMissingTypeStubs]
except ImportError:
    _VERSION = "0.0.0"

__version__: str = str(_VERSION)


def build_info() -> dict:
    """Source revisions this wheel was built from, for integration debugging.

    Keys: ``version``, ``ttlang`` (tt-lang commit), ``tt_metal_tag``,
    ``tt_metal`` (tt-metal commit). Values are ``"unknown"`` when the wheel was
    built without a git checkout and without the CI build-arg overrides.
    """
    try:
        from ttl import config as _config
    except ImportError:
        _config = None
    return {
        "version": __version__,
        "ttlang": str(getattr(_config, "TTLANG_COMMIT", "unknown")),
        "tt_metal_tag": str(getattr(_config, "TT_METAL_TAG", "unknown")),
        "tt_metal": str(getattr(_config, "TT_METAL_COMMIT", "unknown")),
    }
