# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for preparing README content for PyPI long_description.

PyPI renders Markdown without access to the repository file tree, so relative
image references (`docs/...`, `./assets/...`) never resolve. This module
rewrites such references to absolute raw.githubusercontent.com URLs anchored
at a specific git ref so the rendered project page shows the correct images
for that release.
"""

import pathlib
import re

_REPO_RAW_BASE = "https://raw.githubusercontent.com/tenstorrent/tt-lang"

_HTML_IMG_SRC_RE = re.compile(
    r'(<img\b[^>]*\bsrc=")(?!https?://|/|data:)([^"]+)"',
    flags=re.IGNORECASE,
)

_MD_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()(?!https?://|/|data:)([^)\s]+)")

# Release tags in this repo are strictly vMAJOR.MINOR.PATCH; any other shape
# (dev/pre/post/local segments, TTLANG_PRETEND_VERSION garbage, etc.) has no
# corresponding git ref so it must fall back to a branch.
_RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def ref_for_version(version: str) -> str:
    """Pick the git ref that holds the assets for this wheel.

    Returns `vX.Y.Z` only when `version` is exactly three numeric segments;
    anything else (dev wheels, pre/post-release identifiers, malformed
    `TTLANG_PRETEND_VERSION` overrides, empty strings) falls back to `main`.
    """
    if _RELEASE_VERSION_RE.match(version):
        return f"v{version}"
    return "main"


def _collect_relative_image_paths(text: str) -> list[str]:
    paths: list[str] = []
    paths.extend(m.group(2) for m in _HTML_IMG_SRC_RE.finditer(text))
    paths.extend(m.group(2) for m in _MD_IMAGE_RE.finditer(text))
    return paths


def absolutize_readme_images(text: str, ref: str, repo_root: pathlib.Path) -> str:
    """Rewrite relative image URLs in `text` to absolute GitHub raw URLs.

    Every relative path is verified against `repo_root`; missing files raise
    `FileNotFoundError` so a broken README aborts the wheel build instead of
    shipping a 404 to PyPI.
    """
    missing = [
        path
        for path in _collect_relative_image_paths(text)
        if not (repo_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"README image paths do not resolve under {repo_root}: {missing}"
        )

    base = f"{_REPO_RAW_BASE}/{ref}/"
    text = _HTML_IMG_SRC_RE.sub(lambda m: f'{m.group(1)}{base}{m.group(2)}"', text)
    text = _MD_IMAGE_RE.sub(lambda m: f"{m.group(1)}{base}{m.group(2)}", text)
    return text
