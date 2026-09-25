# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve compiler-owned kernel headers for source, build, and wheel installs."""

from pathlib import Path
from typing import Iterable


def kernel_include_paths(include_paths: Iterable[str]) -> list[str]:
    """Append available TT-Lang headers after caller-supplied override paths."""
    paths = list(include_paths)
    module_path = Path(__file__).absolute()
    # Build-tree Python modules may be symlinks into the source tree.
    roots = (
        module_path.parent / "include",
        module_path.resolve().parents[2] / "include",
    )
    for root in roots:
        if (root / "ttlang/Target/TTKernel/LLKs").is_dir():
            paths.append(str(root))
            break
    return paths
