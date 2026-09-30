# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s
#
# Verify the build package stages unchanged DFB helper headers. Installed wheels
# are checked separately by .github/scripts/smoke-test-wheel.py.

import importlib.util
from pathlib import Path

from ttl import config


package_spec = importlib.util.find_spec("ttl")
assert package_spec is not None and package_spec.submodule_search_locations
package_dir = Path(next(iter(package_spec.submodule_search_locations)))
source_dir = Path(__file__).resolve().parents[3]
assert package_dir.is_relative_to(config.BUILD_DIR)
include_dir = config.KERNEL_INCLUDE_DIR.relative_to(package_dir)

for name in ("experimental_dfb_reset.h", "experimental_dfb_reconfiguration.h"):
    header = include_dir / config.KERNEL_HEADER_DIR / name
    packaged_header = package_dir / header
    assert packaged_header.is_file(), f"Missing packaged header: {packaged_header}"
    assert packaged_header.read_bytes() == (source_dir / header).read_bytes(), name
