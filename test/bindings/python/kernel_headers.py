# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# RUN: %python %s
#
# Verify the Python package ships the unchanged DFB helper headers.

import importlib.util
from pathlib import Path


package_spec = importlib.util.find_spec("ttl")
assert package_spec is not None and package_spec.submodule_search_locations
package_dir = Path(next(iter(package_spec.submodule_search_locations)))
source_dir = Path(__file__).resolve().parents[3]

for name in ("experimental_dfb_reset.h", "experimental_dfb_reconfiguration.h"):
    header = Path("include/ttlang/Target/TTKernel/LLKs") / name
    packaged_header = package_dir / header
    assert packaged_header.is_file(), f"Missing packaged header: {packaged_header}"
    assert packaged_header.read_bytes() == (source_dir / header).read_bytes(), name
