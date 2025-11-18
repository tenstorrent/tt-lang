# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re
import platform
import tempfile

import lit.util
import lit.formats

config.name = "TT-Lang"
config.test_format = lit.formats.ShTest(True)

config.suffixes = [".py"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, "temp")

config.substitutions.append(
    (
        "%python",
        f"env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir {sys.executable}",
    )
)

config.environment["PYTHONPATH"] = os.path.pathsep.join(
    [
        os.path.join(os.path.dirname(__file__), "..", "python"),
        os.environ.get("PYTHONPATH", ""),
    ]
)

# Use default system descriptor for compile-only tests
# This can be overridden by setting SYSTEM_DESC_PATH environment variable
if "SYSTEM_DESC_PATH" not in os.environ:
    default_system_desc = os.path.join(
        os.path.dirname(__file__), "..", "ttrt-artifacts", "system_desc.ttsys"
    )
    if os.path.exists(default_system_desc):
        config.environment["SYSTEM_DESC_PATH"] = default_system_desc
elif "SYSTEM_DESC_PATH" in os.environ:
    config.environment["SYSTEM_DESC_PATH"] = os.environ["SYSTEM_DESC_PATH"]

# Metal runtime requires both of these
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]
if "TT_METAL_RUNTIME_ROOT" in os.environ:
    config.environment["TT_METAL_RUNTIME_ROOT"] = os.environ["TT_METAL_RUNTIME_ROOT"]

# Add has-tt-hw feature - unconditionally false until HW CI is ready
# Tests that require actual TT hardware should use REQUIRES: has-tt-hw
# config.available_features.add("has-tt-hw")  # Uncomment when HW is available
