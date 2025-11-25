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

# Add feature for system descriptor availability
# Tests that require hardware execution can use: REQUIRES: system-desc
if "SYSTEM_DESC_PATH" in os.environ and os.path.exists(os.environ["SYSTEM_DESC_PATH"]):
    config.environment["SYSTEM_DESC_PATH"] = os.environ["SYSTEM_DESC_PATH"]
    config.available_features.add("system-desc")

# Metal runtime requires both of these
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]
if "TT_METAL_RUNTIME_ROOT" in os.environ:
    config.environment["TT_METAL_RUNTIME_ROOT"] = os.environ["TT_METAL_RUNTIME_ROOT"]

# Add system platform feature for UNSUPPORTED directives
if platform.system() == "Darwin":
    config.available_features.add("system-darwin")
