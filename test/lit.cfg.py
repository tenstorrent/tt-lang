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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Get Python packages directory from site config, or fall back to default build location.
build_python = getattr(config, "TTLANG_PYTHON_PACKAGES_DIR", None)
if build_python is None:
    # Fallback to default build location if not set by site config.
    build_python = os.path.join(project_root, "build", "python_packages")

python_paths = [
    build_python,
    os.path.join(project_root, "python"),
    os.environ.get("PYTHONPATH", ""),
]

# Prefer built bindings so ttlang._mlir_libs is found.
config.environment["PYTHONPATH"] = os.path.pathsep.join([p for p in python_paths if p])

if "SYSTEM_DESC_PATH" in os.environ:
    config.environment["SYSTEM_DESC_PATH"] = os.environ["SYSTEM_DESC_PATH"]

# Metal runtime requires both of these
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]
if "TT_METAL_RUNTIME_ROOT" in os.environ:
    config.environment["TT_METAL_RUNTIME_ROOT"] = os.environ["TT_METAL_RUNTIME_ROOT"]

# Add system platform feature for UNSUPPORTED directives
if platform.system() == "Darwin":
    config.available_features.add("system-darwin")
