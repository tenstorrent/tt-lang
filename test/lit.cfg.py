# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import platform

import lit.formats
import lit.util
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

config.name = "TTLang"

# Use lit internal shell for better error reporting
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
# Keep .py for lit, pytest tests are excluded via excludes below.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
_default_obj_root = os.environ.get("TTLANG_OBJ_ROOT")
if not _default_obj_root:
    _default_obj_root = os.path.abspath(
        os.path.join(config.test_source_root, os.pardir, "build")
    )

config.ttlang_obj_root = getattr(config, "ttlang_obj_root", _default_obj_root)
config.ttlang_source_dir = getattr(
    config,
    "ttlang_source_dir",
    os.environ.get(
        "TTLANG_SOURCE_DIR",
        os.path.abspath(os.path.join(config.test_source_root, os.pardir)),
    ),
)

config.test_exec_root = os.path.join(config.ttlang_obj_root, "test")

config.excludes = [
    "Inputs",
    "lit.cfg.py",
    "sim",
]

# Exclude pytest-style tests (test_*.py) from lit collection.
import os

for _root, _dirs, _files in os.walk(config.test_source_root):
    for _f in _files:
        if _f.startswith("test_") and _f.endswith(".py"):
            config.excludes.append(_f)

if llvm_config is not None:
    llvm_config.with_system_environment(
        ["HOME", "INCLUDE", "LIB", "PYTHONPATH", "TMP", "TEMP"]
    )

# Use default substitutions from LLVM (includes FileCheck, not, etc.)
if llvm_config is not None:
    llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
tool_dirs = [
    os.path.join(config.ttlang_obj_root, "bin"),
    config.llvm_tools_dir,
    config.lit_tools_dir,
]
for dirs in tool_dirs:
    llvm_config.with_environment("PATH", dirs, append_path=True)

# Add ttlang-opt tool
tools = ["ttlang-opt"]

if llvm_config is not None:
    llvm_config.add_tool_substitutions(tools, tool_dirs)

# Python test configuration
config.substitutions.append(
    (
        "%python",
        f"env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir {sys.executable}",
    )
)

# Get Python packages directory from site config, or fall back to default build location.
build_python = getattr(config, "TTLANG_PYTHON_PACKAGES_DIR", None)
if build_python is None or not build_python:
    # Fallback to default build location if not set by site config.
    build_python = os.path.join(config.ttlang_obj_root, "python_packages")

python_paths = [
    build_python,
    os.path.join(config.ttlang_source_dir, "python"),
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
