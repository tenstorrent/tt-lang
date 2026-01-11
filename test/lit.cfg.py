# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys

import lit.formats
import lit.util
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

config.name = "TTLang"

# Use lit internal shell for better error reporting and built-in commands (like `not`)
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

# Create Output directories for lit temp files (%t substitution).
# This is needed when running from pre-built artifacts where the build
# directory may not have the Output subdirectories created.
for subdir in ["python", "ttlang", "bindings/python"]:
    output_dir = os.path.join(config.test_exec_root, subdir, "Output")
    os.makedirs(output_dir, exist_ok=True)

config.excludes = [
    "Inputs",
    "lit.cfg.py",
    "sim",
    "conftest.py",
    "utils.py",
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
]
# Add tt-mlir tools directory if available
if hasattr(config, "ttmlir_path") and config.ttmlir_path:
    ttmlir_bin = os.path.join(config.ttmlir_path, "bin")
    if os.path.exists(ttmlir_bin):
        tool_dirs.append(ttmlir_bin)
if hasattr(config, "llvm_tools_dir"):
    tool_dirs.append(config.llvm_tools_dir)
if hasattr(config, "lit_tools_dir"):
    tool_dirs.append(config.lit_tools_dir)

if llvm_config is not None:
    for dirs in tool_dirs:
        llvm_config.with_environment("PATH", dirs, append_path=True)

# Add ttlang-opt, and ttlang-translate tools
tools = ["ttlang-opt", "ttlang-translate"]

if llvm_config is not None:
    llvm_config.add_tool_substitutions(tools, tool_dirs)

# Python test configuration
# Note: We cannot use %t directly in env var values because lit doesn't expand
# substitutions recursively. Instead, tests should use %t explicitly:
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir %python %s
config.substitutions.append(("%python", sys.executable))

# Get Python packages directory from site config, or fall back to default build location.
build_python = getattr(config, "TTLANG_PYTHON_PACKAGES_DIR", None)
if build_python is None or not build_python:
    # Fallback to default build location if not set by site config.
    build_python = os.path.join(config.ttlang_obj_root, "python_packages")

python_paths = [
    build_python,
    os.path.join(config.ttlang_source_dir, "python"),
]

# Add tt-mlir Python packages if available
if hasattr(config, "ttmlir_path") and config.ttmlir_path:
    ttmlir_python = os.path.join(config.ttmlir_path, "python_packages")
    if os.path.exists(ttmlir_python):
        python_paths.append(ttmlir_python)

# Include existing PYTHONPATH last
python_paths.append(os.environ.get("PYTHONPATH", ""))

# Prefer built bindings so ttlang._mlir_libs is found.
config.environment["PYTHONPATH"] = os.path.pathsep.join([p for p in python_paths if p])

# Enable FileCheck variable scoping (MLIR default)
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"

# Pass through TT Metal/MLIR environment variables if set
for env_var in [
    "HOME",
    "TT_METAL_SIMULATOR",
    "TT_METAL_SLOW_DISPATCH_MODE",
    "TT_METAL_HOME",
    "TT_METAL_BUILD_HOME",
    "TT_METAL_RUNTIME_ROOT",
    "TT_MLIR_HOME",
    "TTLANG_COMPILE_ONLY",
]:
    if env_var in os.environ:
        config.environment[env_var] = os.environ[env_var]

# Add system platform feature for UNSUPPORTED directives
if platform.system() == "Darwin":
    config.available_features.add("system-darwin")

# Add TTNN feature if available
try:
    import ttnn

    config.available_features.add("ttnn")
except ImportError:
    pass

# Add tt-device feature if hardware is available (detected by CMake at configure time)
if getattr(config, "ttlang_has_device", False):
    config.available_features.add("tt-device")
