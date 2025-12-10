# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import platform

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

config.name = "TTLang"

# Use lit internal shell for better error reporting
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ttlang_obj_root, "test")

config.excludes = [
    "Inputs",
    "lit.cfg.py",
    "sim",
]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# Use default substitutions from LLVM (includes FileCheck, not, etc.)
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

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Python test configuration
config.substitutions.append(
    (
        "%python",
        f"env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_FINAL_MLIR=%t.final.mlir {sys.executable}",
    )
)

# Set up PYTHONPATH for Python tests
python_dir = os.path.join(config.ttlang_source_dir, "python")
config.environment["PYTHONPATH"] = os.path.pathsep.join(
    [python_dir, os.environ.get("PYTHONPATH", "")]
)

if "SYSTEM_DESC_PATH" in os.environ:
    config.environment["SYSTEM_DESC_PATH"] = os.environ["SYSTEM_DESC_PATH"]

# Enable FileCheck variable scoping (MLIR default)
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"

# Add system platform feature for UNSUPPORTED directives
if platform.system() == "Darwin":
    config.available_features.add("system-darwin")
