# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR file test base for E2E tests.

Provides infrastructure for testing manually written MLIR files.
"""

from pathlib import Path

from ttmlir.ir import Module

from ..base import E2ETestBase


class MLIRFileTestBase(E2ETestBase):
    """
    Base for tests using manually written MLIR files.

    Subclasses define MLIR_PATH to specify the input file.
    Override test_validate_golden() for custom validation logic.
    """

    MLIR_PATH: Path  # Override in subclass

    def test_build_module(self):
        """Load MLIR module from file."""
        with open(self.MLIR_PATH) as f:
            mlir_str = f.read()
        self.CACHE["module"] = Module.parse(mlir_str)
