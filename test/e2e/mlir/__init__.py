# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MLIR file test base for E2E tests.

Provides infrastructure for testing manually written MLIR files.
"""

from pathlib import Path

import pytest
from ttmlir.ir import Context, Module

from ..base import E2ETestBase


class MLIRFileTestBase(E2ETestBase):
    """
    Base for tests using manually written MLIR files.

    Subclasses define MLIR_PATH to specify the input file.
    """

    MLIR_PATH: Path  # Override in subclass

    @pytest.mark.order(1)
    def test_build_module(self) -> None:
        """Load MLIR module from file."""
        with Context() as ctx:
            with open(self.MLIR_PATH) as f:
                mlir_str = f.read()
            module = Module.parse(mlir_str, ctx)

            # Save to output directory for subsequent stages.
            module_file = self.output_file("module.mlir")
            with open(module_file, "w") as f:
                f.write(str(module))
