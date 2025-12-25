# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DSL test base for E2E tests.

Provides infrastructure for testing Python DSL kernels (migrated from test/python/).
"""

from typing import Callable

from ..base import E2ETestBase


class DSLTestBase(E2ETestBase):
    """
    Base for Python DSL tests.

    Subclasses define DSL_FUNC to specify the @kernel decorated function.
    Override test_validate_golden() for custom validation logic.
    """

    DSL_FUNC: Callable  # The @kernel decorated function

    def test_build_module(self):
        """
        Trace DSL function to MLIR.

        Note: Actual tracing implementation depends on d2m_api integration.
        For now, this is a placeholder that will be implemented when
        migrating specific DSL tests.
        """
        raise NotImplementedError(
            "DSL tracing not yet implemented - requires d2m_api integration"
        )
