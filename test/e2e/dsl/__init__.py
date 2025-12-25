# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DSL test base for E2E tests.

Provides infrastructure for testing Python DSL kernels using @ttl.kernel decorator.
"""

from typing import Callable

import pytest

from ..base import E2ETestBase


class DSLTestBase(E2ETestBase):
    """
    Base for Python DSL tests.

    Subclasses define DSL_FUNC to specify the @ttl.kernel decorated function.
    """

    DSL_FUNC: Callable  # The @ttl.kernel decorated function

    @pytest.mark.order(1)
    def test_build_module(self) -> None:
        """
        Trace DSL function to MLIR.

        Note: Actual tracing implementation depends on d2m_api integration.
        """
        raise NotImplementedError(
            "DSL tracing not yet implemented - requires d2m_api integration"
        )
