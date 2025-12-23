# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Semaphore operations for multi-core synchronization."""

from typing import Optional, Tuple

from ttmlir.dialects import d2m

from ._src.ttl_ast import syntax
from pykernel._src.utils import _asindex


@syntax("!d2m.semaphore")
class Semaphore:
    """
    Semaphore for multi-core synchronization.

    Semaphores enable coordination between cores through set, increment,
    and wait operations with optional multicast.
    """

    def set(
        ast_self: "Semaphore",
        value: int,
        core: Optional[Tuple[int, int]] = None,
        mcast: Optional[Tuple[int, int]] = None,
    ):
        """
        Set semaphore value, optionally multicasting to other cores.

        Args:
            value: Value to set
            core: Target core coordinates for multicast
            mcast: Multicast dimensions
        """
        return d2m.semaphore_set(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def inc(
        ast_self: "Semaphore",
        value: int,
        core: Optional[Tuple[int, int]] = None,
    ):
        """
        Increment semaphore value on a remote core.

        Args:
            value: Increment amount
            core: Target core coordinates
        """
        return d2m.semaphore_inc(
            ast_self, _asindex(value), _asindex(core), _asindex(None)
        )

    def wait(
        ast_self: "Semaphore",
        value: int,
        reset: Optional[int] = None,
    ):
        """
        Wait for semaphore to reach a value, optionally resetting after.

        Args:
            value: Value to wait for
            reset: Optional value to reset semaphore to after waiting
        """
        return d2m.semaphore_wait(
            ast_self, _asindex(value), reset_value=_asindex(reset)
        )
