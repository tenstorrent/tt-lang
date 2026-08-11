# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Scalar types supported by typed TTL APIs."""

from enum import Enum


class ScalarType(Enum):
    """A signless scalar integer type at the external-call boundary."""

    I32 = 32
    I64 = 64

    @property
    def bit_width(self) -> int:
        return self.value


__all__ = ["ScalarType"]
