# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed values for external C++ template arguments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UInt32TemplateArgument:
    """An unsigned 32-bit value retained through operation composition."""

    value: int


def uint32(value: int) -> UInt32TemplateArgument:
    """Represent an unsigned 32-bit C++ template argument."""
    if type(value) is not int:
        raise TypeError("ttl.uint32() requires an int")
    if not 0 <= value < (1 << 32):
        raise ValueError("ttl.uint32() value must fit in 32 bits")
    return UInt32TemplateArgument(value)


__all__ = ("uint32",)
