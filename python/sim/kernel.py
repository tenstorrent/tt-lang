# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Inert logical-kernel selectors for simulator API compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


class KernelKind(Enum):
    """A target-independent kernel class."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        del start, count, last_values
        return name.lower()

    COMPUTE = auto()
    DATA_MOVEMENT = auto()

    def __or__(self, other: object) -> Tuple[KernelKind, KernelKind]:
        if not isinstance(other, KernelKind):
            return NotImplemented
        return (self, other)


@dataclass(frozen=True)
class Kernel:
    """A logical-kernel declaration ignored by simulator execution."""

    kind: KernelKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, KernelKind):
            raise TypeError(
                "Kernel kind must be a KernelKind, got " f"{type(self.kind).__name__}"
            )


KernelSelector = KernelKind | Kernel
