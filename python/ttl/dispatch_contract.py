# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Logical argument contracts for independently reloaded operations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class DispatchAccess(str, Enum):
    """How one dispatch target accesses an argument."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class DispatchStorage(str, Enum):
    """Lifetime required of an argument across reload image boundaries."""

    ORDINARY = "ordinary"
    HANDOFF = "handoff"
    PERSISTENT_STATE = "persistent_state"
    IMMUTABLE_IMAGE_STATE = "immutable_image_state"


@dataclass(frozen=True)
class DispatchArgument:
    """Access and cross-image lifetime of one target argument."""

    access: DispatchAccess = DispatchAccess.READ_WRITE
    storage: DispatchStorage = DispatchStorage.ORDINARY
    state: str | None = None

    def __post_init__(self) -> None:
        try:
            access = DispatchAccess(self.access)
        except (TypeError, ValueError):
            raise TypeError(
                "dispatch argument access must be 'read', 'write', or " "'read_write'"
            ) from None
        try:
            storage = DispatchStorage(self.storage)
        except (TypeError, ValueError):
            raise TypeError(
                "dispatch argument storage must be 'ordinary', 'handoff', "
                "'persistent_state', or 'immutable_image_state'"
            ) from None
        object.__setattr__(self, "access", access)
        object.__setattr__(self, "storage", storage)

        if storage is DispatchStorage.PERSISTENT_STATE:
            if not isinstance(self.state, str) or not self.state:
                raise ValueError(
                    "persistent_state dispatch arguments require a non-empty state name"
                )
        elif self.state is not None:
            raise ValueError(
                "only persistent_state dispatch arguments may declare a state name"
            )
        if (
            storage is DispatchStorage.IMMUTABLE_IMAGE_STATE
            and access is not DispatchAccess.READ
        ):
            raise ValueError(
                "immutable_image_state dispatch arguments must be read-only"
            )


_DEFAULT_DISPATCH_ARGUMENT = DispatchArgument()


def _normalize_dispatch_arguments(fn, declarations) -> tuple[DispatchArgument, ...]:
    parameters = tuple(inspect.signature(fn).parameters)
    if declarations is None:
        declarations = {}
    if not isinstance(declarations, Mapping):
        raise TypeError("ttl.operation() dispatch_arguments must be a mapping")

    unknown = sorted(set(declarations).difference(parameters))
    if unknown:
        raise ValueError(
            f"@ttl.operation {fn.__name__!r}: dispatch_arguments names absent "
            f"parameter(s) {unknown}"
        )
    contracts = []
    for name in parameters:
        contract = declarations.get(name, _DEFAULT_DISPATCH_ARGUMENT)
        if not isinstance(contract, DispatchArgument):
            raise TypeError(
                f"@ttl.operation {fn.__name__!r}: dispatch argument {name!r} "
                "must be a ttl.DispatchArgument"
            )
        contracts.append(contract)
    return tuple(contracts)


__all__ = ["DispatchAccess", "DispatchArgument", "DispatchStorage"]
