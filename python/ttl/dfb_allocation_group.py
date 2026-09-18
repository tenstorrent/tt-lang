# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed physical-allocation identities for dataflow buffers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping


@dataclass(frozen=True, eq=False)
class DFBAllocationGroup:
    """Immutable compile-time identity for verified DFB physical aliasing."""


@dataclass(frozen=True)
class _BoundDFBAllocationGroup:
    declaration: DFBAllocationGroup
    ordinal: int


class _DFBAllocationGroupBinder:
    def __init__(self) -> None:
        self._bindings: dict[int, _BoundDFBAllocationGroup] = {}

    def bind(self, group: DFBAllocationGroup) -> _BoundDFBAllocationGroup:
        identity = id(group)
        binding = self._bindings.get(identity)
        if binding is None:
            binding = _BoundDFBAllocationGroup(group, len(self._bindings))
            self._bindings[identity] = binding
        return binding


_CURRENT_BINDER: ContextVar[_DFBAllocationGroupBinder | None] = ContextVar(
    "ttl_dfb_allocation_group_binder", default=None
)


@contextmanager
def _dfb_allocation_group_binding_scope(
    initial_groups: Iterable[DFBAllocationGroup] = (),
) -> Iterator[None]:
    binder = _DFBAllocationGroupBinder()
    for group in initial_groups:
        binder.bind(group)
    token = _CURRENT_BINDER.set(binder)
    try:
        yield
    finally:
        _CURRENT_BINDER.reset(token)


def _bind_current_dfb_allocation_group(
    group: DFBAllocationGroup,
) -> _BoundDFBAllocationGroup:
    binder = _CURRENT_BINDER.get()
    if binder is None:
        raise TypeError("DFBAllocationGroup use requires an enclosing @ttl.operation")
    return binder.bind(group)


def _bind_dfb_allocation_groups(
    groups: Mapping[str, DFBAllocationGroup],
) -> dict[str, _BoundDFBAllocationGroup]:
    binder = _DFBAllocationGroupBinder()
    return {name: binder.bind(group) for name, group in groups.items()}


def _dfb_allocation_group_topology(
    groups: Mapping[str, DFBAllocationGroup],
) -> tuple[int, ...]:
    bindings = _bind_dfb_allocation_groups(groups)
    return tuple(binding.ordinal for binding in bindings.values())


def make_dfb_allocation_group() -> DFBAllocationGroup:
    """Create one immutable compile-time DFB allocation identity."""
    return DFBAllocationGroup()


__all__ = ["DFBAllocationGroup", "make_dfb_allocation_group"]
