# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed synchronized dataflow-buffer reset declarations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Mapping

from .kernel import Kernel, KernelKind, KernelSelector


class DFBResetScope(Enum):
    """Set of worker-local DFB interfaces reset by one declaration."""

    TARGETS = auto()
    ALL_LOCAL = auto()


@dataclass(frozen=True, eq=False)
class DFBReset:
    """One worker-local synchronized DFB reset instance.

    Every participant executes the same dynamic reset instance. The external
    operation must synchronize all participants and reset the read pointer,
    write pointer, occupancy, and initialization state of every declared
    target before any participant returns. ``ALL_LOCAL`` applies the contract
    to every worker-local DFB interface without explicit target operands.
    """

    participants: tuple[KernelSelector, ...]
    scope: DFBResetScope = DFBResetScope.TARGETS

    def __post_init__(self) -> None:
        if not isinstance(self.participants, tuple) or not self.participants:
            raise TypeError("DFBReset participants must be a nonempty tuple")
        seen_kinds: set[KernelKind] = set()
        seen_kernels: set[int] = set()
        for participant in self.participants:
            if isinstance(participant, KernelKind):
                if participant in seen_kinds:
                    raise ValueError("DFBReset participants must be distinct")
                seen_kinds.add(participant)
                continue
            if not isinstance(participant, Kernel):
                raise TypeError(
                    "DFBReset participants must contain only KernelKind or "
                    f"Kernel values, got {type(participant).__name__}"
                )
            identity = id(participant)
            if identity in seen_kernels:
                raise ValueError("DFBReset participants must be distinct")
            seen_kernels.add(identity)
        if not isinstance(self.scope, DFBResetScope):
            raise TypeError(
                "DFBReset scope must be a ttl.DFBResetScope, got "
                f"{type(self.scope).__name__}"
            )


@dataclass(frozen=True)
class _BoundDFBReset:
    declaration: DFBReset
    ordinal: int

    @property
    def participants(self) -> tuple[KernelSelector, ...]:
        return self.declaration.participants

    @property
    def scope(self) -> DFBResetScope:
        return self.declaration.scope


class _DFBResetBinder:
    def __init__(self) -> None:
        self._bindings: dict[int, _BoundDFBReset] = {}

    def bind(self, reset: DFBReset) -> _BoundDFBReset:
        identity = id(reset)
        binding = self._bindings.get(identity)
        if binding is None:
            binding = _BoundDFBReset(reset, len(self._bindings))
            self._bindings[identity] = binding
        return binding


_CURRENT_BINDER: ContextVar[_DFBResetBinder | None] = ContextVar(
    "ttl_dfb_reset_binder", default=None
)


@contextmanager
def _dfb_reset_binding_scope() -> Iterator[None]:
    token = _CURRENT_BINDER.set(_DFBResetBinder())
    try:
        yield
    finally:
        _CURRENT_BINDER.reset(token)


def _bind_current_dfb_reset(reset: DFBReset) -> _BoundDFBReset:
    binder = _CURRENT_BINDER.get()
    if binder is None:
        raise TypeError("DFBReset capture requires an enclosing @ttl.operation")
    return binder.bind(reset)


def _bind_dfb_resets(resets: Mapping[str, DFBReset]) -> dict[str, _BoundDFBReset]:
    binder = _DFBResetBinder()
    return {name: binder.bind(reset) for name, reset in resets.items()}


def _participant_topology(
    participant: KernelSelector, logical_kernels: Mapping[str, Kernel]
) -> tuple[str, str]:
    if isinstance(participant, KernelKind):
        return ("kind", participant.name)
    for name, kernel in logical_kernels.items():
        if participant is kernel:
            return ("kernel", name)
    if participant._identity is not None:
        return (
            "kernel",
            ":".join(
                (
                    participant.kind.name,
                    participant.identity,
                    participant._operation_identity or "",
                    participant._implicit_role or "",
                )
            ),
        )
    raise ValueError(
        "DFBReset participant Kernel must be captured by the enclosing "
        "@ttl.operation"
    )


def _dfb_reset_topology(
    resets: Mapping[str, DFBReset], logical_kernels: Mapping[str, Kernel]
) -> tuple[tuple[int, DFBResetScope, tuple[tuple[str, str], ...]], ...]:
    bindings = _bind_dfb_resets(resets)
    return tuple(
        (
            binding.ordinal,
            binding.scope,
            tuple(
                sorted(
                    _participant_topology(participant, logical_kernels)
                    for participant in binding.participants
                )
            ),
        )
        for binding in bindings.values()
    )


__all__ = ["DFBReset", "DFBResetScope"]
