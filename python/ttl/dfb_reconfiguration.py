# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed dataflow-buffer configuration-epoch declarations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Mapping

from .kernel import Kernel, KernelKind, KernelSelector


@dataclass(frozen=True, eq=False)
class DFBReconfiguration:
    """One worker-local synchronized DFB configuration-epoch boundary.

    A ``KernelKind`` names the operation's canonical logical kernel of that
    kind. A ``Kernel`` handle names a specific logical kernel captured by the
    enclosing operation. Every participant executes the same dynamic boundary
    instances in the same order. Each boundary executes zero or one dynamic
    instance per dispatch and launch node. DFB-interface work ordered before the
    boundary completes before the next epoch's compiler-derived configuration
    is installed. Independent math and SFPU work may overlap the boundary.
    """

    participants: tuple[KernelSelector, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.participants, tuple) or not self.participants:
            raise TypeError("DFBReconfiguration participants must be a nonempty tuple")
        seen_kinds: set[KernelKind] = set()
        seen_kernels: set[int] = set()
        for participant in self.participants:
            if isinstance(participant, KernelKind):
                if participant in seen_kinds:
                    raise ValueError("DFBReconfiguration participants must be distinct")
                seen_kinds.add(participant)
                continue
            if not isinstance(participant, Kernel):
                raise TypeError(
                    "DFBReconfiguration participants must contain only "
                    "KernelKind or Kernel values, got "
                    f"{type(participant).__name__}"
                )
            identity = id(participant)
            if identity in seen_kernels:
                raise ValueError("DFBReconfiguration participants must be distinct")
            seen_kernels.add(identity)
        participant_kinds = [
            participant if isinstance(participant, KernelKind) else participant.kind
            for participant in self.participants
        ]
        if (
            len(participant_kinds) != 3
            or participant_kinds.count(KernelKind.COMPUTE) != 1
            or participant_kinds.count(KernelKind.DATA_MOVEMENT) != 2
        ):
            raise ValueError(
                "DFBReconfiguration requires one compute and two data "
                "movement participants"
            )


@dataclass(frozen=True)
class _BoundDFBReconfiguration:
    declaration: DFBReconfiguration
    ordinal: int

    @property
    def participants(self) -> tuple[KernelSelector, ...]:
        return self.declaration.participants


class _DFBReconfigurationBinder:
    def __init__(self) -> None:
        self._bindings: dict[int, _BoundDFBReconfiguration] = {}

    def bind(self, boundary: DFBReconfiguration) -> _BoundDFBReconfiguration:
        identity = id(boundary)
        binding = self._bindings.get(identity)
        if binding is None:
            binding = _BoundDFBReconfiguration(boundary, len(self._bindings))
            self._bindings[identity] = binding
        return binding


_CURRENT_BINDER: ContextVar[_DFBReconfigurationBinder | None] = ContextVar(
    "ttl_dfb_reconfiguration_binder", default=None
)


@contextmanager
def _dfb_reconfiguration_binding_scope() -> Iterator[None]:
    token = _CURRENT_BINDER.set(_DFBReconfigurationBinder())
    try:
        yield
    finally:
        _CURRENT_BINDER.reset(token)


def _bind_current_dfb_reconfiguration(
    boundary: DFBReconfiguration,
) -> _BoundDFBReconfiguration:
    binder = _CURRENT_BINDER.get()
    if binder is None:
        raise TypeError(
            "DFBReconfiguration capture requires an enclosing @ttl.operation"
        )
    return binder.bind(boundary)


def _bind_dfb_reconfigurations(
    boundaries: Mapping[str, DFBReconfiguration],
) -> dict[str, _BoundDFBReconfiguration]:
    binder = _DFBReconfigurationBinder()
    return {name: binder.bind(boundary) for name, boundary in boundaries.items()}


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
        "DFBReconfiguration participant Kernel must be captured by the "
        "enclosing @ttl.operation"
    )


def _dfb_reconfiguration_topology(
    boundaries: Mapping[str, DFBReconfiguration],
    logical_kernels: Mapping[str, Kernel],
) -> tuple[tuple[int, tuple[tuple[str, str], ...]], ...]:
    bindings = _bind_dfb_reconfigurations(boundaries)
    return tuple(
        (
            binding.ordinal,
            tuple(
                sorted(
                    _participant_topology(participant, logical_kernels)
                    for participant in binding.participants
                )
            ),
        )
        for binding in bindings.values()
    )


__all__ = ["DFBReconfiguration"]
