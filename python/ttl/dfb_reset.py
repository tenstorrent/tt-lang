# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed synchronized dataflow-buffer reset declarations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Collection, Iterator, Mapping

from .kernel import Kernel, KernelKind


@dataclass(frozen=True, eq=False)
class DFBReset:
    """One worker-local synchronized DFB reset declaration.

    ``participants`` contains one compute ``Kernel`` and two data movement
    ``Kernel`` handles created in the same enclosing operation factory. A call
    to ``ttl.reset_dfbs`` or ``ttl.reset_all_dfbs`` is replicated to those
    three logical kernels. A declaration may execute once or once per iteration
    of the same immutable sequential loop nest in every participant. Runtime
    lowering is supported only on Blackhole.
    """

    participants: tuple[Kernel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.participants, tuple) or not self.participants:
            raise TypeError("DFBReset participants must be a nonempty tuple")
        seen_kernels: set[int] = set()
        for participant in self.participants:
            if not isinstance(participant, Kernel):
                raise TypeError(
                    "DFBReset participants must contain only Kernel values, "
                    f"got {type(participant).__name__}"
                )
            identity = id(participant)
            if identity in seen_kernels:
                raise ValueError("DFBReset participants must be distinct")
            seen_kernels.add(identity)
        participant_kinds = [participant.kind for participant in self.participants]
        compute_count = participant_kinds.count(KernelKind.COMPUTE)
        data_movement_count = participant_kinds.count(KernelKind.DATA_MOVEMENT)
        if compute_count != 1 or data_movement_count != 2:
            raise ValueError(
                "DFBReset participants must contain one compute kernel and "
                "two data movement kernels"
            )


@dataclass(frozen=True)
class _BoundDFBReset:
    declaration: DFBReset
    ordinal: int

    @property
    def participants(self) -> tuple[Kernel, ...]:
        return self.declaration.participants


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


def _transitive_participant_kernels(
    resets: Mapping[str, DFBReset],
    logical_kernels: Mapping[str, Kernel],
    reserved_names: Collection[str] = (),
) -> dict[str, Kernel]:
    """Name Kernel resources referenced only through reset metadata."""
    participant_names = {id(kernel): name for name, kernel in logical_kernels.items()}
    used_names = set(reserved_names) | set(logical_kernels)
    reset_ordinals: dict[int, int] = {}
    participants: dict[int, Kernel] = {}
    participant_memberships: dict[int, set[int]] = {}
    for reset_name in sorted(resets):
        reset = resets[reset_name]
        reset_ordinal = reset_ordinals.setdefault(id(reset), len(reset_ordinals))
        for participant in reset.participants:
            if not isinstance(participant, Kernel):
                continue
            if (
                participant._implicit_role is not None
                or participant._identity is not None
                or id(participant) in participant_names
            ):
                continue
            participant_identity = id(participant)
            participants[participant_identity] = participant
            participant_memberships.setdefault(participant_identity, set()).add(
                reset_ordinal
            )

    participant_groups: dict[tuple[str, tuple[int, ...]], list[Kernel]] = {}
    for participant_identity, participant in participants.items():
        signature = (
            participant.kind.name,
            tuple(sorted(participant_memberships[participant_identity])),
        )
        participant_groups.setdefault(signature, []).append(participant)

    transitive_kernels: dict[str, Kernel] = {}
    for (kernel_kind, reset_membership), participant_group in sorted(
        participant_groups.items()
    ):
        membership_name = "_".join(str(ordinal) for ordinal in reset_membership)
        for group_index, participant in enumerate(participant_group):
            name_stem = (
                f"dfb_reset_participant_{kernel_kind.lower()}_"
                f"{membership_name}_{group_index}"
            )
            participant_name = name_stem
            suffix_index = 0
            while participant_name in used_names:
                suffix_index += 1
                participant_name = f"{name_stem}_{suffix_index}"
            used_names.add(participant_name)
            transitive_kernels[participant_name] = participant
    return transitive_kernels


def _participant_topology(
    participant: Kernel, logical_kernels: Mapping[str, Kernel]
) -> tuple[str, str]:
    for name, kernel in logical_kernels.items():
        if participant is kernel:
            return ("kernel", f"{participant.kind.name}:{name}")
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
) -> tuple[tuple[int, tuple[tuple[str, str], ...]], ...]:
    bindings = _bind_dfb_resets(resets)
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


__all__ = ["DFBReset"]
