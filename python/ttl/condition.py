# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed dispatch-stable condition declarations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator
from typing import Mapping

from .scalar import ScalarType


@dataclass(frozen=True, eq=False)
class DispatchCondition:
    """An immutable condition whose truth value is stable per dispatch and node.

    Capture one declaration in an enclosing operation factory, then pass it as
    ``condition_result`` to every repeat-safe external evaluation of the same
    runtime condition. Distinct declarations represent independent conditions.
    """

    scalar_type: ScalarType

    def __post_init__(self) -> None:
        if not isinstance(self.scalar_type, ScalarType):
            raise TypeError(
                "DispatchCondition scalar type must be ttl.ScalarType.I32 or "
                f"ttl.ScalarType.I64, got {type(self.scalar_type).__name__}"
            )


@dataclass(frozen=True)
class _BoundDispatchCondition:
    declaration: DispatchCondition
    ordinal: int

    @property
    def scalar_type(self) -> ScalarType:
        return self.declaration.scalar_type


class _DispatchConditionBinder:
    def __init__(self) -> None:
        self._bindings: dict[int, _BoundDispatchCondition] = {}

    def bind(self, condition: DispatchCondition) -> _BoundDispatchCondition:
        identity = id(condition)
        binding = self._bindings.get(identity)
        if binding is None:
            binding = _BoundDispatchCondition(condition, len(self._bindings))
            self._bindings[identity] = binding
        return binding


_CURRENT_BINDER: ContextVar[_DispatchConditionBinder | None] = ContextVar(
    "ttl_dispatch_condition_binder", default=None
)


@contextmanager
def _dispatch_condition_binding_scope() -> Iterator[None]:
    token = _CURRENT_BINDER.set(_DispatchConditionBinder())
    try:
        yield
    finally:
        _CURRENT_BINDER.reset(token)


def _bind_current_dispatch_condition(
    condition: DispatchCondition,
) -> _BoundDispatchCondition:
    binder = _CURRENT_BINDER.get()
    if binder is None:
        raise TypeError(
            "DispatchCondition capture requires an enclosing @ttl.operation"
        )
    return binder.bind(condition)


def _bind_dispatch_conditions(
    conditions: Mapping[str, DispatchCondition],
) -> dict[str, _BoundDispatchCondition]:
    binder = _DispatchConditionBinder()
    return {name: binder.bind(condition) for name, condition in conditions.items()}


def _dispatch_condition_topology(
    conditions: Mapping[str, DispatchCondition],
) -> tuple[tuple[int, ScalarType], ...]:
    bindings = _bind_dispatch_conditions(conditions)
    return tuple(
        (binding.ordinal, binding.scalar_type) for binding in bindings.values()
    )


__all__ = ["DispatchCondition"]
