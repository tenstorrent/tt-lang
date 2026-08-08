# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Target-independent logical kernel selectors for unified operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union


class KernelKind(Enum):
    """A portable class of kernels supported by a target backend."""

    COMPUTE = "compute"
    DATA_MOVEMENT = "data_movement"


@dataclass(frozen=True, eq=False)
class Kernel:
    """An operation-local logical kernel with a stable source identity.

    A factory-owned handle remains unbound so composed operations can share it.
    Compilation binds an immutable operation-local copy using the handle's
    final capture or assignment name.
    """

    kind: KernelKind
    _identity: Optional[str] = field(default=None, repr=False)
    _operation_identity: Optional[str] = field(default=None, repr=False)
    _implicit_role: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, KernelKind):
            raise TypeError(
                "Kernel kind must be a KernelKind, got " f"{type(self.kind).__name__}"
            )
        if self._identity is not None and (
            not isinstance(self._identity, str) or not self._identity
        ):
            raise ValueError("Kernel identity must be a nonempty string")
        if self._operation_identity is not None and (
            not isinstance(self._operation_identity, str)
            or not self._operation_identity
        ):
            raise ValueError("Kernel operation identity must be a nonempty string")
        if self._implicit_role is not None and (
            not isinstance(self._implicit_role, str) or not self._implicit_role
        ):
            raise ValueError("Kernel implicit role must be a nonempty string")
        if self._identity is None:
            if self._operation_identity is not None or self._implicit_role is not None:
                raise ValueError(
                    "Unbound Kernel cannot have an operation identity or implicit role"
                )
        elif (self._operation_identity is None) == (self._implicit_role is None):
            raise ValueError(
                "Bound Kernel requires exactly one operation identity or implicit role"
            )

    @classmethod
    def _create_bound(
        cls,
        kind: KernelKind,
        identity: str,
        operation_identity: Optional[str] = None,
        implicit_role: Optional[str] = None,
    ) -> "Kernel":
        return cls(kind, identity, operation_identity, implicit_role)

    def _bind(self, identity: str, operation_identity: str) -> "Kernel":
        if self._identity is not None:
            if (
                self._identity != identity
                or self._operation_identity != operation_identity
                or self._implicit_role is not None
            ):
                raise ValueError(
                    f"Kernel is already bound as {self._identity!r} to operation "
                    f"{self._operation_identity!r}"
                )
            return self
        if not isinstance(identity, str) or not identity:
            raise ValueError("Kernel identity must be a nonempty string")
        if not isinstance(operation_identity, str) or not operation_identity:
            raise ValueError("Kernel operation identity must be a nonempty string")
        return self._create_bound(
            self.kind,
            identity,
            operation_identity=operation_identity,
        )

    @classmethod
    def _from_metadata(
        cls,
        kind: KernelKind,
        identity: str,
        operation_identity: Optional[str],
        implicit_role: Optional[str] = None,
    ) -> "Kernel":
        return cls._create_bound(
            kind,
            identity,
            operation_identity=operation_identity,
            implicit_role=implicit_role,
        )

    @classmethod
    def _implicit(cls, kind: KernelKind, role: str) -> "Kernel":
        return cls._create_bound(
            kind,
            f"<{role}>",
            implicit_role=role,
        )

    @property
    def identity(self) -> str:
        if self._identity is None:
            raise ValueError(
                "Kernel has no operation-local identity; declare it as a "
                "capture or top-level assignment in @ttl.operation"
            )
        return self._identity

    def __eq__(self, other) -> bool:
        if not isinstance(other, Kernel):
            return NotImplemented
        if self._identity is None or other._identity is None:
            raise ValueError(
                "Kernel equality requires operation-local identities; declare "
                "both kernels as captures or top-level assignments in "
                "@ttl.operation"
            )
        return (
            self.kind,
            self._identity,
            self._operation_identity,
            self._implicit_role,
        ) == (
            other.kind,
            other._identity,
            other._operation_identity,
            other._implicit_role,
        )

    def __hash__(self) -> int:
        if self._identity is None:
            raise ValueError(
                "Kernel hashing requires an operation-local identity; declare "
                "the kernel as a capture or top-level assignment in "
                "@ttl.operation"
            )
        return hash(
            (
                self.kind,
                self._identity,
                self._operation_identity,
                self._implicit_role,
            )
        )

    def __repr__(self) -> str:
        if self._identity is None:
            return f"Kernel({self.kind!r})"
        return f"Kernel({self.kind!r}, identity={self._identity!r})"


KernelSelector = Union[KernelKind, Kernel]
ExternalKernelSelection = Union[KernelSelector, Tuple[KernelSelector, ...]]
ReleaseKernelSelection = KernelSelector


def _selector_kind(selector: KernelSelector) -> KernelKind:
    if isinstance(selector, KernelKind):
        return selector
    return selector.kind


def _selector_sort_key(selector: KernelSelector):
    kind_order = {
        KernelKind.COMPUTE: 0,
        KernelKind.DATA_MOVEMENT: 1,
    }
    if isinstance(selector, KernelKind):
        return kind_order[selector], 0, ""
    role_order = 1 if selector._implicit_role is not None else 2
    return kind_order[selector.kind], role_order, selector.identity


def _selector_implicit_role(selector: KernelSelector) -> Optional[str]:
    if isinstance(selector, KernelKind):
        return None
    return selector._implicit_role


def _format_selector(selector: KernelSelector) -> str:
    if isinstance(selector, KernelKind):
        return selector.value
    return f"{selector.kind.value} kernel {selector.identity!r}"


__all__ = [
    "Kernel",
    "KernelKind",
    "KernelSelector",
    "ExternalKernelSelection",
    "ReleaseKernelSelection",
]
