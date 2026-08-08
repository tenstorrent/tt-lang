# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Target-independent logical kernel selectors for unified operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Optional, Tuple, Union

from .dialects._ttl_enum_gen import LogicalKernelKind as _TableGenLogicalKernelKind


_PIPE_SOURCE_KERNEL_ROLE: Final[str] = "pipe_source"


class KernelKind(Enum):
    """A portable class of kernels supported by a target backend."""

    COMPUTE = str(_TableGenLogicalKernelKind.Compute)
    DATA_MOVEMENT = str(_TableGenLogicalKernelKind.DataMovement)


@dataclass(frozen=True)
class _KernelIdentity:
    name: str
    operation: Optional[str]
    implicit_role: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Kernel identity must be a nonempty string")
        if self.operation is not None and (
            not isinstance(self.operation, str) or not self.operation
        ):
            raise ValueError("Kernel operation identity must be a nonempty string")
        if self.implicit_role is not None and (
            not isinstance(self.implicit_role, str) or not self.implicit_role
        ):
            raise ValueError("Kernel implicit role must be a nonempty string")
        if (self.operation is None) == (self.implicit_role is None):
            raise ValueError(
                "Kernel identity requires exactly one operation or implicit role"
            )


class _KernelBinding:
    """One-time mutable holder for an immutable logical identity."""

    def __init__(self) -> None:
        self._identity: Optional[_KernelIdentity] = None

    @property
    def identity(self) -> Optional[_KernelIdentity]:
        return self._identity

    def bind(self, identity: _KernelIdentity) -> None:
        if self._identity is not None:
            raise ValueError(
                f"Kernel is already bound as {self._identity.name!r} to "
                f"operation {self._identity.operation!r}"
            )
        self._identity = identity


@dataclass(frozen=True, eq=False)
class Kernel:
    """An operation-local logical kernel with a stable source identity.

    Kernel handles are static operation resources. Operation registration or
    setup binds the handle's capture or assignment name as its identity.
    """

    kind: KernelKind
    _binding: _KernelBinding = field(
        default_factory=_KernelBinding,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, KernelKind):
            raise TypeError(
                "Kernel kind must be a KernelKind, got " f"{type(self.kind).__name__}"
            )

    @classmethod
    def _create_bound(
        cls,
        kind: KernelKind,
        identity: str,
        operation_identity: Optional[str] = None,
        implicit_role: Optional[str] = None,
    ) -> "Kernel":
        kernel = cls(kind)
        kernel._binding.bind(
            _KernelIdentity(identity, operation_identity, implicit_role)
        )
        return kernel

    def _bind(
        self,
        identity: str,
        operation_identity: str,
    ) -> "Kernel":
        """Bind this declaration to one operation and return the same handle."""
        self._binding.bind(
            _KernelIdentity(
                identity,
                operation_identity,
                implicit_role=None,
            )
        )
        return self

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
        return cls._create_bound(kind, f"<{role}>", implicit_role=role)

    @property
    def _identity(self) -> Optional[str]:
        identity = self._binding.identity
        return identity.name if identity is not None else None

    @property
    def _operation_identity(self) -> Optional[str]:
        identity = self._binding.identity
        return identity.operation if identity is not None else None

    @property
    def _implicit_role(self) -> Optional[str]:
        identity = self._binding.identity
        return identity.implicit_role if identity is not None else None

    @property
    def identity(self) -> str:
        identity = self._binding.identity
        if identity is None:
            raise ValueError(
                "Kernel has no operation-local identity; declare it as a "
                "capture or top-level assignment in @ttl.operation"
            )
        return identity.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, Kernel):
            return NotImplemented
        identity = self._binding.identity
        other_identity = other._binding.identity
        if identity is None or other_identity is None:
            raise ValueError(
                "Kernel equality requires operation-local identities; declare "
                "both kernels as captures or top-level assignments in "
                "@ttl.operation"
            )
        return (self.kind, identity) == (other.kind, other_identity)

    def __hash__(self) -> int:
        identity = self._binding.identity
        if identity is None:
            raise ValueError(
                "Kernel hashing requires an operation-local identity; declare "
                "the kernel as a capture or top-level assignment in "
                "@ttl.operation"
            )
        return hash((self.kind, identity))

    def __repr__(self) -> str:
        if self._binding.identity is None:
            return f"Kernel({self.kind!r})"
        return f"Kernel({self.kind!r}, identity={self.identity!r})"


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
