# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Target-independent logical kernel selectors for unified operations."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Collection,
    Final,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from ._src.global_semaphore import (
    get_ttnn_global_semaphore_address,
    is_ttnn_global_semaphore,
)
from .condition import DispatchCondition, _bind_dispatch_conditions
from .dfb_allocation_group import (
    DFBAllocationGroup,
    _bind_dfb_allocation_groups,
)
from .dialects._ttl_enum_gen import LogicalKernelKind as _TableGenLogicalKernelKind
from .scalar import ScalarType

_PIPE_SOURCE_KERNEL_ROLE: Final[str] = "pipe_source"
_DFB_RELEASE_METHODS: Final = frozenset(("push", "pop"))


class KernelKind(Enum):
    """A portable class of kernels supported by a target backend."""

    COMPUTE = str(_TableGenLogicalKernelKind.Compute)
    DATA_MOVEMENT = str(_TableGenLogicalKernelKind.DataMovement)

    def __or__(self, other: object) -> tuple[KernelKind, KernelKind]:
        if not isinstance(other, KernelKind):
            return NotImplemented
        return (self, other)


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
    ) -> None:
        """Bind this declaration to one operation."""
        self._binding.bind(
            _KernelIdentity(
                identity,
                operation_identity,
                implicit_role=None,
            )
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


class _KernelParticipantDeclaration(Protocol):
    participants: tuple[KernelSelector, ...]


def _transitive_participant_kernels(
    declarations: Mapping[str, _KernelParticipantDeclaration],
    logical_kernels: Mapping[str, Kernel],
    reserved_names: Collection[str] = (),
    *,
    resource_name: str,
) -> dict[str, Kernel]:
    """Name unbound kernels referenced only through synchronization metadata."""
    participant_names = {id(kernel): name for name, kernel in logical_kernels.items()}
    used_names = set(reserved_names) | set(logical_kernels)
    declaration_ordinals: dict[int, int] = {}
    participants: dict[int, Kernel] = {}
    participant_memberships: dict[int, set[int]] = {}
    for declaration_name in sorted(declarations):
        declaration = declarations[declaration_name]
        declaration_ordinal = declaration_ordinals.setdefault(
            id(declaration), len(declaration_ordinals)
        )
        for participant in declaration.participants:
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
                declaration_ordinal
            )

    participant_groups: dict[tuple[str, tuple[int, ...]], list[Kernel]] = {}
    for participant_identity, participant in participants.items():
        signature = (
            participant.kind.name,
            tuple(sorted(participant_memberships[participant_identity])),
        )
        participant_groups.setdefault(signature, []).append(participant)

    transitive_kernels: dict[str, Kernel] = {}
    for (kernel_kind, declaration_membership), participant_group in sorted(
        participant_groups.items()
    ):
        membership_name = "_".join(str(ordinal) for ordinal in declaration_membership)
        for group_index, participant in enumerate(participant_group):
            name_stem = (
                f"dfb_{resource_name}_participant_{kernel_kind.lower()}_"
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


def _encode_identity_literal(value) -> Optional[bytes]:
    """Encode a supported compile-time literal without Python repr details."""
    if value is None:
        return b"none"
    if isinstance(value, bool):
        return b"bool:true" if value else b"bool:false"
    if isinstance(value, int):
        return f"int:{value}".encode("ascii")
    if isinstance(value, float):
        return f"float:{value.hex()}".encode("ascii")
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return f"str:{len(encoded)}:".encode("ascii") + encoded
    if isinstance(value, ScalarType):
        return f"scalar:{value.name}".encode("ascii")
    if isinstance(value, (tuple, list)):
        elements = []
        for element in value:
            encoded = _encode_identity_literal(element)
            if encoded is None:
                return None
            elements.append(f"{len(encoded)}:".encode("ascii") + encoded)
        kind = b"tuple" if isinstance(value, tuple) else b"list"
        return kind + b":" + b"".join(elements)
    return None


def _encode_identity_capture(
    name: str, value: object, active_functions: set[int]
) -> bytes:
    encoded = _encode_identity_literal(value)
    if encoded is not None:
        return encoded
    if isinstance(value, Kernel):
        return f"kernel-kind:{value.kind.value}".encode("utf-8")
    if is_ttnn_global_semaphore(value):
        address = get_ttnn_global_semaphore_address(value)
        return f"global-semaphore:{address}".encode("ascii")

    semantic_identity = getattr(value, "_operation_identity_capture", None)
    if callable(semantic_identity):
        encoded = _encode_identity_literal(semantic_identity())
        if encoded is not None:
            return b"semantic:" + encoded

    if inspect.ismodule(value):
        return f"module:{value.__name__}".encode("utf-8")
    if inspect.isfunction(value):
        identity = _operation_identity_impl(value, active_functions)
        return f"function:{identity}".encode("utf-8")
    if inspect.isbuiltin(value) or inspect.isclass(value):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", "")
        if module and qualname:
            return f"callable:{module}.{qualname}".encode("utf-8")

    raise TypeError(
        "operation identity cannot encode nonlocal capture "
        f"{name!r} of type {type(value).__name__}"
    )


def _operation_identity_impl(function: Callable, active_functions: set[int]) -> str:
    # Local imports avoid resource-declaration import cycles during module
    # initialization while retaining typed resource checks.
    from .dfb_reset import DFBReset
    from .dfb_reconfiguration import DFBReconfiguration

    function_id = id(function)
    if function_id in active_functions:
        raise TypeError(
            "operation identity cannot encode recursive nonlocal function "
            f"{function.__qualname__!r}"
        )
    active_functions.add(function_id)

    base_identity = f"{function.__module__}.{function.__qualname__}"
    try:
        nonlocal_captures = inspect.getclosurevars(function).nonlocals
    except (TypeError, ValueError):
        active_functions.remove(function_id)
        return base_identity

    try:
        encoded_captures = []
        bound_conditions = _bind_dispatch_conditions(
            {
                name: value
                for name, value in sorted(nonlocal_captures.items())
                if isinstance(value, DispatchCondition)
            }
        )
        bound_allocation_groups = _bind_dfb_allocation_groups(
            {
                name: value
                for name, value in sorted(nonlocal_captures.items())
                if isinstance(value, DFBAllocationGroup)
            }
        )
        reset_ordinals = {}
        reconfiguration_ordinals = {}
        kernel_capture_names = {
            id(value): name
            for name, value in nonlocal_captures.items()
            if isinstance(value, Kernel)
        }
        reset_captures = {
            name: value
            for name, value in nonlocal_captures.items()
            if isinstance(value, DFBReset)
        }
        direct_kernels = {
            name: value
            for name, value in nonlocal_captures.items()
            if isinstance(value, Kernel)
        }
        transitive_reset_kernels = _transitive_participant_kernels(
            reset_captures,
            direct_kernels,
            nonlocal_captures.keys(),
            resource_name="reset",
        )
        kernel_capture_names.update(
            {id(kernel): name for name, kernel in transitive_reset_kernels.items()}
        )
        reconfiguration_captures = {
            name: value
            for name, value in nonlocal_captures.items()
            if isinstance(value, DFBReconfiguration)
        }
        transitive_reconfiguration_kernels = _transitive_participant_kernels(
            reconfiguration_captures,
            {**direct_kernels, **transitive_reset_kernels},
            nonlocal_captures.keys(),
            resource_name="reconfiguration",
        )
        kernel_capture_names.update(
            {
                id(kernel): name
                for name, kernel in transitive_reconfiguration_kernels.items()
            }
        )
        for name, value in sorted(nonlocal_captures.items()):
            if isinstance(value, DispatchCondition):
                binding = bound_conditions[name]
                encoded = (
                    f"dispatch-condition:{binding.ordinal}:"
                    f"{binding.scalar_type.name}"
                ).encode("ascii")
            elif isinstance(value, DFBAllocationGroup):
                binding = bound_allocation_groups[name]
                encoded = f"dfb-allocation-group:{binding.ordinal}".encode("ascii")
            elif isinstance(value, DFBReset):
                reset_identity = id(value)
                ordinal = reset_ordinals.setdefault(reset_identity, len(reset_ordinals))
                participant_tokens = []
                for participant in value.participants:
                    if participant._implicit_role is not None:
                        participant_tokens.append(
                            "role:"
                            f"{participant.kind.name}:"
                            f"{participant._implicit_role}"
                        )
                        continue
                    participant_name = kernel_capture_names.get(id(participant))
                    if participant_name is None:
                        if participant._identity is not None:
                            raise TypeError(
                                "DFBReset participant Kernel is already bound to "
                                f"operation {participant._operation_identity!r}"
                            )
                        raise TypeError(
                            "DFBReset participant Kernel must be captured by "
                            "the enclosing @ttl.operation"
                        )
                    participant_tokens.append(
                        f"kernel:{participant.kind.name}:{participant_name}"
                    )
                encoded = (
                    f"dfb-reset:{ordinal}:" + ",".join(sorted(participant_tokens))
                ).encode("utf-8")
            elif isinstance(value, DFBReconfiguration):
                boundary_identity = id(value)
                ordinal = reconfiguration_ordinals.setdefault(
                    boundary_identity, len(reconfiguration_ordinals)
                )
                participant_tokens = []
                for participant in value.participants:
                    if isinstance(participant, KernelKind):
                        participant_tokens.append(f"kind:{participant.name}")
                        continue
                    if participant._implicit_role is not None:
                        participant_tokens.append(
                            "role:"
                            f"{participant.kind.name}:"
                            f"{participant._implicit_role}"
                        )
                        continue
                    participant_name = kernel_capture_names.get(id(participant))
                    if participant_name is None:
                        raise TypeError(
                            "DFBReconfiguration participant Kernel must be "
                            "captured by the enclosing @ttl.operation"
                        )
                    participant_tokens.append(f"kernel:{participant_name}")
                encoded = (
                    f"dfb-reconfiguration:{ordinal}:"
                    + ",".join(sorted(participant_tokens))
                ).encode("utf-8")
            else:
                encoded = _encode_identity_capture(name, value, active_functions)
            encoded_name = name.encode("utf-8")
            encoded_captures.append(
                f"{len(encoded_name)}:".encode("ascii")
                + encoded_name
                + f"{len(encoded)}:".encode("ascii")
                + encoded
            )
    finally:
        active_functions.remove(function_id)
    if not encoded_captures:
        return base_identity

    digest = hashlib.sha256(b"".join(encoded_captures)).hexdigest()[:16]
    return f"{base_identity}[captures={digest}]"


def _operation_identity(function: Callable) -> str:
    """Return a deterministic semantic identity shared by both operation forms."""
    return _operation_identity_impl(function, set())


def _bind_kernel_declarations(
    logical_kernels: Mapping[str, Kernel], operation_identity: str
) -> None:
    """Bind uniquely named declarations during operation registration."""
    source_names = {}
    for name, kernel in logical_kernels.items():
        previous_name = source_names.get(id(kernel))
        if previous_name is not None:
            raise ValueError(
                "one logical Kernel handle reached the final operation under "
                f"multiple names: {previous_name!r} and {name!r}"
            )
        source_names[id(kernel)] = name
        if kernel._identity is not None:
            raise ValueError(
                f"logical Kernel {name!r} is already bound as "
                f"{kernel.identity!r} to operation "
                f"{kernel._operation_identity!r}"
            )

    for name, kernel in logical_kernels.items():
        kernel._bind(name, operation_identity)


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
        return kind_order[selector], 0, "", "", ""
    role_order = 1 if selector._implicit_role is not None else 2
    return (
        kind_order[selector.kind],
        role_order,
        selector.identity,
        selector._operation_identity or "",
        selector._implicit_role or "",
    )


def _selector_implicit_role(selector: KernelSelector) -> Optional[str]:
    if isinstance(selector, KernelKind):
        return None
    return selector._implicit_role


def _format_selector(selector: KernelSelector) -> str:
    if isinstance(selector, KernelKind):
        return selector.value
    return f"{selector.kind.value} kernel {selector.identity!r}"


def _format_kernel_capacity_error(
    kind: KernelKind,
    selected: Iterable[KernelSelector],
    capacity: int,
) -> str:
    selected = tuple(selected)
    selected_text = ", ".join(_format_selector(selector) for selector in selected)
    required = len(selected)
    return (
        f"operation requires {required} {kind.value} kernels, but the target "
        f"supports {capacity}; selected kernels: {selected_text}"
    )


__all__ = [
    "Kernel",
    "KernelKind",
    "KernelSelector",
    "ExternalKernelSelection",
    "ReleaseKernelSelection",
]
