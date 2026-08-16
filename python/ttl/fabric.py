# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative ownership for external fabric connection managers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional

from .domains import DeviceRef
from .kernel import Kernel


class FabricManagerEffectKind(Enum):
    ACQUIRE = "acquire"
    USE = "use"
    RELEASE = "release"
    SCOPED = "scoped"


@dataclass(frozen=True)
class FabricManagerEffect:
    claim: "FabricManagerClaim"
    kind: FabricManagerEffectKind


class _FabricManagerClaimBinding:
    def __init__(self) -> None:
        self.operation: Optional[str] = None

    def bind(self, operation: str, name: str) -> None:
        if self.operation is not None:
            raise ValueError(
                f"FabricManagerClaim {name!r} is already bound to operation "
                f"{self.operation!r}"
            )
        self.operation = operation


@dataclass(frozen=True)
class FabricManagerExecutionLocation:
    """One logical device and worker core that may own an external manager."""

    device: DeviceRef
    worker_node: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.device, DeviceRef):
            raise TypeError("FabricManagerExecutionLocation device must be a DeviceRef")
        if (
            not isinstance(self.worker_node, tuple)
            or len(self.worker_node) != 2
            or any(type(coordinate) is not int for coordinate in self.worker_node)
            or any(coordinate < 0 for coordinate in self.worker_node)
        ):
            raise TypeError(
                "FabricManagerExecutionLocation worker_node must be a "
                "non-negative integer coordinate pair"
            )


@dataclass(frozen=True, eq=False)
class FabricManagerClaim:
    """Operation-local identity for one external fabric manager lifetime.

    ``worker_nodes`` declares a finite worker-node domain on every logical
    device. ``execution_locations`` declares an exact device-qualified domain.
    When both are ``None``, the complete logical-kernel range owns the manager.
    """

    name: str
    kernel: Kernel
    worker_nodes: Optional[tuple[tuple[int, int], ...]] = None
    execution_locations: Optional[tuple[FabricManagerExecutionLocation, ...]] = None
    _binding: _FabricManagerClaimBinding = field(
        default_factory=_FabricManagerClaimBinding,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("FabricManagerClaim name must be a nonempty string")
        if not isinstance(self.kernel, Kernel):
            raise TypeError(
                "FabricManagerClaim kernel must be a Kernel, got "
                f"{type(self.kernel).__name__}"
            )
        if self.worker_nodes is not None and self.execution_locations is not None:
            raise ValueError(
                "FabricManagerClaim accepts worker_nodes or execution_locations, "
                "not both"
            )
        if self.worker_nodes is not None:
            if not isinstance(self.worker_nodes, tuple) or not self.worker_nodes:
                raise TypeError(
                    "FabricManagerClaim worker_nodes must be a nonempty tuple"
                )
            for node in self.worker_nodes:
                if (
                    not isinstance(node, tuple)
                    or len(node) != 2
                    or any(type(coordinate) is not int for coordinate in node)
                    or any(coordinate < 0 for coordinate in node)
                ):
                    raise TypeError(
                        "FabricManagerClaim worker_nodes entries must be "
                        "non-negative integer coordinate pairs"
                    )
            if len(set(self.worker_nodes)) != len(self.worker_nodes):
                raise ValueError(
                    "FabricManagerClaim worker_nodes must not contain duplicates"
                )
        if self.execution_locations is not None:
            if (
                not isinstance(self.execution_locations, tuple)
                or not self.execution_locations
            ):
                raise TypeError(
                    "FabricManagerClaim execution_locations must be a nonempty tuple"
                )
            if any(
                not isinstance(location, FabricManagerExecutionLocation)
                for location in self.execution_locations
            ):
                raise TypeError(
                    "FabricManagerClaim execution_locations entries must be "
                    "FabricManagerExecutionLocation values"
                )
            if len(set(self.execution_locations)) != len(self.execution_locations):
                raise ValueError(
                    "FabricManagerClaim execution_locations must not contain "
                    "duplicates"
                )

    @property
    def operation_identity(self) -> str:
        if self._binding.operation is None:
            raise ValueError(
                f"FabricManagerClaim {self.name!r} has no operation identity; "
                "capture it in @ttl.operation"
            )
        return self._binding.operation

    @property
    def identity(self) -> str:
        return self.name

    def _bind(
        self,
        operation_identity: str,
        logical_kernels: Optional[Iterable[Kernel]] = None,
    ) -> None:
        kernel_belongs_to_operation = (
            self.kernel._operation_identity == operation_identity
            if logical_kernels is None
            else any(self.kernel is kernel for kernel in logical_kernels)
        )
        kernel_is_compiler_owned = self.kernel._implicit_role is not None
        if not kernel_belongs_to_operation and not kernel_is_compiler_owned:
            raise ValueError(
                f"FabricManagerClaim {self.name!r} selects a Kernel owned by "
                "another operation"
            )
        self._binding.bind(operation_identity, self.name)

    def _operation_identity_capture(self) -> tuple:
        return (
            "fabric-manager-claim",
            self.name,
            self.kernel.kind.value,
            self.worker_nodes,
            tuple(
                (location.device.coordinates, location.worker_node)
                for location in (self.execution_locations or ())
            ),
        )

    def _effect(self, kind: FabricManagerEffectKind) -> FabricManagerEffect:
        return FabricManagerEffect(self, kind)

    def acquire(self) -> FabricManagerEffect:
        return self._effect(FabricManagerEffectKind.ACQUIRE)

    def use(self) -> FabricManagerEffect:
        return self._effect(FabricManagerEffectKind.USE)

    def release(self) -> FabricManagerEffect:
        return self._effect(FabricManagerEffectKind.RELEASE)

    def scoped(self) -> FabricManagerEffect:
        return self._effect(FabricManagerEffectKind.SCOPED)

    def __hash__(self) -> int:
        return hash((self.name, self.operation_identity))


def _validate_fabric_manager_claims(
    claims: Mapping[str, FabricManagerClaim],
) -> None:
    source_names = {}
    claim_names = {}
    for source_name, claim in claims.items():
        previous_source = source_names.get(id(claim))
        if previous_source is not None:
            raise ValueError(
                "one FabricManagerClaim reached the final operation under "
                f"multiple names: {previous_source!r} and {source_name!r}"
            )
        source_names[id(claim)] = source_name
        previous_claim = claim_names.get(claim.name)
        if previous_claim is not None:
            raise ValueError(
                f"FabricManagerClaim identity {claim.name!r} is declared by "
                f"both {previous_claim!r} and {source_name!r}"
            )
        claim_names[claim.name] = source_name


def _bind_fabric_manager_claims(
    claims: Mapping[str, FabricManagerClaim],
    operation_identity: str,
    logical_kernels: Mapping[str, Kernel],
) -> None:
    _validate_fabric_manager_claims(claims)
    for claim in claims.values():
        claim._bind(operation_identity, logical_kernels.values())


__all__ = [
    "FabricManagerClaim",
    "FabricManagerExecutionLocation",
    "FabricManagerEffect",
    "FabricManagerEffectKind",
]
