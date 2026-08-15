# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative runtime resources for a TT-Lang operation invocation."""

from dataclasses import dataclass

from .domains import DeviceRef
from .fabric import FabricManagerClaim
from .kernel import KernelSelector


@dataclass(frozen=True)
class CoreRuntimeArgs:
    core: object
    values: tuple[int, ...]


@dataclass(frozen=True)
class KernelDefine:
    name: str
    value: str


@dataclass(frozen=True)
class KernelRuntimeResources:
    kernel: KernelSelector
    runtime_args: tuple[CoreRuntimeArgs, ...] = ()
    defines: tuple[KernelDefine, ...] = ()


@dataclass(frozen=True)
class FabricConnectionRequirement:
    local_device: DeviceRef
    remote_device: DeviceRef
    worker_nodes: tuple[tuple[int, int], ...]
    fixed_link_index: int


@dataclass(frozen=True)
class FabricConnectionBinding:
    claim: FabricManagerClaim
    connections: tuple[FabricConnectionRequirement, ...]
    abi_identity: str
    lifetimes: tuple[object, ...] = ()


@dataclass(frozen=True)
class ProgramRuntimeResources:
    semaphore_descriptors: tuple[object, ...] = ()
    kernel_resources: tuple[KernelRuntimeResources, ...] = ()
    lifetimes: tuple[object, ...] = ()
    fabric_connections: tuple[FabricConnectionBinding, ...] = ()


__all__ = [
    "CoreRuntimeArgs",
    "KernelDefine",
    "KernelRuntimeResources",
    "FabricConnectionRequirement",
    "FabricConnectionBinding",
    "ProgramRuntimeResources",
]
