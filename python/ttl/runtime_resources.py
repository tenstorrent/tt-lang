# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative runtime resources for a TT-Lang operation invocation."""

from dataclasses import dataclass

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
class ProgramRuntimeResources:
    semaphore_descriptors: tuple[object, ...] = ()
    kernel_resources: tuple[KernelRuntimeResources, ...] = ()
    lifetimes: tuple[object, ...] = ()


__all__ = [
    "CoreRuntimeArgs",
    "KernelDefine",
    "KernelRuntimeResources",
    "ProgramRuntimeResources",
]
