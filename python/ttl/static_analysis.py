# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only entry points for validating TT-Lang operations.

This module deliberately reuses the compiler frontend and validation pipeline.
It stops before TTKernel/EmitC lowering, runtime artifact generation, and all
device access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .atom import _operation_validator


@dataclass(frozen=True)
class StaticTensorSpec:
    """The shape and dtype information needed to analyze a tensor argument."""

    shape: tuple[int, ...]
    dtype: Any
    _ttlang_static_tensor = True

    def __init__(self, shape: Sequence[int], dtype: Any):
        object.__setattr__(self, "shape", tuple(int(dimension) for dimension in shape))
        object.__setattr__(self, "dtype", dtype)

    @property
    def padded_shape(self) -> tuple[int, ...]:
        return self.shape


def build_operation_validator(
    function: Callable,
    *,
    grid,
    fp32_dest_acc_en: Optional[bool] = None,
    dst_full_sync_en: Optional[bool] = None,
    math_fidelity: Optional[str] = None,
    target_arch: Optional[str] = None,
) -> Callable:
    """Return a cached, host-only validator for ``function``.

    Calls accept :class:`StaticTensorSpec` objects in place of TTNN tensors.
    A successful call returns ``None``; compiler diagnostics are raised with
    the same source-aware errors as normal compilation.
    """
    return _operation_validator(
        grid=grid,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=dst_full_sync_en,
        math_fidelity=math_fidelity,
        target_arch=target_arch,
    )(function)


__all__ = ["StaticTensorSpec", "build_operation_validator"]
