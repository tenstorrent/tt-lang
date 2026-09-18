# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composition fixtures whose module does not bind the name ``ttl``."""

from ttl import KernelKind, ScalarType, call_extern_func, operation


FAKE_HEADER = "/dev/null/scalar_result.hpp"
CAPTURED_RESULT_TYPE = ScalarType.I64


@operation()
def helper_with_scalar_type_alias():
    predicate = call_extern_func(
        FAKE_HEADER,
        "alias_result",
        result_type=ScalarType.I64,
        kernel=KernelKind.COMPUTE,
    )
    call_extern_func(
        FAKE_HEADER, "alias_consume", func_args=[predicate], kernel=KernelKind.COMPUTE
    )


@operation()
def helper_with_captured_scalar_type():
    predicate = call_extern_func(
        FAKE_HEADER,
        "captured_result",
        result_type=CAPTURED_RESULT_TYPE,
        kernel=KernelKind.COMPUTE,
    )
    call_extern_func(
        FAKE_HEADER,
        "captured_consume",
        func_args=[predicate],
        kernel=KernelKind.COMPUTE,
    )


@operation(grid=(1, 1))
def caller_without_ttl_binding(inp):
    helper_with_captured_scalar_type()
