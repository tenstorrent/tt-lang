# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

"""Verify shape-view diagnostics before any IR mutation, without a device."""

from ttl import operators
from ttl.dialects import func
from ttl.ir import Context, InsertionPoint, Location, Module, Type


def check_invalid(operation, type_text, dims, message):
    module = Module.create()
    with InsertionPoint(module.body):
        function = func.FuncOp("invalid", ([Type.parse(type_text)], []))
        entry = function.add_entry_block()
        with InsertionPoint(entry):
            try:
                operation(entry.arguments[0], dims=dims)
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError(
                    f"{operation.__name__} accepted invalid dimensions"
                )
            assert len(entry.operations) == 0
            func.ReturnOp([])
    assert module.operation.verify()


with Context(), Location.unknown():
    # Invalid indices, non-unit removal and duplicate insertion are rejected.
    cases = [
        (operators.squeeze, "tensor<2x3xf32>", [0], "grid size is 2, expected 1"),
        (operators.squeeze, "tensor<1x2xf32>", [2], "only 2 dimensions"),
        (operators.squeeze, "tensor<1x2xf32>", [-3], "only 2 dimensions"),
        (operators.unsqueeze, "tensor<2x3xf32>", [3], "would have 3 dimensions"),
        (operators.unsqueeze, "tensor<2x3xf32>", [-4], "would have 3 dimensions"),
        (operators.unsqueeze, "tensor<2x3xf32>", [0, -4], "duplicate dimension"),
        (operators.squeeze, "tensor<f32>", [0], "only 0 dimensions"),
        (operators.unsqueeze, "tensor<f32>", [1], "would have 1 dimensions"),
        (operators.squeeze, "f32", [], "must be a ranked tensor"),
        (operators.unsqueeze, "f32", [], "must be a ranked tensor"),
        (operators.squeeze, "tensor<1x?xf32>", [0], "requires a static block shape"),
        (operators.unsqueeze, "tensor<?xf32>", [0], "requires a static block shape"),
    ]
    for operation, type_text, dims, message in cases:
        check_invalid(operation, type_text, dims, message)
    print(f"Verified {len(cases)} shape-view diagnostics")
    # CHECK: Verified 12 shape-view diagnostics
