# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

"""Verify block shape-view emission without TTNN tensors or a device."""

from itertools import combinations, product

from ttl import operators
from ttl.dialects import func
from ttl.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
    RankedTensorType,
    StringAttr,
    Type,
)


def check_view(result, source, operation_name, shape, singleton_dims):
    result_type = RankedTensorType(result.type)
    source_type = RankedTensorType(source.type)
    assert list(result_type.shape) == list(shape)
    assert result_type.element_type == source_type.element_type
    assert result_type.encoding == source_type.encoding
    if not singleton_dims:
        assert result == source
        return

    operation = result.owner
    assert operation.name == operation_name
    assert list(operation.operands) == [source]
    groups = [list(group) for group in operation.attributes["reassociation"]]
    expanded_rank = max(source_type.rank, result_type.rank)
    compressed_rank = min(source_type.rank, result_type.rank)
    assert len(groups) == compressed_rank
    if compressed_rank:
        assert [int(axis) for group in groups for axis in group] == list(
            range(expanded_rank)
        )
        for group in groups:
            assert sum(int(axis) not in singleton_dims for axis in group) == 1
    else:
        assert groups == []


def check_roundtrip(shape, dims, tile_type, encoding=None):
    input_type = RankedTensorType.get(shape, tile_type, encoding)
    module = Module.create()
    with InsertionPoint(module.body):
        function = func.FuncOp("roundtrip", ([input_type], [input_type]))
        entry = function.add_entry_block()
        with InsertionPoint(entry):
            source = entry.arguments[0]
            # Mixed positive/negative duplicates use the original input rank.
            squeeze_dims = list(dims) + [axis - len(shape) for axis in reversed(dims)]
            squeezed = operators.squeeze(source, dims=squeeze_dims)
            squeezed_shape = [
                size for axis, size in enumerate(shape) if axis not in dims
            ]
            check_view(squeezed, source, "tensor.collapse_shape", squeezed_shape, dims)
            # Negative unsqueeze positions refer to the resulting rank.
            restored = operators.unsqueeze(
                squeezed, dims=[axis - len(shape) for axis in reversed(dims)]
            )
            check_view(restored, squeezed, "tensor.expand_shape", shape, dims)
            func.ReturnOp([restored])
    assert module.operation.verify()
    assert "unrealized_conversion_cast" not in str(module)


with Context(), Location.unknown():
    # Cover leading, interleaved, trailing, all-singleton and rank-zero views.
    cases = 0
    for dtype in ("bf16", "f32"):
        tile_type = Type.parse(f"!ttcore.tile<32x32, {dtype}>")
        for rank in range(5):
            for shape in product((1, 2), repeat=rank):
                singleton_axes = [axis for axis, size in enumerate(shape) if size == 1]
                for count in range(len(singleton_axes) + 1):
                    for dims in combinations(singleton_axes, count):
                        check_roundtrip(shape, dims, tile_type)
                        cases += 1
    assert cases == 242
    print(f"Verified {cases} block shape-view roundtrips")
    # CHECK: Verified 242 block shape-view roundtrips

    # Encoding metadata is retained by both directions of a shape view.
    check_roundtrip(
        (1, 2, 1, 3, 1),
        (0, 2, 4),
        Type.parse("!ttcore.tile<32x32, f32>"),
        StringAttr.get("shape-view-encoding"),
    )
    print("Verified encoding preservation")
    # CHECK: Verified encoding preservation
