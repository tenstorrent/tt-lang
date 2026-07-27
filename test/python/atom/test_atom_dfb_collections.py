# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device tests for DFB collections in operations."""

import ast
import importlib
import textwrap
import types

import pytest

import ttl


atom_module = importlib.import_module("ttl.atom")


class _FakeDFB:
    pass


def _function(source: str) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(source)).body[0]


def _lift(source: str, monkeypatch):
    def make_dfb(*positional_args, **keyword_args):
        return _FakeDFB()

    monkeypatch.setattr(atom_module, "DataflowBuffer", _FakeDFB)
    fake_ttl = types.SimpleNamespace(make_dfb=make_dfb)
    return atom_module._lift_setup(
        _function(source),
        {"ttl": fake_ttl},
        "collection_test",
    )


def test_dfb_collection_is_flattened_into_static_captures(monkeypatch):
    stripped, dfbs, pipe_nets = _lift(
        """
        def kernel():
            buffers = [
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                for buffer_index in range(3)
            ]
            first = buffers[0].wait()
            second = buffers[1].reserve()
            third = buffers[2].wait()
        """,
        monkeypatch,
    )

    assert list(dfbs) == ["buffers_0", "buffers_1", "buffers_2"]
    assert pipe_nets == {}
    stripped_source = ast.unparse(stripped)
    assert "buffers =" not in stripped_source
    assert "buffers_0.wait()" in stripped_source
    assert "buffers_1.reserve()" in stripped_source
    assert "buffers_2.wait()" in stripped_source


def test_dfb_collection_accepts_list_literal(monkeypatch):
    stripped, dfbs, pipe_nets = _lift(
        """
        def kernel():
            buffers = [
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2),
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2),
            ]
            first = buffers[0].wait()
            second = buffers[1].reserve()
        """,
        monkeypatch,
    )

    assert list(dfbs) == ["buffers_0", "buffers_1"]
    assert pipe_nets == {}
    stripped_source = ast.unparse(stripped)
    assert "buffers_0.wait()" in stripped_source
    assert "buffers_1.reserve()" in stripped_source


def test_dfb_collection_accepts_static_comprehension_filter(monkeypatch):
    stripped, dfbs, pipe_nets = _lift(
        """
        def kernel():
            buffers = [
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                for buffer_index in range(4)
                if buffer_index % 2 == 0
            ]
            first = buffers[0].wait()
            second = buffers[1].reserve()
        """,
        monkeypatch,
    )

    assert list(dfbs) == ["buffers_0", "buffers_1"]
    assert pipe_nets == {}
    stripped_source = ast.unparse(stripped)
    assert "buffers_0.wait()" in stripped_source
    assert "buffers_1.reserve()" in stripped_source


@pytest.mark.parametrize(("open_target", "close_target"), [("(", ")"), ("[", "]")])
def test_dfb_collection_destructuring_preserves_names(
    open_target, close_target, monkeypatch
):
    stripped, dfbs, pipe_nets = _lift(
        f"""
        def kernel():
            {open_target}
            first_buffer,
            second_buffer,
            third_buffer,
            {close_target} = [
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                for buffer_index in range(3)
            ]
            first = first_buffer.wait()
            second = second_buffer.reserve()
            third = third_buffer.wait()
        """,
        monkeypatch,
    )

    assert list(dfbs) == ["first_buffer", "second_buffer", "third_buffer"]
    assert pipe_nets == {}
    stripped_source = ast.unparse(stripped)
    assert "ttl.make_dfb" not in stripped_source
    assert "first_buffer.wait()" in stripped_source
    assert "second_buffer.reserve()" in stripped_source
    assert "third_buffer.wait()" in stripped_source


def test_dfb_collection_destructuring_rejects_arity_mismatch(monkeypatch):
    with pytest.raises(
        ValueError,
        match="destructuring has 2 targets for 3 elements",
    ):
        _lift(
            """
            def kernel():
                first_buffer, second_buffer = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for buffer_index in range(3)
                ]
            """,
            monkeypatch,
        )


def test_dfb_collection_destructuring_rejects_duplicate_targets(monkeypatch):
    function = _function(
        """
        def kernel():
            buffer, buffer = [
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                for buffer_index in range(2)
            ]
        """
    )

    with pytest.raises(ValueError, match="destructuring targets must be unique"):
        atom_module._validate_resource_declarations(function, "collection_test")


@pytest.mark.parametrize(
    ("index_expression", "message"),
    [
        ("buffer_index", "requires a non-negative integer literal index"),
        ("-1", "requires a non-negative integer literal index"),
        ("True", "requires a non-negative integer literal index"),
        ("2", "index 2 is out of range for 2 elements"),
    ],
)
def test_dfb_collection_rejects_non_static_or_out_of_range_indices(
    index_expression, message, monkeypatch
):
    with pytest.raises(ValueError, match=message):
        _lift(
            f"""
            def kernel():
                buffers = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for collection_index in range(2)
                ]
                buffer_index = 0
                block = buffers[{index_expression}].wait()
            """,
            monkeypatch,
        )


def test_dfb_collection_rejects_empty_comprehension(monkeypatch):
    with pytest.raises(ValueError, match="must contain at least one DFB"):
        _lift(
            """
            def kernel():
                buffers = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for collection_index in range(0)
                ]
            """,
            monkeypatch,
        )


def test_dfb_collection_rejects_generated_name_collision(monkeypatch):
    with pytest.raises(ValueError, match="element name 'buffers_0' conflicts"):
        _lift(
            """
            def kernel():
                buffers = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for collection_index in range(2)
                ]
                buffers_0 = 0
                block = buffers[0].wait()
            """,
            monkeypatch,
        )


def test_dfb_collection_rejects_bare_collection_use(monkeypatch):
    with pytest.raises(ValueError, match="must be accessed with"):
        _lift(
            """
            def kernel():
                buffers = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for collection_index in range(2)
                ]
                alias = buffers
            """,
            monkeypatch,
        )


def test_dfb_collection_rejects_element_rebinding(monkeypatch):
    with pytest.raises(
        ValueError,
        match="DFB collection 'buffers' elements cannot be rebound",
    ):
        _lift(
            """
            def kernel():
                buffers = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for collection_index in range(2)
                ]
                buffers[0] = buffers[1]
            """,
            monkeypatch,
        )


def test_dfb_collection_rejects_nested_thread_declaration():
    function = _function(
        """
        def kernel():
            @ttl.compute()
            def compute():
                buffers = [
                    ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                    for buffer_index in range(2)
                ]
        """
    )

    with pytest.raises(
        ValueError,
        match="resource declaration 'make_dfb' must be a simple top-level assignment",
    ):
        atom_module._validate_resource_declarations(function, "collection_test")


def test_composed_operation_accepts_static_dfb_collection_elements():
    @ttl.operation()
    def copy_stage(source: ttl.DFB, destination: ttl.DFB):
        source_block = source.wait()
        destination_block = destination.reserve()
        destination_block.store(source_block)

    @ttl.operation(grid=(1, 1))
    def chain():
        buffers = [
            ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
            for buffer_index in range(2)
        ]
        copy_stage(buffers[0], buffers[1])

    assert "buffers[0].wait()" in chain._spec.source
    assert "buffers[1].reserve()" in chain._spec.source


def test_composed_operation_rejects_dynamic_dfb_collection_index():
    @ttl.operation()
    def copy_stage(source: ttl.DFB, destination: ttl.DFB):
        source_block = source.wait()
        destination_block = destination.reserve()
        destination_block.store(source_block)

    with pytest.raises(
        TypeError,
        match="must be a resource name or statically indexed DFB collection element",
    ):

        @ttl.operation(grid=(1, 1))
        def chain(buffer_index):
            buffers = [
                ttl.make_dfb("bf16", shape=(1, 1), block_count=2)
                for collection_index in range(2)
            ]
            copy_stage(buffers[buffer_index], buffers[0])
