# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for closure-captured shape sequences.

A shape written inline as ``shape=(1, 2)`` and a shape captured from the
enclosing scope as ``shape=my_shape`` reach the same consumers, so both spell
the same kernel. These tests cover the capture collection rule and the
end-to-end equivalence of the two spellings.
"""

import pytest
import torch
import ttl

from ttl.ttl_api import _collect_captures

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram


def _captures_of(value):
    """Collect the captures of a function closing over a single value."""

    def outer():
        def inner():
            return value

        return inner

    return _collect_captures(outer())


class TestCollectCaptures:
    """The capture rule that admits shape and axis sequences."""

    def test_tuple_of_ints(self):
        assert _captures_of((1, 1, 4, 8))["value"] == (1, 1, 4, 8)

    def test_list_of_ints(self):
        assert _captures_of([0, 1])["value"] == [0, 1]

    def test_empty_sequence(self):
        assert _captures_of(())["value"] == ()

    def test_floats_are_admitted(self):
        assert _captures_of((0.5, 1.5))["value"] == (0.5, 1.5)

    def test_identity_is_preserved(self):
        """The captured object passes through rather than being rebuilt."""
        shape = (1, 2, 3)
        assert _captures_of(shape)["value"] is shape

    def test_scalar_still_admitted(self):
        assert _captures_of(7)["value"] == 7

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"a": 1}, id="dict"),
            pytest.param({1, 2}, id="set"),
            pytest.param("shape", id="str"),
            pytest.param(("a", "b"), id="tuple_of_str"),
            pytest.param((1, None), id="tuple_with_none"),
            pytest.param(((1, 2), (3, 4)), id="nested_tuple"),
            pytest.param([[1], [2]], id="nested_list"),
        ],
    )
    def test_unsupported_captures_are_refused(self, value):
        """Only flat sequences of scalars are shapes; the rest are still errors."""
        with pytest.raises(TypeError, match="Unhandled capture"):
            _captures_of(value)

    def test_refusal_names_the_type(self):
        with pytest.raises(TypeError, match="dict"):
            _captures_of({"a": 1})


@ttl.operation(grid=(1, 1))
def add_one_captured_shape(inp, out):
    """Shapes are bound in the operation body and captured by the threads."""
    in_shape = (1, 1)
    out_shape = (1, 1)
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=in_shape, block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=out_shape, block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(x + ttl.block.fill(1.0, shape=out_shape))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def add_one_inline_shape(inp, out):
    """The same kernel with every shape written inline."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            o.store(x + ttl.block.fill(1.0, shape=(1, 1)))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "kernel", [add_one_captured_shape, add_one_inline_shape], ids=["captured", "inline"]
)
def test_captured_shape_matches_inline_shape(device, kernel):
    """Both spellings compile and produce the reference result."""
    inp = torch.randn(32, 32, dtype=torch.bfloat16)
    inp_dev = to_dram(inp, device)
    out_dev = to_dram(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    kernel(inp_dev, out_dev)

    torch.testing.assert_close(
        ttnn.to_torch(out_dev).to(torch.float32),
        (inp + 1.0).to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
