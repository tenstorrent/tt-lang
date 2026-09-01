# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the ``ttl.block`` namespace.

Inside a kernel the AST compiler resolves ``ttl.block.fill`` by name, so kernel
bodies work whether or not the attribute exists on the module. These tests
cover the other half: resolution performed by Python itself, when a kernel
module imports, aliases, or introspects an operation at module scope.

Specification revision 0.16 places ``broadcast``, ``transpose`` and ``fill``
under ``ttl.block``, and the reducers under ``ttl.math``.
"""

import inspect

import pytest
import ttl
from ttl import operators

# Operations the specification places under ttl.block and the compiler
# implements.
IMPLEMENTED = ["broadcast", "fill", "transpose"]

# Operations the specification places under ttl.block that the compiler does
# not implement. They are absent rather than bound to a stub, so that a kernel
# using one fails at the call site instead of at run time.
UNIMPLEMENTED = ["mask", "mask_posinf", "squeeze", "unsqueeze", "where"]


def test_block_is_exposed_on_the_package():
    assert inspect.ismodule(ttl.block)


def test_block_is_public():
    assert "block" in ttl.__all__


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_implemented_operation_is_present(name):
    assert callable(getattr(ttl.block, name))


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_operation_is_the_implementation_not_a_wrapper(name):
    assert getattr(ttl.block, name) is getattr(operators, name)


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_operation_signature_is_introspectable(name):
    # Kernel modules probe for optional parameters at import time, which
    # requires a real function object rather than a compiler-side syntax entry.
    assert inspect.signature(getattr(ttl.block, name)).parameters


def test_fill_exposes_specified_parameters():
    parameters = inspect.signature(ttl.block.fill).parameters
    assert {"value", "shape", "dtype", "tile"} <= set(parameters)


def test_fill_dtype_is_optional():
    assert inspect.signature(ttl.block.fill).parameters["dtype"].default is None


@pytest.mark.parametrize("name", UNIMPLEMENTED)
def test_unimplemented_operation_is_absent(name):
    assert not hasattr(ttl.block, name)


@pytest.mark.parametrize("name", ["reduce_sum", "reduce_max"])
def test_reducers_stay_in_the_math_namespace(name):
    assert not hasattr(ttl.block, name)
    assert callable(getattr(ttl.math, name))


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_operation_is_importable_from_the_module(name):
    module = __import__("ttl.ttl_block", fromlist=[name])
    assert getattr(module, name) is getattr(ttl.block, name)


def test_namespace_exports_exactly_the_implemented_operations():
    assert sorted(ttl.ttl_block.__all__) == sorted(IMPLEMENTED)
