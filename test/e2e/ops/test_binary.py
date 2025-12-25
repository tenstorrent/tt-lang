# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Binary operation tests.

Each test class defines a single OP_STR attribute to specify the operation.
"""

from . import BinaryOpTestBase


class TestAdd(BinaryOpTestBase):
    OP_STR = "add"


class TestSub(BinaryOpTestBase):
    OP_STR = "sub"


class TestMul(BinaryOpTestBase):
    OP_STR = "mul"


class TestMax(BinaryOpTestBase):
    OP_STR = "max"
