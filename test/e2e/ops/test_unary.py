# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unary operation tests.

Each test class defines OP_STR and optionally INPUT_RANGE for domain constraints.
"""

from . import UnaryOpTestBase


class TestExp(UnaryOpTestBase):
    OP_STR = "exp"


class TestLog(UnaryOpTestBase):
    OP_STR = "log"
    INPUT_RANGE = (0.01, 10.0)  # Domain constraint: log requires positive inputs


class TestSqrt(UnaryOpTestBase):
    OP_STR = "sqrt"
    INPUT_RANGE = (0.01, 10.0)  # Domain constraint: sqrt requires positive inputs


class TestRsqrt(UnaryOpTestBase):
    OP_STR = "rsqrt"
    INPUT_RANGE = (0.01, 10.0)  # Domain constraint: rsqrt requires positive inputs


class TestTanh(UnaryOpTestBase):
    OP_STR = "tanh"


class TestAbs(UnaryOpTestBase):
    OP_STR = "abs"


class TestNeg(UnaryOpTestBase):
    OP_STR = "neg"


class TestRelu(UnaryOpTestBase):
    OP_STR = "relu"


class TestSigmoid(UnaryOpTestBase):
    OP_STR = "sigmoid"
