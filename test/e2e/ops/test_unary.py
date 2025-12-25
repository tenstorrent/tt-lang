# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unary operation tests.

Test classes are auto-generated from TTLElementwiseOps.def.
Import them into module namespace so pytest discovers them.
"""

from . import ELEMENTWISE_OPS, GENERATED_OP_TESTS

# Import auto-generated unary op test classes into this module.
# This makes pytest discover them as test classes.
for name, cls in GENERATED_OP_TESTS.items():
    if ELEMENTWISE_OPS.get(cls.OP_STR) == 1:  # Unary ops have arity 1
        globals()[name] = cls
