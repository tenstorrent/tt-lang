# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Binary operation tests.

Test classes are auto-generated from TTLElementwiseOps.def.
Import them into module namespace so pytest discovers them.
"""

from . import ELEMENTWISE_OPS, GENERATED_OP_TESTS

# Import auto-generated binary op test classes into this module.
# This makes pytest discover them as test classes.
for name, cls in GENERATED_OP_TESTS.items():
    if ELEMENTWISE_OPS.get(cls.OP_STR) == 2:  # Binary ops have arity 2
        globals()[name] = cls
