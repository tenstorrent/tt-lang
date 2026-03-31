# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Binary operation tests.

Test classes are auto-generated from TTLElementwiseOps.def.
Import them into module namespace so pytest discovers them.
"""

from . import ELEMENTWISE_OPS, GENERATED_OP_TESTS

# Compare ops: do not register Bfloat16 test classes here; exact 0/1
# mask checks are brittle for BF16 vs float32 golden. Other suites (e.g.
# test_compare_ops) may still exercise those dtypes. Names follow
# generate_op_test_classes: Test{Op}{dtype_suffix} (e.g. TestGtBfloat16).
_COMPARE_SKIP_BF16 = frozenset({"eq", "ne", "gt", "lt"})

# Import auto-generated binary op test classes into this module.
# This makes pytest discover them as test classes.
for name, cls in GENERATED_OP_TESTS.items():
    if ELEMENTWISE_OPS.get(cls.OP_STR) != 2:  # Binary ops have arity 2
        continue
    # TODO: remove this once we have BF16 support for compare ops
    if cls.OP_STR in _COMPARE_SKIP_BF16 and name.endswith("Bfloat16"):
        continue
    globals()[name] = cls
