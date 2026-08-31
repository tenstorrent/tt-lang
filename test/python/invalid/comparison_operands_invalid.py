# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Comparisons reject host objects that are not compiler values."""

import pytest
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)


@ttl.operation(grid=(1, 1))
def invalid_comparison_operands():
    @ttl.compute()
    def compute():
        compiler_value, _ = ttl.node(dims=2)
        if compiler_value < "zero":
            pass


if __name__ == "__main__":
    invalid_comparison_operands()


# CHECK: error: comparison operands must be compiler values, got OpResult and str
# CHECK: if compiler_value < "zero":
