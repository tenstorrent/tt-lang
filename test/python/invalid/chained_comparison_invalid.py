# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Chained comparisons produce a source diagnostic before lowering."""

import pytest
import ttl

pytest.importorskip("ttnn", exc_type=ImportError)


@ttl.operation(grid=(1, 1))
def invalid_chained_comparison():
    @ttl.compute()
    def compute():
        node_x, _ = ttl.node(dims=2)
        if 0 < node_x < 2:
            pass


if __name__ == "__main__":
    invalid_chained_comparison()


# CHECK: error: chained comparisons are not supported
# CHECK: if 0 < node_x < 2:
