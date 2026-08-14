# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Verify that chained comparisons produce a source diagnostic."""

import torch
import ttl
import ttnn

from ttlang_test_utils import to_dram


@ttl.operation(grid=(1, 1))
def invalid_chained_comparison(input_tensor):
    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def reader():
        node_x, _ = ttl.node(dims=2)
        if 0 < node_x < 2:
            pass

    @ttl.datamovement()
    def writer():
        pass


# CHECK: error: chained comparisons are not supported
# CHECK: if 0 < node_x < 2:


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        input_tensor = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
        invalid_chained_comparison(input_tensor)
    finally:
        ttnn.close_device(device)
