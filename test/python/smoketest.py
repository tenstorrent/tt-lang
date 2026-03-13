# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from ttl.dialects import ttl, ttkernel, ttcore
from ttl.ir import *

with Context() as ctx:

    module = Module.parse(
        """
    module {
      func.func @test_ttl_cb() -> tensor<1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
        %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
        %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
        ttl.cb_push %cb : <[1, 1], f32, 2>
        return %view : tensor<1x1xf32>
      }
    }
    """
    )
    # CHECK: #ttkernel.thread<noc>
    # CHECK: ttl.bind_cb
    # CHECK: ttl.cb_reserve
    # CHECK: ttl.cb_push
    print(str(module))
