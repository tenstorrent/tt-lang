# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# RUN: %python %s | FileCheck %s
#
# Ensure TTL bufferization interfaces are registered in Python by running the
# upstream one-shot-bufferize pass over a tiny module with ttl.attach_cb.

from ttmlir import ir
from ttmlir import passmanager
from ttlang.dialects import ttl

MODULE = r"""
module {
  func.func @attach(%arg0: tensor<4xf32>) {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], f32, 2>
    %attached = ttl.attach_cb %arg0, %cb
        : (tensor<4xf32>, !ttl.cb<[4], f32, 2>) -> tensor<4xf32>
    func.return
  }

  func.func @dual_attach(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], f32, 2>
    %use0 = ttl.attach_cb %arg0, %cb0
        : (tensor<4xf32>, !ttl.cb<[4], f32, 2>) -> tensor<4xf32>
    %use1 = ttl.attach_cb %arg1, %cb1
        : (tensor<4xf32>, !ttl.cb<[4], f32, 2>) -> tensor<4xf32>
    func.return
  }
}
"""


def main():
    with ir.Context() as ctx, ir.Location.unknown():
        ttl.ensure_dialects_registered(ctx)

        module = ir.Module.parse(MODULE)
        pm = passmanager.PassManager.parse(
            "builtin.module(one-shot-bufferize{bufferize-function-boundaries=false allow-unknown-ops=true})"
        )
        pm.run(module.operation)
        print(module)


# CHECK-LABEL: module
# CHECK: func.func @attach
# CHECK: %[[BUF:.*]] = bufferization.to_buffer
# CHECK-SAME: memref<4xf32, strided<[?], offset: ?>>
# CHECK: %[[ATTACHED:.*]] = ttl.attach_cb %[[BUF]]
# CHECK-SAME: memref<4xf32, strided<[?], offset: ?>>
# CHECK: return
# CHECK: func.func @dual_attach
# CHECK-DAG: %[[XBUF:.*]] = bufferization.to_buffer %arg0
# CHECK-DAG: %[[YBUF:.*]] = bufferization.to_buffer %arg1
# CHECK: ttl.attach_cb %[[XBUF]]
# CHECK: ttl.attach_cb %[[YBUF]]


if __name__ == "__main__":
    main()
