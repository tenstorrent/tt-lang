# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# RUN: %python %s | FileCheck %s
#
# Ensure TTL bufferization interfaces are registered in Python by running the
# upstream one-shot-bufferize pass over tiny modules with ttl.attach_cb and CB
# protocol ops.

from ttmlir import ir
from ttmlir import passmanager
from ttlang.dialects import ttl

MODULE = r"""
#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

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

  func.func @cb_views() {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], f32, 2>
    %view = ttl.cb_reserve %cb : <[4], f32, 2> -> tensor<4xf32>
    %ready = ttl.cb_wait %cb : <[4], f32, 2> -> tensor<4xf32>
    ttl.cb_push %cb : <[4], f32, 2>
    ttl.cb_pop %cb : <[4], f32, 2>
    func.return
  }

  func.func @copy_to_cb(%t: tensor<32x32xf32, #layout>) {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %t, %cb
        : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>)
        -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
"""


def main():
    with ir.Context() as ctx, ir.Location.unknown():
        ttl.ensure_dialects_registered(ctx)

        module = ir.Module.parse(MODULE)
        pm = passmanager.PassManager.parse(
            "builtin.module(one-shot-bufferize{bufferize-function-boundaries=false allow-unknown-ops=false})"
        )
        pm.run(module.operation)
        print(module)


# CHECK-LABEL: module {
# CHECK-NEXT:   func.func @attach(%arg0: tensor<4xf32>)
# CHECK-NEXT:     %[[BUF:.*]] = bufferization.to_buffer %arg0 : tensor<4xf32> to memref<4xf32{{.*}}>
# CHECK-NEXT:     %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4], f32, 2>
# CHECK-NEXT:     %[[ATTACHED:.*]] = ttl.attach_cb %[[BUF]], %[[CB]] : (memref<4xf32{{.*}}>, !ttl.cb<[4], f32, 2>) -> memref<4xf32{{.*}}>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @dual_attach(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
# CHECK-NEXT:     %[[ARG1_BUF:.*]] = bufferization.to_buffer %arg1 : tensor<4xf32> to memref<4xf32{{.*}}>
# CHECK-NEXT:     %[[ARG0_BUF:.*]] = bufferization.to_buffer %arg0 : tensor<4xf32> to memref<4xf32{{.*}}>
# CHECK-NEXT:     %[[CB0:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4], f32, 2>
# CHECK-NEXT:     %[[CB1:.*]] = ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[4], f32, 2>
# CHECK-NEXT:     %[[ATT0:.*]] = ttl.attach_cb %[[ARG0_BUF]], %[[CB0]] : (memref<4xf32{{.*}}>, !ttl.cb<[4], f32, 2>) -> memref<4xf32{{.*}}>
# CHECK-NEXT:     %[[ATT1:.*]] = ttl.attach_cb %[[ARG1_BUF]], %[[CB1]] : (memref<4xf32{{.*}}>, !ttl.cb<[4], f32, 2>) -> memref<4xf32{{.*}}>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @cb_views()
# CHECK-NEXT:     %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4], f32, 2>
# CHECK-NEXT:     %[[RES:.*]] = ttl.cb_reserve %[[CB]] : <[4], f32, 2> -> memref<4xf32{{.*}}>
# CHECK-NEXT:     %[[WAIT:.*]] = ttl.cb_wait %[[CB]] : <[4], f32, 2> -> memref<4xf32{{.*}}>
# CHECK-NEXT:     ttl.cb_push %[[CB]] : <[4], f32, 2>
# CHECK-NEXT:     ttl.cb_pop %[[CB]] : <[4], f32, 2>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK:   func.func @copy_to_cb(%[[ARG:.*]]: tensor<32x32xf32, #ttnn_layout>)
# CHECK-NEXT:     %[[BUF32:.*]] = bufferization.to_buffer %[[ARG]] : tensor<32x32xf32, #ttnn_layout> to memref<32x32xf32{{.*}}>
# CHECK-NEXT:     %[[CB32:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
# CHECK-NEXT:     %[[XF:.*]] = ttl.copy %[[BUF32]], %[[CB32]] {tensor_type = tensor<32x32xf32, #ttnn_layout>} : (memref<32x32xf32{{.*}}>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
# CHECK-NEXT:     ttl.wait %[[XF]] : !ttl.transfer_handle<read>
# CHECK-NEXT:     return
# CHECK-NEXT:   }
# CHECK-NEXT: }


if __name__ == "__main__":
    main()
