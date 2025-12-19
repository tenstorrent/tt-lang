// RUN: ttlang-opt %s --one-shot-bufferize="bufferize-function-boundaries=false allow-unknown-ops=false" | FileCheck %s
// Verify ttl.attach_cb survives bufferization and swaps tensors for memrefs.

module {
// CHECK-LABEL: func.func @attach
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>)
// CHECK-NEXT:   %[[BUF:.*]] = bufferization.to_buffer %[[ARG0]] : tensor<4xf32> to memref<4xf32, strided<[?], offset: ?>>
// CHECK-NEXT:   %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[4], f32, 2>
// CHECK-NEXT:   %[[ATTACHED:.*]] = ttl.attach_cb %[[BUF]], %[[CB]] : (memref<4xf32, strided<[?], offset: ?>>, !ttl.cb<[4], f32, 2>) -> memref<4xf32, strided<[?], offset: ?>>
// CHECK-NEXT:   return
  func.func @attach(%arg0: tensor<4xf32>) {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2}
        : !ttl.cb<[4], f32, 2>
    %attached = ttl.attach_cb %arg0, %cb
        : (tensor<4xf32>, !ttl.cb<[4], f32, 2>)
        -> tensor<4xf32>
    func.return
  }
}
