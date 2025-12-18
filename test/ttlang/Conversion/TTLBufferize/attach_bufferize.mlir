// RUN: ttlang-opt %s --one-shot-bufferize="bufferize-function-boundaries=false allow-unknown-ops=true" | FileCheck %s
// Verify ttl.attach_cb survives bufferization and swaps tensors for memrefs.

module {
// CHECK-LABEL: func.func @attach
// CHECK-SAME: (%[[ARG0:.*]]: memref<4xf32>)
// CHECK-NEXT:   %[[CB:.*]] = ttl.bind_cb
// CHECK-NEXT:   %[[ATTACHED:.*]] = ttl.attach_cb %[[ARG0]], %[[CB]]
// CHECK-SAME: : (memref<4xf32>, !ttl.cb<[4], f32, 2>)
// CHECK-NEXT:   func.return
  func.func @attach(%arg0: tensor<4xf32>) {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2}
        : !ttl.cb<[4], f32, 2>
    %attached = ttl.attach_cb %arg0, %cb
        : (tensor<4xf32>, !ttl.cb<[4], f32, 2>)
        -> tensor<4xf32>
    func.return
  }
}

