// RUN: ttlang-opt %s --one-shot-bufferize="bufferize-function-boundaries=false allow-unknown-ops=false" | FileCheck %s
// Summary: Verify ttl.cb_reserve and ttl.cb_wait bufferize their tensor results
// to memrefs while keeping CB metadata intact.

module {
  // CHECK-LABEL: func.func @reserve
  // CHECK-SAME: ()
  // CHECK-NEXT:   %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
  // CHECK-NEXT:   %[[VIEW:.*]] = ttl.cb_reserve %[[CB]] : <[1, 1], f32, 2> -> memref<1x1xf32, strided<[?, ?], offset: ?>>
  // CHECK-NEXT:   return
  func.func @reserve() {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf32>
    func.return
  }

  // CHECK-LABEL: func.func @wait
  // CHECK-SAME: ()
  // CHECK-NEXT:   %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[2, 2], f32, 2>
  // CHECK-NEXT:   %[[VIEW:.*]] = ttl.cb_wait %[[CB]] : <[2, 2], f32, 2> -> memref<2x2xf32, strided<[?, ?], offset: ?>>
  // CHECK-NEXT:   return
  func.func @wait() {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %view = ttl.cb_wait %cb : <[2, 2], f32, 2> -> tensor<2x2xf32>
    func.return
  }
}

