// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s
// Summary: Invalid TTL DMA cases to exercise diagnostics for unsupported copy pairs.

// -----

module {
  func.func @tensor_to_tensor_invalid(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) {
    // expected-error @below {{unsupported ttl.copy src/dst combination}}
    %xf = ttl.copy %arg0, %arg1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> !ttl.xf
    ttl.wait %xf
    return
  }
}

// -----

module {
  func.func @cb_to_cb_invalid(%cb0: !ttl.cb<[1, 1], f32, 2>, %cb1: !ttl.cb<[1, 1], f32, 2>) {
    // expected-error @below {{unsupported ttl.copy src/dst combination}}
    %xf = ttl.copy %cb0, %cb1 : (!ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    return
  }
}
