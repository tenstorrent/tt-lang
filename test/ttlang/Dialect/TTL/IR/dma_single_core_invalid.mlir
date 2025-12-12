// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s
// Summary: Invalid TTL DMA cases to exercise diagnostics for unsupported copy pairs.

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// expected-error @below {{failed to legalize operation 'ttl.copy'}}
module {
  func.func @tensor_to_tensor_invalid(%arg0: tensor<32x32xf32, #layout>, %arg1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %xf = ttl.copy %arg0, %arg1 : (tensor<32x32xf32, #layout>, tensor<32x32xf32, #layout>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

// CB-to-CB copy is invalid. CBs are created inside kernels, not passed as arguments.
// expected-error @below {{failed to legalize operation 'ttl.copy'}}
module {
  func.func @cb_to_cb_invalid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb0, %cb1 : (!ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}
