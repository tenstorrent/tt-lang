// RUN: ttlang-opt --verify-diagnostics --split-input-file %s
// Summary: Invalid ttl.copy cases rejected by the CopyOp verifier.

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @tensor_to_tensor_invalid(%arg0: tensor<32x32xf32, #layout>, %arg1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{expects exactly one operand to be !ttl.cb}}
    %xf = ttl.copy %arg0, %arg1 : (tensor<32x32xf32, #layout>, tensor<32x32xf32, #layout>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

// CB-to-CB copy is invalid. CBs are created inside kernels, not passed as arguments.
module {
  func.func @cb_to_cb_invalid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{expects exactly one operand to be !ttl.cb}}
    %xf = ttl.copy %cb0, %cb1 : (!ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

// Tensor operand must carry TTNNLayout encoding.
module {
  func.func @tensor_missing_layout_invalid(%arg0: tensor<32x32xf32>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{expects tensor operand to carry TTNNLayout encoding}}
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

// Non-CB operand must be a ranked tensor.
module {
  func.func @non_tensor_operand_invalid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : i32
    // expected-error @below {{expects the non-CB operand to be a ranked tensor}}
    %xf = ttl.copy %c0, %cb : (i32, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}
