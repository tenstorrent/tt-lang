// RUN: ttlang-opt --verify-diagnostics --split-input-file %s
// Summary: Invalid ttl.copy and ttl.wait operands rejected by op verifiers.

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @tensor_to_tensor_invalid(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %arg1: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{expects exactly one operand to be !ttl.cb}}
    %xf = ttl.copy %arg0, %arg1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// CB-to-CB copy is invalid. CBs are created inside kernels, not passed as arguments.
module {
  func.func @cb_to_cb_invalid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{expects exactly one operand to be !ttl.cb}}
    %xf = ttl.copy %cb0, %cb1 : (!ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Tensor operand must carry ttl.layout encoding.
module {
  func.func @tensor_missing_layout_invalid(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{expects tensor operand to carry ttl.layout encoding}}
    %xf = ttl.copy %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Non-CB operand must be a ranked tensor.
module {
  func.func @non_tensor_operand_invalid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
    %int_val = arith.constant 0 : i32
    // expected-error @below {{expects the non-CB operand to be a ranked tensor}}
    %xf = ttl.copy %int_val, %cb : (i32, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Tensor element type must match the CB element type.
#layout_f32 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                          buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @copy_element_type_mismatch(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_f32>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{tensor element type ('!ttcore.tile<32x32, f32>') must match CB element type ('!ttcore.tile<32x32, bf16>')}}
    %xf = ttl.copy %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout_f32>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Tensor block shape must match the CB shape.
#layout_2x1 = #ttl.layout<shape = [2, 1], element_type = !ttcore.tile<32x32, f32>,
                          buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @copy_shape_mismatch(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>, #layout_2x1>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    // expected-error @below {{tensor shape dimension 0 (2) must match CB shape dimension (1)}}
    %xf = ttl.copy %arg0, %cb : (tensor<2x1x!ttcore.tile<32x32, f32>, #layout_2x1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
