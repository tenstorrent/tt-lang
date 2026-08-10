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

// Tensor and CB must both be tiled (reject tile vs scalar element type).
#layout_tiled = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                            buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @copy_tile_vs_scalar_invalid(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_tiled>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{cannot mix tiled and non-tiled element types}}
    %xf = ttl.copy %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout_tiled>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Tensor tile shape must match the CB tile shape (tensor -> CB).
#layout_32 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                         buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @copy_tile_shape_mismatch_tensor_to_cb(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_32>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x16, f32>, 2>
    // expected-error @below {{tensor tile shape (32x32) must match CB tile shape (16x16)}}
    %xf = ttl.copy %arg0, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout_32>, !ttl.cb<[1, 1], !ttcore.tile<16x16, f32>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Tensor tile shape must match the CB tile shape (CB -> tensor).
#layout_16 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<16x16, f32>,
                         buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @copy_tile_shape_mismatch_cb_to_tensor(%arg0: tensor<1x1x!ttcore.tile<16x16, f32>, #layout_16>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    // expected-error @below {{tensor tile shape (16x16) must match CB tile shape (32x32)}}
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, tensor<1x1x!ttcore.tile<16x16, f32>, #layout_16>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
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

// -----

// The copy span cannot exceed the dataflow buffer allocation.
#layout_1x3 = #ttl.layout<shape = [1, 3], element_type = !ttcore.tile<32x32, f32>,
                          buffer = dram, grid = [1, 1], memory = interleaved>

module {
  func.func @copy_span_exceeds_capacity(%arg0: tensor<1x3x!ttcore.tile<32x32, f32>, #layout_1x3>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    // expected-error @below {{copy block span (3) exceeds DFB block count (2)}}
    %xf = ttl.copy %arg0, %cb : (tensor<1x3x!ttcore.tile<32x32, f32>, #layout_1x3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
