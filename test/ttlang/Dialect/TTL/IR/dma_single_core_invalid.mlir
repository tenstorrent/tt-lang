// RUN: ttlang-opt --verify-diagnostics --split-input-file %s
// Summary: Invalid ttl.copy cases rejected by the CopyOp verifier.

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

// Wait without a corresponding copy is invalid.
module {
  func.func @wait_without_copy_invalid(%xf: !ttl.transfer_handle<read>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{expects operand to be the result of ttl.copy}}
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

// Wait on a handle that is routed through a tensor container but does not come
// from ttl.copy is invalid. This exercises the container-aware verifier.
module {
  func.func @wait_from_container_without_copy_invalid(%xf: !ttl.transfer_handle<read>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c1) : tensor<?x!ttl.transfer_handle<read>>
    %handles = tensor.insert %xf into %handles0[%c0] : tensor<?x!ttl.transfer_handle<read>>
    %loaded = tensor.extract %handles[%c0] : tensor<?x!ttl.transfer_handle<read>>
    // expected-error @below {{expects operand to be the result of ttl.copy}}
    ttl.wait %loaded : !ttl.transfer_handle<read>
    func.return
  }
}
