// RUN: ttlang-opt --verify-diagnostics --split-input-file %s
// Summary: Invalid ttl.copy cases rejected by the CopyOp verifier.

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @tensor_to_tensor_invalid(%arg0: tensor<32x32xf32, #layout>, %arg1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{expects exactly one operand to be !ttl.cb}}
    %xf = ttl.copy %arg0, %arg1 : (tensor<32x32xf32, #layout>, tensor<32x32xf32, #layout>) -> !ttl.xf
    ttl.wait %xf : !ttl.xf
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
    ttl.wait %xf : !ttl.xf
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
    ttl.wait %xf : !ttl.xf
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
    ttl.wait %xf : !ttl.xf
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Copy without a corresponding wait is invalid in the MVP.
module {
  func.func @copy_without_wait_invalid(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    // expected-error @below {{expects transfer handle to be synchronized with ttl.wait}}
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    func.return
  }
}

// -----

// Wait without a corresponding copy is invalid.
module {
  func.func @wait_without_copy_invalid(%xf: !ttl.xf) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{expects operand to be the result of ttl.copy}}
    ttl.wait %xf : !ttl.xf
    func.return
  }
}

// -----

// Wait on a handle that is routed through a tensor container but does not come
// from ttl.copy is invalid. This exercises the container-aware verifier.
module {
  func.func @wait_from_container_without_copy_invalid(%xf: !ttl.xf) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c1) : tensor<?x!ttl.xf>
    %handles = tensor.insert %xf into %handles0[%c0] : tensor<?x!ttl.xf>
    %loaded = tensor.extract %handles[%c0] : tensor<?x!ttl.xf>
    // expected-error @below {{expects operand to be the result of ttl.copy}}
    ttl.wait %loaded : !ttl.xf
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Two-phase loops with fewer wait iterations than copy iterations is invalid.
// This exercises the verifier's loop iteration space comparison.
module {
  func.func @two_phase_loops_missing_wait_invalid(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c4) : tensor<?x!ttl.xf>

    %handles = scf.for %i = %c0 to %c4 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.xf> {
      // expected-error @below {{expects transfer handle to be synchronized with ttl.wait}}
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.xf>
      scf.yield %h2 : tensor<?x!ttl.xf>
    }

    // Wait loop covers only [0, 3) while copy loop covers [0, 4).
    scf.for %i = %c0 to %c3 step %c1 {
      %xf = tensor.extract %handles[%i] : tensor<?x!ttl.xf>
      ttl.wait %xf : !ttl.xf
    }
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Pipelined loop with a loop-carried handle but no wait in the loop body is
// invalid: it drops intermediate transfers without synchronization.
module {
  func.func @pipelined_loop_missing_wait_invalid(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index

    // expected-error @below {{expects transfer handle to be synchronized with ttl.wait}}
    %xf_init = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    %last = scf.for %i = %c0 to %c2 step %c1 iter_args(%prev = %xf_init) -> (!ttl.xf) {
      %xf_next = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
      // Intentionally missing: ttl.wait %prev
      scf.yield %xf_next : !ttl.xf
    }
    ttl.wait %last : !ttl.xf
    func.return
  }
}
