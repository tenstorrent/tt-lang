// Verifies ttl-insert-copy-wait: ttl.wait is inserted after copies without a
// guaranteed explicit completion wait.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-copy-wait))' --split-input-file | FileCheck %s

// Test 1: copy without wait, auto-insert wait after copy.

#layout0 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                       buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @copy_no_wait
// CHECK: %[[XF:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[XF]] : !ttl.transfer_handle<read>
// CHECK: return
func.func @copy_no_wait(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout0>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout0> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout0>
  %xf = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  func.return
}

// -----

// Test 2: copy with explicit wait, pass should not insert a second wait.

#layout1 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                       buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @copy_with_wait
// CHECK: %[[XF:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[XF]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @copy_with_wait(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout1>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout1> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout1>
  %xf = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}

// -----

// Test 3: write direction copy without wait.

#layout2 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                       buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @write_copy_no_wait
// CHECK: %[[XF:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[XF]] : !ttl.transfer_handle<write>
// CHECK: return
func.func @write_copy_no_wait(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout2>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout2> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout2>
  %xf = ttl.copy %cb, %slice : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout2>) -> !ttl.transfer_handle<write>
  func.return
}

// -----

// Test 4: multiple copies, one with wait and one without.

#layout3 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                       buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @mixed_copy_wait
// CHECK: %[[XF1:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[XF1]]
// CHECK: %[[XF2:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[XF2]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @mixed_copy_wait(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout3>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout3> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout3>
  %xf1 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf1 : !ttl.transfer_handle<read>
  %slice1 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout3> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout3>
  %xf2 = ttl.copy %slice1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout3>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  func.return
}

// -----

// Same-block waits complete each copy without changing their overlap.

#linear_layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                             buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @same_block_nonadjacent_waits
// CHECK: %[[XF1:.+]] = ttl.copy
// CHECK-NEXT: %[[SLICE2:.+]] = ttl.tensor_slice
// CHECK-NEXT: %[[XF2:.+]] = ttl.copy %[[SLICE2]]
// CHECK-NEXT: ttl.wait %[[XF1]]
// CHECK-NEXT: ttl.wait %[[XF2]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @same_block_nonadjacent_waits(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout>
  %xf1 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  %slice1 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout>
  %xf2 = ttl.copy %slice1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #linear_layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf1 : !ttl.transfer_handle<read>
  ttl.wait %xf2 : !ttl.transfer_handle<read>
  func.return
}

// -----

// Branch-local waits discard a partial local plan and use CFG analysis.

#partial_layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                              buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @discard_partial_local_plan
// CHECK: %[[XF1:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[XF1]]
// CHECK-NEXT: %[[SLICE2:.+]] = ttl.tensor_slice
// CHECK-NEXT: %[[XF2:.+]] = ttl.copy %[[SLICE2]]
// CHECK-NEXT: scf.if
// CHECK: ttl.wait %[[XF2]]
// CHECK: } else {
// CHECK-NEXT: ttl.wait %[[XF2]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @discard_partial_local_plan(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout>,
    %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice0 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout>
  %xf1 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  %slice1 = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout>
  %xf2 = ttl.copy %slice1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #partial_layout>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  scf.if %condition {
    ttl.wait %xf2 : !ttl.transfer_handle<read>
  } else {
    ttl.wait %xf2 : !ttl.transfer_handle<read>
  }
  func.return
}

// -----

// A copy defined before a conditional still requires an implicit wait when one
// branch replaces its handle.

#layout4 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                       buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @reassigned_copy_wait
// CHECK: %[[FIRST:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[FIRST]]
// CHECK-NEXT: %[[SELECTED:.+]] = scf.if
// CHECK: %[[SECOND:.+]] = ttl.copy
// CHECK-NEXT: scf.yield %[[SECOND]]
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[FIRST]]
// CHECK: ttl.wait %[[SELECTED]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @reassigned_copy_wait(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>, %condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice0 = ttl.tensor_slice %arg0[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>
      -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>
  %slice1 = ttl.tensor_slice %arg0[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>
      -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>
  %first = ttl.copy %slice0, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<read>
  %selected = scf.if %condition -> (!ttl.transfer_handle<read>) {
    %second = ttl.copy %slice1, %cb
        : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout4>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
    scf.yield %second : !ttl.transfer_handle<read>
  } else {
    scf.yield %first : !ttl.transfer_handle<read>
  }
  ttl.wait %selected : !ttl.transfer_handle<read>
  func.return
}

// -----

// A wait on the previous loop-carried handle does not complete the final
// request created by the loop body.

#layout5 = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                       buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @loop_backedge_wait
// CHECK: %[[FIRST:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[FIRST]]
// CHECK-NEXT: scf.for
// CHECK: %[[NEXT:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[NEXT]]
// CHECK-NEXT: ttl.wait
// CHECK-NEXT: scf.yield %[[NEXT]]
// CHECK: return
func.func @loop_backedge_wait(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout5>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %slice = ttl.tensor_slice %arg0[%c0, %c0]
      : tensor<1x1x!ttcore.tile<32x32, f32>, #layout5>
      -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout5>
  %first = ttl.copy %slice, %cb
      : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout5>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<read>
  %last = scf.for %index = %c0 to %c3 step %c1
      iter_args(%previous = %first) -> (!ttl.transfer_handle<read>) {
    %next = ttl.copy %slice, %cb
        : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout5>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<read>
    ttl.wait %previous : !ttl.transfer_handle<read>
    scf.yield %next : !ttl.transfer_handle<read>
  }
  func.return
}

// -----

// A conditional wait-any does not complete a request issued before the
// conditional when the other branch executes.

// CHECK-LABEL: func.func @conditional_wait_any
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[REQUEST]]
// CHECK-NEXT: scf.if
// CHECK: ttl.wait_any %[[REQUEST]]
// CHECK: return
func.func @conditional_wait_any(%condition: i1) {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  scf.if %condition {
    %start = arith.constant 0 : index
    %ready = ttl.wait_any %request start %start
        : (!ttl.receive_request, index) -> !ttl.ready_receive
  }
  func.return
}

// -----

// A multi-request wait-any selects one request and does not complete the other
// candidates.

// CHECK-LABEL: func.func @multi_request_wait_any
// CHECK: %[[REQUEST0:.+]] = ttl.copy
// CHECK: %[[REQUEST1:.+]] = ttl.copy
// CHECK-NEXT: %[[START:.+]] = arith.constant 0
// CHECK-NEXT: ttl.wait_any %[[REQUEST0]], %[[REQUEST1]] start %[[START]]
// CHECK-NEXT: %[[SEND:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[SEND]]
// CHECK-NEXT: ttl.wait %[[REQUEST0]]
// CHECK-NEXT: ttl.wait %[[REQUEST1]]
// CHECK: return
func.func @multi_request_wait_any() {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %dst0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request0 = ttl.copy %pipe0, %dst0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %dst1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  %send = ttl.copy %source, %pipe0
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  func.return
}

// -----

// Cleanup follows a conditional selection even when the selection is skipped.

// CHECK-LABEL: func.func @conditional_multi_request_wait_any
// CHECK: %[[REQUEST0:.+]] = ttl.copy
// CHECK: %[[REQUEST1:.+]] = ttl.copy
// CHECK-NOT: ttl.wait
// CHECK: scf.if
// CHECK: ttl.wait_any %[[REQUEST0]], %[[REQUEST1]]
// CHECK: }
// CHECK-NEXT: ttl.wait %[[REQUEST0]]
// CHECK-NEXT: ttl.wait %[[REQUEST1]]
// CHECK: return
func.func @conditional_multi_request_wait_any(%condition: i1) {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %dst0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request0 = ttl.copy %pipe0, %dst0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %dst1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  scf.if %condition {
    %start = arith.constant 0 : index
    %ready = ttl.wait_any %request0, %request1 start %start
        : (!ttl.receive_request, !ttl.receive_request, index)
        -> !ttl.ready_receive
  }
  func.return
}

// -----

// Cleanup follows the last selection that can observe a request's readiness.

// CHECK-LABEL: func.func @repeated_multi_request_wait_any
// CHECK: %[[REQUEST0:.+]] = ttl.copy
// CHECK: %[[REQUEST1:.+]] = ttl.copy
// CHECK-NOT: ttl.wait
// CHECK: ttl.wait_any %[[REQUEST0]], %[[REQUEST1]]
// CHECK-NOT: ttl.wait
// CHECK: ttl.wait_any %[[REQUEST0]], %[[REQUEST1]]
// CHECK-NEXT: ttl.wait %[[REQUEST0]]
// CHECK-NEXT: ttl.wait %[[REQUEST1]]
// CHECK: return
func.func @repeated_multi_request_wait_any() {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %dst0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request0 = ttl.copy %pipe0, %dst0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %dst1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %start0 = arith.constant 0 : index
  %ready0 = ttl.wait_any %request0, %request1 start %start0
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  %start1 = arith.constant 1 : index
  %ready1 = ttl.wait_any %request0, %request1 start %start1
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  func.return
}

// -----

// CFG block arguments retain cleanup after their readiness observation.

// CHECK-LABEL: func.func @block_argument_multi_request_wait_any
// CHECK: %[[REQUEST0:.+]] = ttl.copy
// CHECK: %[[REQUEST1:.+]] = ttl.copy
// CHECK-NOT: ttl.wait
// CHECK: cf.br ^[[SELECT:.+]](%[[REQUEST0]], %[[REQUEST1]]
// CHECK: ^[[SELECT]](%[[CANDIDATE0:.+]]: !ttl.receive_request, %[[CANDIDATE1:.+]]: !ttl.receive_request):
// CHECK: ttl.wait_any %[[CANDIDATE0]], %[[CANDIDATE1]]
// CHECK-NEXT: ttl.wait %[[CANDIDATE0]]
// CHECK-NEXT: ttl.wait %[[CANDIDATE1]]
// CHECK: return
func.func @block_argument_multi_request_wait_any() {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %dst0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request0 = ttl.copy %pipe0, %dst0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %dst1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.br ^select(%request0, %request1 : !ttl.receive_request,
                !ttl.receive_request)
^select(%candidate0: !ttl.receive_request,
        %candidate1: !ttl.receive_request):
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %candidate0, %candidate1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  func.return
}

// -----

// Multiple returns receive cleanup after selection on each continuation.

// CHECK-LABEL: func.func @multi_exit_multi_request_wait_any
// CHECK: %[[REQUEST0:.+]] = ttl.copy
// CHECK: %[[REQUEST1:.+]] = ttl.copy
// CHECK-NOT: ttl.wait
// CHECK: cf.cond_br
// CHECK: ^[[SELECT:.+]]:
// CHECK: ttl.wait_any %[[REQUEST0]], %[[REQUEST1]]
// CHECK-NEXT: ttl.wait %[[REQUEST0]]
// CHECK-NEXT: ttl.wait %[[REQUEST1]]
// CHECK-NEXT: return
// CHECK: ^[[SKIP:.+]]:
// CHECK-NEXT: ttl.wait %[[REQUEST0]]
// CHECK-NEXT: ttl.wait %[[REQUEST1]]
// CHECK-NEXT: return
func.func @multi_exit_multi_request_wait_any(%condition: i1) {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %dst0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request0 = ttl.copy %pipe0, %dst0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %dst1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.cond_br %condition, ^select, ^skip
^select:
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  func.return
^skip:
  func.return
}

// -----

// Branch-local receive requests remain handled by wait-any after an SSA merge.

// CHECK-LABEL: func.func @merged_receive_wait_any
// CHECK-COUNT-2: ttl.copy
// CHECK: ttl.wait_any
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @merged_receive_wait_any(%condition: i1) {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %request = scf.if %condition -> (!ttl.receive_request) {
    %dst = ttl.cb_reserve %landing
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %then_request = ttl.copy %pipe, %dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    scf.yield %then_request : !ttl.receive_request
  } else {
    %dst = ttl.cb_reserve %landing
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %else_request = ttl.copy %pipe, %dst
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    scf.yield %else_request : !ttl.receive_request
  }
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request start %start
      : (!ttl.receive_request, index) -> !ttl.ready_receive
  func.return
}

// -----

// A wait in a zero-trip loop does not complete an outside copy.

// CHECK-LABEL: func.func @zero_trip_loop_wait
// CHECK: %[[HANDLE:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[HANDLE]]
// CHECK-NEXT: scf.for
// CHECK: ttl.wait %[[HANDLE]]
// CHECK: return
func.func @zero_trip_loop_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %handle = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  scf.for %index = %c0 to %c0 step %c1 {
    ttl.wait %handle : !ttl.receive_request
  }
  func.return
}

// -----

// The waits in both alternatives jointly complete an outside copy.

// CHECK-LABEL: func.func @branch_union_wait
// CHECK: %[[HANDLE:.+]] = ttl.copy
// CHECK-NEXT: scf.if
// CHECK: ttl.wait %[[HANDLE]]
// CHECK: } else {
// CHECK-NEXT: ttl.wait %[[HANDLE]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @branch_union_wait(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %handle = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  scf.if %condition {
    ttl.wait %handle : !ttl.receive_request
  } else {
    ttl.wait %handle : !ttl.receive_request
  }
  func.return
}

// -----

// Completion coverage propagates through nested conditional alternatives.

// CHECK-LABEL: func.func @nested_branch_union_wait
// CHECK: %[[HANDLE:.+]] = ttl.copy
// CHECK-NEXT: scf.if
// CHECK-COUNT-3: ttl.wait %[[HANDLE]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @nested_branch_union_wait(%outer: i1, %inner: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %handle = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  scf.if %outer {
    scf.if %inner {
      ttl.wait %handle : !ttl.receive_request
    } else {
      ttl.wait %handle : !ttl.receive_request
    }
  } else {
    ttl.wait %handle : !ttl.receive_request
  }
  func.return
}

// -----

// Completion coverage joins waits in distinct CFG blocks.

// CHECK-LABEL: func.func @cfg_branch_union_wait
// CHECK: %[[HANDLE:.+]] = ttl.copy
// CHECK-NEXT: cf.cond_br
// CHECK-COUNT-2: ttl.wait %[[HANDLE]]
// CHECK-NOT: ttl.wait
// CHECK: return
func.func @cfg_branch_union_wait(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %handle = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.cond_br %condition, ^then, ^else
^then:
  ttl.wait %handle : !ttl.receive_request
  cf.br ^return
^else:
  ttl.wait %handle : !ttl.receive_request
  cf.br ^return
^return:
  func.return
}

// -----

// Selection cleanup follows producer work in successor blocks.

// CHECK-LABEL: func.func @wait_any_successor_producer
// CHECK: %[[REQUEST0:.+]] = ttl.copy
// CHECK: %[[REQUEST1:.+]] = ttl.copy
// CHECK: %[[SEND0:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[SEND0]]
// CHECK: ttl.wait_any %[[REQUEST0]], %[[REQUEST1]]
// CHECK-NEXT: cf.br
// CHECK: ^[[PROGRESS:.+]]:
// CHECK-NEXT: %[[SEND1:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[SEND1]]
// CHECK-NEXT: ttl.wait %[[REQUEST0]]
// CHECK-NEXT: ttl.wait %[[REQUEST1]]
// CHECK-NEXT: return
func.func @wait_any_successor_producer() {
  %landing0 = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %landing1 = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source0 = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source1 = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %dst0 = ttl.cb_reserve %landing0
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %dst1 = ttl.cb_reserve %landing1
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request0 = ttl.copy %pipe0, %dst0
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %request1 = ttl.copy %pipe1, %dst1
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %send0 = ttl.copy %source0, %pipe0
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %request0, %request1 start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  cf.br ^progress
^progress:
  %send1 = ttl.copy %source1, %pipe1
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  func.return
}

// -----

// An exact wait through a single-predecessor block argument completes the
// original request.

// CHECK-LABEL: func.func @block_argument_exact_wait
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: cf.br ^[[COMPLETE:.+]](%[[REQUEST]]
// CHECK: ^[[COMPLETE]](%[[CANDIDATE:.+]]: !ttl.receive_request):
// CHECK-NEXT: %[[SEND:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[SEND]]
// CHECK-NEXT: ttl.wait %[[CANDIDATE]]
// CHECK-NEXT: return
func.func @block_argument_exact_wait() {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.br ^complete(%request : !ttl.receive_request)
^complete(%candidate: !ttl.receive_request):
  %send = ttl.copy %source, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %candidate : !ttl.receive_request
  func.return
}

// -----

// A loop-carried replacement does not change which dynamic request the
// explicit wait completes.

// CHECK-LABEL: func.func @loop_carried_replacement_exact_wait
// CHECK: %[[SEED:.+]] = ttl.copy
// CHECK-NEXT: cf.br ^[[LOOP:.+]](%[[SEED]]
// CHECK: ^[[LOOP]](%[[CANDIDATE:.+]]: !ttl.receive_request):
// CHECK-NEXT: %[[SEND:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[SEND]]
// CHECK-NEXT: ttl.wait %[[CANDIDATE]]
// CHECK-NEXT: cf.cond_br
// CHECK: ^[[NEXT:.+]]:
// CHECK-NEXT: %[[NEXT_DST:.+]] = ttl.cb_reserve
// CHECK-NEXT: %[[NEXT_REQUEST:.+]] = ttl.copy
// CHECK-NEXT: cf.br ^[[LOOP]](%[[NEXT_REQUEST]]
func.func @loop_carried_replacement_exact_wait(%repeat: i1) {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %source = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %seed_dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %seed = ttl.copy %pipe, %seed_dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.br ^loop(%seed : !ttl.receive_request)
^loop(%candidate: !ttl.receive_request):
  %send = ttl.copy %source, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %candidate : !ttl.receive_request
  cf.cond_br %repeat, ^next, ^exit
^next:
  %next_dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %next_request = ttl.copy %pipe, %next_dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.br ^loop(%next_request : !ttl.receive_request)
^exit:
  func.return
}

// -----

// A wait on the final loop result does not complete requests from earlier
// loop iterations.

// CHECK-LABEL: func.func @final_loop_request_wait
// CHECK: ^[[LOOP:.+]](%[[INDEX:.+]]: index):
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[REQUEST]]
// CHECK-NEXT: %[[NEXT_INDEX:.+]] = arith.addi
// CHECK: cf.cond_br
// CHECK: ^[[EXIT:.+]](%[[FINAL_REQUEST:.+]]: !ttl.receive_request):
// CHECK-NEXT: ttl.wait %[[FINAL_REQUEST]]
// CHECK-NEXT: return
func.func @final_loop_request_wait() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %landing = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  cf.br ^loop(%c0 : index)
^loop(%index: index):
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %request = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  %next_index = arith.addi %index, %c1 : index
  %repeat = arith.cmpi ult, %next_index, %c2 : index
  cf.cond_br %repeat, ^loop(%next_index : index),
      ^exit(%request : !ttl.receive_request)
^exit(%final_request: !ttl.receive_request):
  ttl.wait %final_request : !ttl.receive_request
  func.return
}

// -----

// A block argument provides the cleanup handle for mutually exclusive
// branch-local requests.

// CHECK-LABEL: func.func @branch_local_block_argument_wait_any
// CHECK: %[[STATIC:.+]] = ttl.copy
// CHECK: cf.cond_br
// CHECK: ^[[LEFT:.+]]:
// CHECK: %[[LEFT_REQUEST:.+]] = ttl.copy
// CHECK-NEXT: cf.br ^[[SELECT:.+]](%[[LEFT_REQUEST]]
// CHECK: ^[[RIGHT:.+]]:
// CHECK: %[[RIGHT_REQUEST:.+]] = ttl.copy
// CHECK-NEXT: cf.br ^[[SELECT]](%[[RIGHT_REQUEST]]
// CHECK: ^[[SELECT]](%[[MERGED:.+]]: !ttl.receive_request):
// CHECK: ttl.wait_any %[[STATIC]], %[[MERGED]]
// CHECK: scf.if
// CHECK: ttl.wait %[[STATIC]]
// CHECK-NEXT: ttl.wait %[[MERGED]]
// CHECK-NEXT: return
func.func @branch_local_block_argument_wait_any(%condition: i1) {
  %static_landing = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %branch_landing = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source0 = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source1 = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source2 = ttl.bind_cb {cb_index = 4, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %pipe1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
  %pipe2 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 2
      : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 2>
  %static_dst = ttl.cb_reserve %static_landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %static_request = ttl.copy %pipe0, %static_dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.cond_br %condition, ^left, ^right
^left:
  %left_dst = ttl.cb_reserve %branch_landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %left_request = ttl.copy %pipe1, %left_dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.br ^select(%left_request : !ttl.receive_request)
^right:
  %right_dst = ttl.cb_reserve %branch_landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %right_request = ttl.copy %pipe2, %right_dst
      : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 2>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  cf.br ^select(%right_request : !ttl.receive_request)
^select(%merged_request: !ttl.receive_request):
  %send0 = ttl.copy %source0, %pipe0
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
         !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %start = arith.constant 0 : index
  %ready = ttl.wait_any %static_request, %merged_request start %start
      : (!ttl.receive_request, !ttl.receive_request, index)
      -> !ttl.ready_receive
  scf.if %condition {
    %send1 = ttl.copy %source1, %pipe1
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  } else {
    %send2 = ttl.copy %source2, %pipe2
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 2>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
  }
  func.return
}

// -----

// A nested source iteration completes a request issued by the destination
// iteration when both roles contain the same launch node.

// CHECK-LABEL: func.func @self_loop_foreach_wait
// CHECK: ttl.pipenet_foreach_dst
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: ttl.pipenet_foreach_src
// CHECK: %[[SEND:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[REQUEST]]
// CHECK-NEXT: ttl.wait %[[SEND]]
func.func @self_loop_foreach_wait() {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "net" pipes[
        <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0>
      ]>} {
  ^bb0(%selected_dst: !ttl.selected_pipe_dst):
    %request = ttl.copy %selected_dst, %dst
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "net" pipes[
          <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
           dstEndX = 0, dstEndY = 0>
        ]>} {
    ^bb0(%selected_src: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %selected_src
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %request : !ttl.receive_request
      ttl.wait %send : !ttl.transfer_handle<write>
    }
  }
  func.return
}

// -----

// A source iteration on a different node cannot complete the destination
// node's request, so an implicit wait remains necessary.

// CHECK-LABEL: func.func @disjoint_foreach_wait
// CHECK: ttl.pipenet_foreach_dst
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: ttl.pipenet_foreach_src
// CHECK: ttl.wait %[[REQUEST]]
// CHECK: }
// CHECK-NEXT: ttl.wait %[[REQUEST]]
func.func @disjoint_foreach_wait() {
  %landing = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %source = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %dst = ttl.cb_reserve %landing
      : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "net" pipes[
        <srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
         dstEndX = 0, dstEndY = 0>
      ]>} {
  ^bb0(%selected_dst: !ttl.selected_pipe_dst):
    %request = ttl.copy %selected_dst, %dst
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "net" pipes[
          <srcX = 1, srcY = 0, dstStartX = 0, dstStartY = 0,
           dstEndX = 0, dstEndY = 0>
        ]>} {
    ^bb0(%selected_src: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %selected_src
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %request : !ttl.receive_request
      ttl.wait %send : !ttl.transfer_handle<write>
    }
  }
  func.return
}

// -----

// A collective source region completes the request on the source node. The
// implicit wait after that region completes it on receiver-only nodes.

// CHECK-LABEL: func.func @partial_foreach_domain_wait
// CHECK: ttl.pipenet_foreach_dst
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: arith.constant
// CHECK-NEXT: ttl.pipenet_foreach_src
// CHECK: %[[SEND:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[REQUEST]]
// CHECK-NEXT: ttl.wait %[[SEND]]
// CHECK: }
// CHECK-NEXT: ttl.wait %[[REQUEST]]
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @partial_foreach_domain_wait() {
    %landing = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %source = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.cb_reserve %landing
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "net" pipes[
          <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
           dstEndX = 1, dstEndY = 0, isCollective = true>
        ]>} {
    ^bb0(%selected_dst: !ttl.selected_pipe_dst):
      %request = ttl.copy %selected_dst, %dst
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      %c0 = arith.constant 0 : index
      ttl.pipenet_foreach_src attributes {
          records = #ttl.pipenet_records<net 0 name "net" pipes[
            <srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
             dstEndX = 1, dstEndY = 0, isCollective = true>
          ]>} {
      ^bb0(%selected_src: !ttl.selected_pipe_src):
        %send = ttl.copy %source, %selected_src
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %request : !ttl.receive_request
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// Static if_src and if_dst regions use the same launch-node completion proof
// as PipeNet iteration regions.

// CHECK-LABEL: func.func @self_loop_if_wait
// CHECK: ttl.if_dst
// CHECK: %[[REQUEST:.+]] = ttl.copy
// CHECK-NEXT: ttl.if_src
// CHECK: %[[SEND:.+]] = ttl.copy
// CHECK-NEXT: ttl.wait %[[REQUEST]]
// CHECK-NEXT: ttl.wait %[[SEND]]
// CHECK-NOT: ttl.wait %[[REQUEST]]
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @self_loop_if_wait() {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %landing = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %source = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst = ttl.cb_reserve %landing
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %request = ttl.copy %pipe, %dst
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %request : !ttl.receive_request
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}
