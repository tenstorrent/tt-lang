// Verifies ttl-insert-copy-wait: missing ttl.wait is inserted after
// ttl.copy ops whose result has no synchronization consumer.

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

// A wait-any synchronizes every possible receive origin through an SSA merge.

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
