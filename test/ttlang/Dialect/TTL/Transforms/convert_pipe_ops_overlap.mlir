// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Issue #505: overlapping multicast destinations within one PipeNet.

//===----------------------------------------------------------------------===//
// Two receives in one function share a single counter; the counter walks
// 1, 2.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_two_receives_share_counter
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]

// First Pipe->DFB receive:
// CHECK: ttkernel.noc_semaphore_inc({{.*}})
// CHECK: %[[V1:.*]] = memref.load %[[CTR]]
// CHECK: %[[N1:.*]] = arith.addi %[[V1]]
// CHECK: memref.store %[[N1]], %[[CTR]]
// CHECK: ttkernel.experimental::semaphore_wait_min({{.*}}, %[[N1]])

// Second Pipe->CB receive uses the SAME counter:
// CHECK: ttkernel.noc_semaphore_inc({{.*}})
// CHECK: %[[V2:.*]] = memref.load %[[CTR]]
// CHECK: %[[N2:.*]] = arith.addi %[[V2]]
// CHECK: memref.store %[[N2]], %[[CTR]]
// CHECK: ttkernel.experimental::semaphore_wait_min({{.*}}, %[[N2]])
func.func @overlap_two_receives_share_counter() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %recv1 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 4>
  %recv2 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 4>
  func.return
}

// -----

//===----------------------------------------------------------------------===//
// Two PipeNets in one function get distinct counters.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @two_pipenets_two_counters
// CHECK: %[[CTR_A:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[CTR_B:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.load %[[CTR_A]]
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK: memref.load %[[CTR_B]]
// CHECK: ttkernel.experimental::semaphore_wait_min
func.func @two_pipenets_two_counters() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p_net0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p_net1 = ttl.create_pipe src(0, 1) dst(2, 0) to(2, 3) net 1 : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>
  %recv0 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf0 = ttl.copy %p_net0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf0 : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv1 = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf1 = ttl.copy %p_net1, %recv1 : (!ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

//===----------------------------------------------------------------------===//
// Two senders to the same destination range use receiver-published
// addresses. Each send reads the posted destination address from the
// sender-visible mailbox before issuing its multicast write.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_distinct_slots
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast
func.func @overlap_distinct_slots() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv2 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send2 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4>
  ttl.wait %post2 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4>
  func.return
}

// -----

//===----------------------------------------------------------------------===//
// Send program order is independent of the stable PipeGraph slot assignment:
// the sender still reads the destination address posted for the specific pipe.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_distinct_slots_reversed_order
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast
func.func @overlap_distinct_slots_reversed_order() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %recv2 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  // Reverse program order: p2's send runs before p1's send.
  %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send2 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post2 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 4>
  func.return
}

// -----

//===----------------------------------------------------------------------===//
// Loopback sender: data path uses noc_async_write_multicast_loopback_src
// (sender included). The signal path is split: noc_semaphore_inc_multicast
// to remote receivers + local noc_semaphore_inc on the sender's own
// recvSem (no inc_multicast loopback variant in tt-metal).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loopback_self_inc
// CHECK: ttkernel.noc_async_write_multicast_loopback_src
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.experimental::get_noc_multicast_addr
// CHECK: ttkernel.noc_semaphore_inc_multicast
// CHECK: ttkernel.experimental::convert_logical_x_to_translated
// CHECK: ttkernel.experimental::convert_logical_y_to_translated
// CHECK: ttkernel.get_noc_addr
// CHECK: ttkernel.noc_semaphore_inc(
func.func @loopback_self_inc() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}
