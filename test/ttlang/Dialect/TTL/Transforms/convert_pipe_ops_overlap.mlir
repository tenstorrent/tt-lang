// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Issue #505: overlapping multicast destinations within one PipeNet.

//===----------------------------------------------------------------------===//
// Two receives in one function share a single counter; the counter walks
// 1, 2.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_two_receives_share_counter
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]

// First Pipe->CB receive:
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
  %xf1 = ttl.copy %p1, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  %xf2 = ttl.copy %p2, %cb : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> !ttl.transfer_handle
  ttl.wait %xf2 : !ttl.transfer_handle
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
  %xf0 = ttl.copy %p_net0, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle
  ttl.wait %xf0 : !ttl.transfer_handle
  %xf1 = ttl.copy %p_net1, %cb : (!ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle
  ttl.wait %xf1 : !ttl.transfer_handle
  func.return
}

// -----

//===----------------------------------------------------------------------===//
// Two senders to the SAME destination range get distinct slot offsets.
// Slot 0 writes at offset 0 (no addi); slot 1 writes at offset 4096
// (one f32 tile = 4096 bytes for this CB shape).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_distinct_slots
// CHECK: ttkernel.noc_async_write_multicast
// CHECK: arith.addi {{.*}}, %{{c4096|.*}} : index
// CHECK: ttkernel.noc_async_write_multicast
func.func @overlap_distinct_slots() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  %xf1 = ttl.copy %cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf1 : !ttl.transfer_handle<write>
  %xf2 = ttl.copy %cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf2 : !ttl.transfer_handle<write>
  // Receivers needed so PipeGraph sees both pipes.
  %xf3 = ttl.copy %p1, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> !ttl.transfer_handle
  ttl.wait %xf3 : !ttl.transfer_handle
  %xf4 = ttl.copy %p2, %cb : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> !ttl.transfer_handle
  ttl.wait %xf4 : !ttl.transfer_handle
  func.return
}

// -----

//===----------------------------------------------------------------------===//
// Slot assignment is order-independent: declaring src(2,0) before src(0,0)
// in program order must produce the same slot map (src(0,0) -> slot 0,
// src(2,0) -> slot 1) because `assignGatherSlotIndices` sorts pipes by
// (srcX, srcY, ...) before assigning. Program order only controls which
// multicast IR appears first; the slot offset still tracks the sorted
// assignment. So the first multicast in the output (the src(2,0) copy)
// carries the slot-1 offset (addi c4096) and the second has no offset.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_distinct_slots_reversed_order
// First multicast: slot 1, uses addi with c4096.
// CHECK: arith.addi %{{.*}}, %c4096 : index
// CHECK: ttkernel.noc_async_write_multicast
// Second multicast: slot 0, no addi with c4096 between get_write_ptr and mcast.
// CHECK: ttkernel.get_write_ptr
// CHECK-NOT: arith.addi %{{.*}}, %c4096 : index
// CHECK: ttkernel.noc_async_write_multicast
func.func @overlap_distinct_slots_reversed_order() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  // Reverse program order: p2's send comes first.
  %xf2 = ttl.copy %cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf2 : !ttl.transfer_handle<write>
  %xf1 = ttl.copy %cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf1 : !ttl.transfer_handle<write>
  // Receivers needed so PipeGraph sees both pipes.
  %xf3 = ttl.copy %p2, %cb : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> !ttl.transfer_handle
  ttl.wait %xf3 : !ttl.transfer_handle
  %xf4 = ttl.copy %p1, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>) -> !ttl.transfer_handle
  ttl.wait %xf4 : !ttl.transfer_handle
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
