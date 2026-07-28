// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Issue #505: overlapping multicast destinations within one PipeNet.

//===----------------------------------------------------------------------===//
// Two sequential collective receives use computed addresses for one DFB.
//===----------------------------------------------------------------------===//

// Transfers that share physical receivers use distinct completion counters
// and independent expected-count state.
// CHECK-LABEL: func.func @overlap_two_receives_use_distinct_completion
// CHECK: %[[CTR1:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR1]]
// CHECK: %[[CTR2:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR2]]
// CHECK: %[[DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: %[[V1:.*]] = memref.load %[[CTR1]]
// CHECK: %[[N1:.*]] = arith.addi %[[V1]]
// CHECK: memref.store %[[N1]], %[[CTR1]]
// CHECK: %[[WAIT_PTR1:.*]] = ttkernel.reinterpret_cast
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[WAIT_PTR1]], %[[N1]])
// CHECK: ttkernel.cb_push_back(%[[DFB]]
// CHECK: ttkernel.cb_reserve_back(%[[DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: %[[V2:.*]] = memref.load %[[CTR2]]
// CHECK: %[[N2:.*]] = arith.addi %[[V2]]
// CHECK: memref.store %[[N2]], %[[CTR2]]
// CHECK: %[[WAIT_PTR2:.*]] = ttkernel.reinterpret_cast
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[WAIT_PTR2]], %[[N2]])
// CHECK: ttkernel.cb_push_back(%[[DFB]]
module attributes {ttl.launch_grid = array<i64: 3, 4>} {
func.func @overlap_two_receives_use_distinct_completion() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
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

// Define both senders so the test exercises two complete transfers that share
// one receiver DFB.
func.func @overlap_two_receives_use_distinct_completion_senders() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %send1 = ttl.copy %src, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p2 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0> {
    %send2 = ttl.copy %src, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

//===----------------------------------------------------------------------===//
// Transfers with disjoint physical receivers may reuse one completion
// semaphore index because each receiver has independent semaphore storage.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @disjoint_receivers_reuse_completion
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK-NOT: memref.alloca
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: %[[VA:.*]] = memref.load %[[CTR]]
// CHECK: %[[NA:.*]] = arith.addi %[[VA]]
// CHECK: memref.store %[[NA]], %[[CTR]]
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[NA]])
// CHECK: %[[VB:.*]] = memref.load %[[CTR]]
// CHECK: %[[NB:.*]] = arith.addi %[[VB]]
// CHECK: memref.store %[[NB]], %[[CTR]]
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[NB]])
module attributes {ttl.launch_grid = array<i64: 3, 4>} {
func.func @disjoint_receivers_reuse_completion() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
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

// Define one sender for each PipeNet so both receiver counters belong to
// complete transfers.
func.func @disjoint_receivers_reuse_completion_senders() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p_net0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p_net1 = ttl.create_pipe src(0, 1) dst(2, 0) to(2, 3) net 1 : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>
  ttl.if_src %p_net0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %send0 = ttl.copy %src, %p_net0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p_net1 : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1> {
    %send1 = ttl.copy %src, %p_net1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

//===----------------------------------------------------------------------===//
// Two senders to the same destination range compute receiver DFB addresses.
// The second live receive uses the next physical DFB slot while the first
// receive remains live.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_distinct_slots
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: %[[SRC_ADDR1:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START1:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START1:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END1:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END1:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[BASE1:.*]] = ttkernel.get_common_arg_val
// CHECK-NOT: arith.remui
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR1]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START1]], %[[DST_Y_START1]]], end_xy[%[[DST_X_END1]], %[[DST_Y_END1]]], %[[BASE1]]
// CHECK: %[[SRC_ADDR2:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START2:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START2:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END2:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END2:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[BASE2:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DST_ADDR2:.*]] = arith.addi %[[BASE2]], %[[OFFSET]]
// CHECK-NOT: arith.remui
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR2]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START2]], %[[DST_Y_START2]]], end_xy[%[[DST_X_END2]], %[[DST_Y_END2]]], %[[DST_ADDR2]]
module attributes {ttl.launch_grid = array<i64: 3, 4>} {
func.func @overlap_distinct_slots() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post1 : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %recv2 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post2 : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p2 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0> {
    %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

//===----------------------------------------------------------------------===//
// Send program order is independent of receiver post order: the computed
// address uses the pipe-specific slot, so the first send below targets the
// second live receive slot.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @overlap_distinct_slots_reversed_order
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: %[[SRC_ADDR1:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START1:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START1:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END1:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END1:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[BASE1:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DST_ADDR1:.*]] = arith.addi %[[BASE1]], %[[OFFSET]]
// CHECK-NOT: arith.remui
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR1]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START1]], %[[DST_Y_START1]]], end_xy[%[[DST_X_END1]], %[[DST_Y_END1]]], %[[DST_ADDR1]]
// CHECK: %[[SRC_ADDR2:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START2:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START2:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END2:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END2:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[BASE2:.*]] = ttkernel.get_common_arg_val
// CHECK-NOT: arith.remui
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR2]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START2]], %[[DST_Y_START2]]], end_xy[%[[DST_X_END2]], %[[DST_Y_END2]]], %[[BASE2]]
module attributes {ttl.launch_grid = array<i64: 3, 4>} {
func.func @overlap_distinct_slots_reversed_order() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 4} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post1 : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %recv2 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post2 : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  // Reverse program order: p2's send runs before p1's send.
  ttl.if_src %p2 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0> {
    %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

//===----------------------------------------------------------------------===//
// A two-block receiver reserve advances the receiver slot cursor by two blocks.
// The following separate reserve must use base slot 2, not alias the grouped
// reserve's second block at slot 1.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @grouped_reserve_advances_following_slot
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[OFFSET2:.*]] = arith.constant 8192 : i32
// CHECK-DAG: %[[SRC_GROUP_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-DAG: %[[SRC_SINGLE_DFB:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: %[[SRC_ADDR1:.*]] = ttkernel.get_write_ptr(%[[SRC_GROUP_DFB]])
// CHECK: %[[BASE1:.*]] = ttkernel.get_common_arg_val
// CHECK-NOT: arith.remui
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR1]], {{.*}}, {{.*}}, start_xy[{{.*}}], end_xy[{{.*}}], %[[BASE1]]
// CHECK: %[[SRC_ADDR2:.*]] = ttkernel.get_write_ptr(%[[SRC_SINGLE_DFB]])
// CHECK: %[[BASE2:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DST_ADDR2:.*]] = arith.addi %[[BASE2]], %[[OFFSET2]]
// CHECK-NOT: arith.remui
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR2]], {{.*}}, {{.*}}, start_xy[{{.*}}], end_xy[{{.*}}], %[[DST_ADDR2]]
module attributes {ttl.launch_grid = array<i64: 4, 4>} {
func.func @grouped_reserve_advances_following_slot() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_group_cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 3} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 3>
  %src_single_cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %p2 = ttl.create_pipe src(3, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(3, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %recv_group = ttl.cb_reserve %dst_cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 3> -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv_group : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x2x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post1 : !ttl.transfer_handle
    ttl.cb_push %dst_cb {num_tiles = 2 : i64} : <[1, 1], !ttcore.tile<32x32, f32>, 3>
    %recv2 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 3> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(3, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post2 : !ttl.transfer_handle
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 3>
  }
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %send1 = ttl.copy %src_group_cb, %p1 : (!ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p2 : !ttl.pipe<src(3, 0) dst(1, 0) to(1, 3) net 0> {
    %send2 = ttl.copy %src_single_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(3, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

//===----------------------------------------------------------------------===//
// Loopback sender: payload writes use multicast with the computed common
// receiver DFB address. Completion signaling uses a multicast increment for
// remote receivers and a point-to-point increment for the local receiver.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loopback_self_inc
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-DAG: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK-DAG: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[REMOTE_DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[REMOTE_DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: %[[LOCAL_DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[LOCAL_DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
module attributes {ttl.launch_grid = array<i64: 1, 4>} {
func.func @loopback_self_inc() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
    %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
  }
  func.return
}

// Define the receiver half to prove the loopback multicast address sequence.
func.func @loopback_self_inc_receiver() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
    %recv = ttl.cb_reserve %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
    ttl.wait %post : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  func.return
}
}
