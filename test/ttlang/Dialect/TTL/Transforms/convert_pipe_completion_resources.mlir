// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s --check-prefixes=CHECK,LOCAL,GLOBAL

// Summary: Verifies transfer-specific receiver-completion resource lowering.

// Two transfers in one PipeNet share a receiver but can complete in either
// order. Distinct completion counters ensure each wait observes only the send
// associated with its receive token.

// CHECK: module attributes
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 3 : i64
// CHECK-LABEL: func.func @independently_ordered_completions
// CHECK-DAG: %[[COMPLETION_A_INDEX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[COMPLETION_B_INDEX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[READY_INDEX:.*]] = arith.constant 2 : index
// CHECK: %[[SEQUENCE_A:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[SEQUENCE_B:.*]] = memref.alloca() : memref<1xi32>
// CHECK-NOT: memref.alloca

// The first post produces its sequence from completion counter A.
// CHECK: scf.if
// CHECK: %[[READY_A:.*]] = ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK: %[[READY_A_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[READY_A]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[READY_A_NOC]]
// CHECK: %[[PREVIOUS_A:.*]] = memref.load %[[SEQUENCE_A]]
// CHECK: %[[TOKEN_A:.*]] = arith.addi %[[PREVIOUS_A]]
// CHECK: memref.store %[[TOKEN_A]], %[[SEQUENCE_A]]

// The second post and its wait use a separate sequence and counter B.
// CHECK: scf.if
// CHECK: %[[READY_B:.*]] = ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK: %[[READY_B_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[READY_B]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[READY_B_NOC]]
// CHECK: %[[PREVIOUS_B:.*]] = memref.load %[[SEQUENCE_B]]
// CHECK: %[[TOKEN_B:.*]] = arith.addi %[[PREVIOUS_B]]
// CHECK: memref.store %[[TOKEN_B]], %[[SEQUENCE_B]]
// CHECK-NOT: ttkernel.get_semaphore(%[[COMPLETION_A_INDEX]])
// CHECK: %[[WAIT_B_SEM:.*]] = ttkernel.get_semaphore(%[[COMPLETION_B_INDEX]])
// CHECK: %[[WAIT_B_PTR:.*]] = ttkernel.reinterpret_cast(%[[WAIT_B_SEM]])
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[WAIT_B_PTR]], %[[TOKEN_B]])

// Send B increments only counter B.
// CHECK: scf.if
// CHECK: ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK-NOT: ttkernel.get_semaphore(%[[COMPLETION_A_INDEX]])
// CHECK: %[[SEND_B_SEM:.*]] = ttkernel.get_semaphore(%[[COMPLETION_B_INDEX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, %[[SEND_B_SEM]]
// CHECK: ttkernel.noc_semaphore_inc

// Send A increments only counter A.
// CHECK: scf.if
// CHECK: ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK-NOT: ttkernel.get_semaphore(%[[COMPLETION_B_INDEX]])
// CHECK: %[[SEND_A_SEM:.*]] = ttkernel.get_semaphore(%[[COMPLETION_A_INDEX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, %[[SEND_A_SEM]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: return

module attributes {ttl.launch_grid = [3 : i64, 1 : i64]} {
  func.func @independently_ordered_completions()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe_a = ttl.create_pipe src(2, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(2, 0) dst(2, 0) to(2, 0) net 0>
    %pipe_b = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    ttl.if_dst %pipe_a : !ttl.pipe<src(2, 0) dst(2, 0) to(2, 0) net 0> {
      %reserve_a = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive_a = ttl.copy %pipe_a, %reserve_a
          : (!ttl.pipe<src(2, 0) dst(2, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    ttl.if_dst %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0> {
      %reserve_b = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive_b = ttl.copy %pipe_b, %reserve_b
          : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %receive_b : !ttl.transfer_handle
    }
    ttl.if_src %pipe_b : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0> {
      %send_b = ttl.copy %send_cb, %pipe_b
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_b : !ttl.transfer_handle<write>
    }
    ttl.if_src %pipe_a : !ttl.pipe<src(2, 0) dst(2, 0) to(2, 0) net 0> {
      %send_a = ttl.copy %send_cb, %pipe_a
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(2, 0) dst(2, 0) to(2, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_a : !ttl.transfer_handle<write>
    }
    func.return
  }
}


// -----

// Fifteen overlapping source-side rendezvous intervals use the fifteen local
// sender-ready semaphore ids that remain after one reusable completion counter.

// LOCAL-LABEL: module attributes
// LOCAL-SAME: ttl.pipe_sync_semaphore_count = 16 : i64
// LOCAL-NOT: ttl.pipe_global_semaphore_count
// LOCAL-LABEL: func.func @local_ready_counters_fit_at_hardware_limit
// LOCAL: %[[LOCAL_TABLE_ADDR:.*]] = ttkernel.get_common_arg_val
// LOCAL: ttkernel.noc_inline_dw_write({{.*}}, %[[LOCAL_TABLE_ADDR]]
// LOCAL: %[[LOCAL_READY:.*]] = ttkernel.get_semaphore
// LOCAL: %[[LOCAL_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[LOCAL_READY]]
// LOCAL: ttkernel.noc_semaphore_inc(%[[LOCAL_READY_NOC]]
// LOCAL: return

module {
  func.func @local_ready_counters_fit_at_hardware_limit()
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %dst_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %local_pipe_0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %local_transfer_0 = ttl.pipe_transfer.create %local_pipe_0 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_0 = ttl.pipe_transfer.post %local_transfer_0, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %local_transfer_1 = ttl.pipe_transfer.create %local_pipe_1 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_1 = ttl.pipe_transfer.post %local_transfer_1, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
    %local_transfer_2 = ttl.pipe_transfer.create %local_pipe_2 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_2 = ttl.pipe_transfer.post %local_transfer_2, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_3 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0
        : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
    %local_transfer_3 = ttl.pipe_transfer.create %local_pipe_3 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_3 = ttl.pipe_transfer.post %local_transfer_3, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_4 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0
        : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
    %local_transfer_4 = ttl.pipe_transfer.create %local_pipe_4 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_4 = ttl.pipe_transfer.post %local_transfer_4, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_5 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0
        : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
    %local_transfer_5 = ttl.pipe_transfer.create %local_pipe_5 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_5 = ttl.pipe_transfer.post %local_transfer_5, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_6 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0
        : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
    %local_transfer_6 = ttl.pipe_transfer.create %local_pipe_6 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_6 = ttl.pipe_transfer.post %local_transfer_6, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_7 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0
        : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
    %local_transfer_7 = ttl.pipe_transfer.create %local_pipe_7 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_7 = ttl.pipe_transfer.post %local_transfer_7, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_8 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0
        : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
    %local_transfer_8 = ttl.pipe_transfer.create %local_pipe_8 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_8 = ttl.pipe_transfer.post %local_transfer_8, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_9 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0
        : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
    %local_transfer_9 = ttl.pipe_transfer.create %local_pipe_9 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_9 = ttl.pipe_transfer.post %local_transfer_9, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_10 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0
        : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
    %local_transfer_10 = ttl.pipe_transfer.create %local_pipe_10 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_10 = ttl.pipe_transfer.post %local_transfer_10, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_11 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0
        : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
    %local_transfer_11 = ttl.pipe_transfer.create %local_pipe_11 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_11 = ttl.pipe_transfer.post %local_transfer_11, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_12 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0
        : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
    %local_transfer_12 = ttl.pipe_transfer.create %local_pipe_12 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_12 = ttl.pipe_transfer.post %local_transfer_12, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_13 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0
        : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
    %local_transfer_13 = ttl.pipe_transfer.create %local_pipe_13 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_13 = ttl.pipe_transfer.post %local_transfer_13, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %local_pipe_14 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0
        : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
    %local_transfer_14 = ttl.pipe_transfer.create %local_pipe_14 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
        -> !ttl.pipe_transfer
    %local_token_14 = ttl.pipe_transfer.post %local_transfer_14, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    func.return
  }
}

// -----

// A sixteenth overlapping source-side interval moves every sender-ready
// counter to GlobalSemaphore-backed storage while completion remains local.

// GLOBAL-LABEL: module attributes
// GLOBAL-SAME: ttl.pipe_global_semaphore_count = 16 : i64
// GLOBAL-SAME: ttl.pipe_sync_semaphore_count = 1 : i64
// GLOBAL-LABEL: func.func @ready_counters_use_global_storage_over_local_limit
// GLOBAL: %[[TABLE_ADDR:.*]] = ttkernel.get_common_arg_val
// GLOBAL: ttkernel.noc_inline_dw_write({{.*}}, %[[TABLE_ADDR]]
// GLOBAL: %[[READY_ADDR:.*]] = ttkernel.get_common_arg_val
// GLOBAL: %[[READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, %[[READY_ADDR]]
// GLOBAL: ttkernel.noc_semaphore_inc(%[[READY_NOC]]
// GLOBAL: return

module {
  func.func @ready_counters_use_global_storage_over_local_limit()
      attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %dst_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %global_pipe_0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %global_transfer_0 = ttl.pipe_transfer.create %global_pipe_0 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_0 = ttl.pipe_transfer.post %global_transfer_0, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
    %global_transfer_1 = ttl.pipe_transfer.create %global_pipe_1 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_1 = ttl.pipe_transfer.post %global_transfer_1, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
    %global_transfer_2 = ttl.pipe_transfer.create %global_pipe_2 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_2 = ttl.pipe_transfer.post %global_transfer_2, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_3 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0
        : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
    %global_transfer_3 = ttl.pipe_transfer.create %global_pipe_3 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_3 = ttl.pipe_transfer.post %global_transfer_3, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_4 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0
        : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
    %global_transfer_4 = ttl.pipe_transfer.create %global_pipe_4 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_4 = ttl.pipe_transfer.post %global_transfer_4, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_5 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0
        : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
    %global_transfer_5 = ttl.pipe_transfer.create %global_pipe_5 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_5 = ttl.pipe_transfer.post %global_transfer_5, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_6 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0
        : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
    %global_transfer_6 = ttl.pipe_transfer.create %global_pipe_6 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_6 = ttl.pipe_transfer.post %global_transfer_6, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_7 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0
        : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
    %global_transfer_7 = ttl.pipe_transfer.create %global_pipe_7 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_7 = ttl.pipe_transfer.post %global_transfer_7, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_8 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0
        : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
    %global_transfer_8 = ttl.pipe_transfer.create %global_pipe_8 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_8 = ttl.pipe_transfer.post %global_transfer_8, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_9 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0
        : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
    %global_transfer_9 = ttl.pipe_transfer.create %global_pipe_9 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_9 = ttl.pipe_transfer.post %global_transfer_9, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_10 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0
        : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
    %global_transfer_10 = ttl.pipe_transfer.create %global_pipe_10 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_10 = ttl.pipe_transfer.post %global_transfer_10, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_11 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0
        : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
    %global_transfer_11 = ttl.pipe_transfer.create %global_pipe_11 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_11 = ttl.pipe_transfer.post %global_transfer_11, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_12 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0
        : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
    %global_transfer_12 = ttl.pipe_transfer.create %global_pipe_12 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_12 = ttl.pipe_transfer.post %global_transfer_12, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_13 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0
        : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
    %global_transfer_13 = ttl.pipe_transfer.create %global_pipe_13 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_13 = ttl.pipe_transfer.post %global_transfer_13, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_14 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0
        : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
    %global_transfer_14 = ttl.pipe_transfer.create %global_pipe_14 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_14 = ttl.pipe_transfer.post %global_transfer_14, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    %global_pipe_15 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 0
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>
    %global_transfer_15 = ttl.pipe_transfer.create %global_pipe_15 {
        expectedReceivers = 1 : i64,
        kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>
        -> !ttl.pipe_transfer
    %global_token_15 = ttl.pipe_transfer.post %global_transfer_15, %dst
        : (!ttl.pipe_transfer,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    func.return
  }
}
