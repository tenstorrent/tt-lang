// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})' | FileCheck %s

// Summary: Verifies receiver-published synchronization for overlapping
// collective destinations and loopback collectives.

// Transfers that share physical receivers use distinct completion counters.
// CHECK-LABEL: func.func @overlap_two_receives_use_distinct_completion
// CHECK-DAG: %[[SEM_A:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SEM_B:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CTR_A:.*]] = memref.alloca() : memref<1xi32>
// CHECK-DAG: %[[CTR_B:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[COMP_A:.*]] = ttkernel.get_semaphore(%[[SEM_A]])
// CHECK: %[[WAIT_PTR1:.*]] = ttkernel.reinterpret_cast(%[[COMP_A]])
// CHECK: ttkernel.cb_reserve_back(%[[DFB]]
// CHECK: %[[WP1:.*]] = ttkernel.get_write_ptr(%[[DFB]])
// CHECK: ttkernel.noc_inline_dw_write({{.*}}, %[[WP1]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[V1:.*]] = memref.load %[[CTR_A]]
// CHECK: %[[N1:.*]] = arith.addi %[[V1]]
// CHECK: memref.store %[[N1]], %[[CTR_A]]
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[WAIT_PTR1]], %[[N1]])
// CHECK: ttkernel.cb_push_back(%[[DFB]]
// CHECK: %[[COMP_B:.*]] = ttkernel.get_semaphore(%[[SEM_B]])
// CHECK: %[[WAIT_PTR2:.*]] = ttkernel.reinterpret_cast(%[[COMP_B]])
// CHECK: ttkernel.cb_reserve_back(%[[DFB]]
// CHECK: %[[WP2:.*]] = ttkernel.get_write_ptr(%[[DFB]])
// CHECK: ttkernel.noc_inline_dw_write({{.*}}, %[[WP2]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[V2:.*]] = memref.load %[[CTR_B]]
// CHECK: %[[N2:.*]] = arith.addi %[[V2]]
// CHECK: memref.store %[[N2]], %[[CTR_B]]
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[WAIT_PTR2]], %[[N2]])
// CHECK: ttkernel.cb_push_back(%[[DFB]]
module attributes {ttl.launch_grid = array<i64: 3, 4>} {
  func.func @overlap_two_receives_use_distinct_completion()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 0, block_count = 4}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 4>
    %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
    %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0
        : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
    ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
      %recv1 = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post1 = ttl.copy %p1, %recv1
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post1 : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    }
    ttl.if_dst %p2 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0> {
      %recv2 = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post2 = ttl.copy %p2, %recv2
          : (!ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post2 : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 4>
    }
    func.return
  }

  func.func @overlap_two_receives_use_distinct_completion_senders()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
    %p2 = ttl.create_pipe src(2, 0) dst(1, 0) to(1, 3) net 0
        : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>
    ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
      %send1 = ttl.copy %src, %p1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    ttl.if_src %p2 : !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0> {
      %send2 = ttl.copy %src, %p2
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(2, 0) dst(1, 0) to(1, 3) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send2 : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Transfers with disjoint physical receivers may reuse one semaphore index.
// CHECK-LABEL: func.func @disjoint_receivers_reuse_completion
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK-NOT: memref.alloca
// CHECK: %[[VA:.*]] = memref.load %[[CTR]]
// CHECK: %[[NA:.*]] = arith.addi %[[VA]]
// CHECK: memref.store %[[NA]], %[[CTR]]
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[NA]])
// CHECK: %[[VB:.*]] = memref.load %[[CTR]]
// CHECK: %[[NB:.*]] = arith.addi %[[VB]]
// CHECK: memref.store %[[NB]], %[[CTR]]
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[NB]])
module attributes {ttl.launch_grid = array<i64: 3, 4>} {
  func.func @disjoint_receivers_reuse_completion()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
    %p1 = ttl.create_pipe src(0, 1) dst(2, 0) to(2, 3) net 1
        : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>
    ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
      %recv0 = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post0 = ttl.copy %p0, %recv0
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post0 : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_dst %p1 : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1> {
      %recv1 = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post1 = ttl.copy %p1, %recv1
          : (!ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post1 : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }

  func.func @disjoint_receivers_reuse_completion_senders()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
    %p1 = ttl.create_pipe src(0, 1) dst(2, 0) to(2, 3) net 1
        : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>
    ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
      %send0 = ttl.copy %src, %p0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
    }
    ttl.if_src %p1 : !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1> {
      %send1 = ttl.copy %src, %p1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 1) dst(2, 0) to(2, 3) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A loopback multicast signals remote receivers with a multicast increment and
// signals its local receiver with a separate point-to-point increment.
// CHECK-LABEL: func.func @loopback_self_inc
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[REMOTE_DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: %[[LOCAL_DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_multicast_addr({{.*}}, %[[DST_ADDR]]
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[REMOTE_DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[LOCAL_DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
module attributes {ttl.launch_grid = array<i64: 1, 4>} {
  func.func @loopback_self_inc()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
    ttl.if_src %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
      %send = ttl.copy %src, %p
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @loopback_self_inc_receiver()
      attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %dst = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
    ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
      %recv = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %p, %recv
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }
}
