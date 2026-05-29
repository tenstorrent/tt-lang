// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// CHECK-LABEL: func.func @if_src_lowering
// CHECK: ttkernel.my_logical_x_
// CHECK: ttkernel.my_logical_y_
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi eq
// CHECK: arith.andi
// CHECK: scf.if
// CHECK:   ttkernel.noc_async_write_barrier
// CHECK: }
func.func @if_src_lowering() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    "ttkernel.noc_async_write_barrier"() : () -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: func.func @if_dst_lowering
// CHECK: ttkernel.my_logical_x_
// CHECK: ttkernel.my_logical_y_
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi sle
// CHECK: arith.cmpi sge
// CHECK: arith.cmpi sle
// CHECK: arith.andi
// CHECK: arith.andi
// CHECK: arith.andi
// CHECK: scf.if
// CHECK:   ttkernel.noc_async_read_barrier
// CHECK: }
func.func @if_dst_lowering() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    "ttkernel.noc_async_read_barrier"() : () -> ()
  }
  func.return
}

// -----

// CB -> Pipe copy (unicast): lowers to noc_async_write + semaphore inc
// CHECK-LABEL: func.func @copy_cb_to_pipe
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[MAILBOX_SEM]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1(%[[MAILBOX_PTR]]
// CHECK-NOT: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DST_ADDR]])
// CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[%[[DST_X]], %[[DST_Y]]], %[[DST_ADDR]], {{.*}} : (i32, index, index, i32, i32) -> ()
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[DONE_NOC]], {{.*}}, %[[NOC]])
func.func @copy_cb_to_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Pipe -> DFB copy (unicast receiver): publish the reserved destination
// address, then wait for sender completion.
// CHECK-LABEL: func.func @copy_pipe_to_cb
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[STAGING_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[STAGING_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[STAGING_SEM]])
// CHECK: ttkernel.store_to_l1(%[[DST_ADDR]], %[[STAGING_PTR]]
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[MAILBOX_SEM]], %[[NOC]])
// CHECK: ttkernel.remote_sram_write_u32(%[[STAGING_SEM]], %[[MAILBOX_NOC]], %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]], {{.*}}, %[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[DONE_SEM]])
// CHECK: %[[OLD:.*]] = memref.load %[[CTR]]
// CHECK: %[[NEW:.*]] = arith.addi %[[OLD]]
// CHECK: memref.store %[[NEW]], %[[CTR]]
// CHECK: ttkernel.experimental::semaphore_wait_min(%[[DONE_PTR]], %[[NEW]])
// CHECK: ttkernel.cb_push_back(%[[DST_DFB]]
func.func @copy_pipe_to_cb() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Two pipes in the same PipeNet with the same source need distinct ready
// semaphores and mailbox words, otherwise posts for one pipe can satisfy the
// other pipe's send.
// CHECK-LABEL: func.func @same_source_two_pipes_use_distinct_rendezvous_state
// CHECK-DAG: %[[STAGING_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P0_MAILBOX_IDX:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[P1_MAILBOX_IDX:.*]] = arith.constant 5 : index
// First receive post publishes to p0 mailbox and increments p0 ready sem.
// CHECK: ttkernel.get_semaphore(%[[STAGING_IDX]])
// CHECK: %[[P0_MAILBOX:.*]] = ttkernel.get_semaphore(%[[P0_MAILBOX_IDX]])
// CHECK: %[[P0_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// Second receive post publishes to p1 mailbox and increments p1 ready sem.
// CHECK: ttkernel.get_semaphore(%[[STAGING_IDX]])
// CHECK: %[[P1_MAILBOX:.*]] = ttkernel.get_semaphore(%[[P1_MAILBOX_IDX]])
// CHECK: %[[P1_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// First send waits on p0 ready sem and reads p0 mailbox.
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[P0_MAILBOX_IDX]])
// Second send waits on p1 ready sem and reads p1 mailbox.
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[P1_MAILBOX_IDX]])
func.func @same_source_two_pipes_use_distinct_rendezvous_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// CB -> Pipe (multicast, non-loopback): sender waits for all receivers to
// publish a common multicast destination address, writes payload with multicast,
// and inc_multicast signals every receiver's recvSem.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_multicast_addr({{.*}}, %[[DST_ADDR]]
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_semaphore_set_multicast
func.func @copy_cb_to_pipe_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// NOC1 multicast lowering reverses the translated destination rectangle before
// constructing tt-metal multicast transactions and semaphore addresses.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast_noc1
// CHECK: %[[NOC:.*]] = arith.constant 1 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_END]], %[[DST_Y_END]]], end_xy[%[[DST_X_START]], %[[DST_Y_START]]], %[[DST_ADDR]], %[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_END]], %[[DST_Y_END]], %[[DST_X_START]], %[[DST_Y_START]], %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
func.func @copy_cb_to_pipe_multicast_noc1() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc>, "ttl.noc_index" = 1 : i64 } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// CB -> Pipe (multicast loopback): payload writes use multicast with the
// receiver-published common destination address. Signaling splits into
// inc_multicast to remote receivers + local noc_semaphore_inc on self.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast_loopback
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_multicast_addr({{.*}}, %[[DST_ADDR]]
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[REMOTE_DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[REMOTE_DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: %[[LOCAL_DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[LOCAL_DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_semaphore_set_multicast
func.func @copy_cb_to_pipe_multicast_loopback() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Pipe -> DFB (multicast receiver): per-PipeNet counter ++, wait_min on
// recvSem. With one pipe the counter walks 0->1; with N overlapping
// pipes a receiver walks 1..N.
// CHECK-LABEL: func.func @copy_pipe_to_cb_multicast
// CHECK: %[[NOC:.*]] = arith.constant {{.*}} : i8
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[STAGING_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[STAGING_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[STAGING_SEM]])
// CHECK: ttkernel.store_to_l1(%[[DST_ADDR]], %[[STAGING_PTR]]
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[MAILBOX_SEM]], %[[NOC]])
// CHECK: ttkernel.remote_sram_write_u32(%[[STAGING_SEM]], %[[MAILBOX_NOC]], %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]], {{.*}}, %[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[DONE_SEM]])
// CHECK: %[[V:.*]] = memref.load %[[CTR]]
// CHECK: %[[NEW:.*]] = arith.addi %[[V]]
// CHECK: memref.store %[[NEW]], %[[CTR]]
// CHECK: ttkernel.experimental::semaphore_wait_min(%[[DONE_PTR]], %[[NEW]])
// CHECK: ttkernel.cb_push_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.experimental::semaphore_wait(
func.func @copy_pipe_to_cb_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}
