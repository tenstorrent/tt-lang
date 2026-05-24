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
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[TABLE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[SCRATCH]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1(%[[TABLE_PTR]]
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
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[TABLE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SCRATCH]], %[[NOC]])
// CHECK: ttkernel.noc_inline_dw_write(%[[TABLE_NOC]], %[[DST_ADDR]], {{.*}}, %[[NOC]])
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

// Pipe values carried through region results still lower at the receive site.
// CHECK-LABEL: func.func @copy_loop_carried_pipe_to_cb
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK-NOT: ttl.pipe_transfer
// CHECK-NOT: unrealized_conversion_cast
func.func @copy_loop_carried_pipe_to_cb() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %loop_pipe = scf.for %iter = %zero to %one step %one iter_args(%pipe_arg = %p)
      -> (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    scf.yield %pipe_arg : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  }
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %loop_pipe, %recv
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Explicit Pipe Transfer IR lowers through the same receiver-authored
// address publication, sender-ready wait, payload write, and completion wait.
// CHECK-LABEL: func.func @explicit_pipe_transfer_ir
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK-NOT: ttl.pipe_transfer
// CHECK-NOT: unrealized_conversion_cast
func.func @explicit_pipe_transfer_ir() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer_init = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %transfer = scf.for %iter = %zero to %one step %one iter_args(%transfer_arg = %transfer_init)
      -> (!ttl.pipe_transfer) {
    scf.yield %transfer_arg : !ttl.pipe_transfer
  }
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Explicit receive post without a wait still removes the internal token
// materialization after lowering.
// CHECK-LABEL: func.func @explicit_pipe_transfer_receive_only
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttl.pipe_transfer
// CHECK-NOT: unrealized_conversion_cast
func.func @explicit_pipe_transfer_receive_only() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  func.return
}

// -----

// Two pipes in the same PipeNet with the same source need distinct ready
// semaphores and SRAM address-table slots, otherwise posts for one pipe can
// satisfy the other pipe's send.
// CHECK-LABEL: func.func @same_source_two_pipes_use_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P1_TABLE_OFF:.*]] = arith.constant 4 : i32
// First receive post publishes to p0 table slot and increments p0 ready sem.
// CHECK: %[[SCRATCH0:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SCRATCH0]], {{.*}})
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[P0_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// Second receive post publishes to p1 table slot and increments p1 ready sem.
// CHECK: %[[SCRATCH1:.*]] = ttkernel.get_common_arg_val
// CHECK: arith.addi %[[SCRATCH1]], %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[P1_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// First send waits on p0 ready sem and reads p0 table slot.
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.reinterpret_cast{{.*}}(%{{.*}})
// CHECK: ttkernel.load_from_l1
// Second send waits on p1 ready sem and reads p1 table slot.
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: arith.addi {{.*}}, %[[P1_TABLE_OFF]]
// CHECK: ttkernel.load_from_l1
func.func @same_source_two_pipes_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
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

// Same-source pipe counts below the local semaphore boundary keep ready
// counters in local hardware semaphores.
// CHECK-LABEL: func.func @same_source_pipes_keep_local_ready_counters_below_limit
// CHECK-DAG: %[[READY_IDX_BELOW:.*]] = arith.constant 1 : index
// CHECK: ttkernel.get_semaphore(%[[READY_IDX_BELOW]])
// CHECK: ttkernel.experimental::semaphore_wait
func.func @same_source_pipes_keep_local_ready_counters_below_limit() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  %p3 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0 : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
  %p4 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0 : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
  %p5 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0 : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
  %p6 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0 : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
  %p7 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0 : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
  %p8 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0 : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
  %p9 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0 : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
  %p10 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0 : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
  %p11 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0 : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
  %p12 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0 : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
  %p13 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0 : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p0, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Same-source pipe counts at the local semaphore boundary keep ready
// counters in local hardware semaphores.
// CHECK-LABEL: func.func @same_source_pipes_keep_local_ready_counters_at_limit
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[READY_PTR]]
func.func @same_source_pipes_keep_local_ready_counters_at_limit() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  %p3 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0 : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
  %p4 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0 : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
  %p5 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0 : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
  %p6 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0 : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
  %p7 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0 : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
  %p8 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0 : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
  %p9 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0 : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
  %p10 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0 : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
  %p11 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0 : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
  %p12 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0 : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
  %p13 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0 : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
  %p14 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0 : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p0, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Same-source pipe counts that exceed local semaphore capacity use
// GlobalSemaphore-backed ready counters passed after the SRAM scratch base in
// common runtime args.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 16 : i64
// CHECK-LABEL: func.func @same_source_pipes_use_global_ready_counters
// CHECK-DAG: %[[SCRATCH_ARG_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[READY_ARG_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[SCRATCH_POST:.*]] = ttkernel.get_common_arg_val(%[[SCRATCH_ARG_IDX]])
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[READY_POST:.*]] = ttkernel.get_common_arg_val(%[[READY_ARG_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_common_arg_val(%[[READY_ARG_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set
// CHECK: %[[SCRATCH_SEND:.*]] = ttkernel.get_common_arg_val(%[[SCRATCH_ARG_IDX]])
// CHECK: ttkernel.reinterpret_cast{{.*}}(%[[SCRATCH_SEND]])
// CHECK: ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_semaphore(%[[READY_ARG_IDX]])
func.func @same_source_pipes_use_global_ready_counters() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  %p3 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0 : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
  %p4 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0 : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
  %p5 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0 : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
  %p6 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0 : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
  %p7 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0 : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
  %p8 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0 : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
  %p9 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0 : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
  %p10 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0 : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
  %p11 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0 : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
  %p12 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0 : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
  %p13 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0 : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
  %p14 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0 : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
  %p15 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 0 : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p0, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// When any source-local pipe count exceeds local semaphore capacity, ready
// counters use GlobalSemaphore addresses while per-PipeNet completion counters
// remain local semaphores.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 17 : i64
// CHECK-LABEL: func.func @interleaved_pipenets_use_global_ready_and_local_completion
// CHECK-DAG: %[[SCRATCH_ARG_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[FIRST_READY_ARG_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_common_arg_val(%[[FIRST_READY_ARG_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_common_arg_val(%[[FIRST_READY_ARG_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.get_common_arg_val(%[[SCRATCH_ARG_IDX]])
// CHECK: ttkernel.load_from_l1
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
func.func @interleaved_pipenets_use_global_ready_and_local_completion() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  %p3 = ttl.create_pipe src(0, 0) dst(4, 0) to(4, 0) net 0 : !ttl.pipe<src(0, 0) dst(4, 0) to(4, 0) net 0>
  %p4 = ttl.create_pipe src(0, 0) dst(5, 0) to(5, 0) net 0 : !ttl.pipe<src(0, 0) dst(5, 0) to(5, 0) net 0>
  %p5 = ttl.create_pipe src(0, 0) dst(6, 0) to(6, 0) net 0 : !ttl.pipe<src(0, 0) dst(6, 0) to(6, 0) net 0>
  %p6 = ttl.create_pipe src(0, 0) dst(7, 0) to(7, 0) net 0 : !ttl.pipe<src(0, 0) dst(7, 0) to(7, 0) net 0>
  %p7 = ttl.create_pipe src(0, 0) dst(8, 0) to(8, 0) net 0 : !ttl.pipe<src(0, 0) dst(8, 0) to(8, 0) net 0>
  %p8 = ttl.create_pipe src(0, 0) dst(9, 0) to(9, 0) net 0 : !ttl.pipe<src(0, 0) dst(9, 0) to(9, 0) net 0>
  %p9 = ttl.create_pipe src(0, 0) dst(10, 0) to(10, 0) net 0 : !ttl.pipe<src(0, 0) dst(10, 0) to(10, 0) net 0>
  %p10 = ttl.create_pipe src(0, 0) dst(11, 0) to(11, 0) net 0 : !ttl.pipe<src(0, 0) dst(11, 0) to(11, 0) net 0>
  %p11 = ttl.create_pipe src(0, 0) dst(12, 0) to(12, 0) net 0 : !ttl.pipe<src(0, 0) dst(12, 0) to(12, 0) net 0>
  %p12 = ttl.create_pipe src(0, 0) dst(13, 0) to(13, 0) net 0 : !ttl.pipe<src(0, 0) dst(13, 0) to(13, 0) net 0>
  %p13 = ttl.create_pipe src(0, 0) dst(14, 0) to(14, 0) net 0 : !ttl.pipe<src(0, 0) dst(14, 0) to(14, 0) net 0>
  %p14 = ttl.create_pipe src(0, 0) dst(15, 0) to(15, 0) net 0 : !ttl.pipe<src(0, 0) dst(15, 0) to(15, 0) net 0>
  %p15 = ttl.create_pipe src(0, 0) dst(16, 0) to(16, 0) net 0 : !ttl.pipe<src(0, 0) dst(16, 0) to(16, 0) net 0>
  %side = ttl.create_pipe src(1, 0) dst(17, 0) to(17, 0) net 1 : !ttl.pipe<src(1, 0) dst(17, 0) to(17, 0) net 1>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %side, %recv1 : (!ttl.pipe<src(1, 0) dst(17, 0) to(17, 0) net 1>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %side : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(17, 0) to(17, 0) net 1>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// CB -> Pipe (multicast, non-loopback): sender waits for all receivers to
// publish a common multicast destination address, writes payload with multicast,
// and inc_multicast signals every receiver's recvSem.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
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
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
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

// Source-in-destination multicast uses the same receiver-authored SRAM address
// table as non-loopback multicast.
// CHECK-LABEL: func.func @loopback_multicast_aggregate_ready_counting
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}})
func.func @loopback_multicast_aggregate_ready_counting() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Non-loopback multicast publishes receiver-authored addresses through the
// SRAM address table and uses one aggregate ready count.
// CHECK-LABEL: func.func @non_loopback_multicast_sram_address_table
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}})
// CHECK-NOT: ttkernel.noc_async_write_multicast_loopback_src
func.func @non_loopback_multicast_sram_address_table() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Degenerate multicast metadata preserves aggregate ready counting when a
// slice-origin multicast covers one destination.
// CHECK-LABEL: func.func @degenerate_multicast_aggregate_ready_counting
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}} : (i32, index, index, i32, i32) -> ()
func.func @degenerate_multicast_aggregate_ready_counting() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {isCollective = true} : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Pipe -> DFB (multicast receiver): publish the destination address through
// the SRAM address table, then wait on the per-PipeNet counter.
// CHECK-LABEL: func.func @copy_pipe_to_cb_multicast
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[TABLE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SCRATCH]], %[[NOC]])
// CHECK: ttkernel.noc_inline_dw_write(%[[TABLE_NOC]], %[[DST_ADDR]], {{.*}}, %[[NOC]])
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
