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

// Adjacent barriers on different NoCs are not redundant.
// CHECK-LABEL: func.func @different_noc_write_barriers_survive
// CHECK-DAG: %[[NOC0:.*]] = arith.constant 0 : i8
// CHECK-DAG: %[[NOC1:.*]] = arith.constant 1 : i8
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC0]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC1]])
func.func @different_noc_write_barriers_survive() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %noc0 = arith.constant 0 : i8
  %noc1 = arith.constant 1 : i8
  "ttkernel.noc_async_write_barrier"(%noc0) : (i8) -> ()
  "ttkernel.noc_async_write_barrier"(%noc1) : (i8) -> ()
  func.return
}

// -----

// CB -> Pipe copy (unicast): lowers to noc_async_write + semaphore inc
// CHECK-LABEL: func.func @copy_cb_to_pipe
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[TABLE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[SCRATCH]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1(%[[TABLE_PTR]]
// CHECK-NOT: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DST_ADDR]])
// CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[%[[DST_X]], %[[DST_Y]]], %[[DST_ADDR]], {{.*}}, noc %[[NOC]] : (i32, index, index, i32, i32, i8) -> ()
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_async_write_barrier
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
// CHECK: ttkernel.noc_inline_dw_write(core[{{.*}}, {{.*}}], %[[SCRATCH]], %[[DST_ADDR]], {{.*}}, noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]], {{.*}}, %[[NOC]])
// CHECK: %[[OLD:.*]] = memref.load %[[CTR]]
// CHECK: %[[NEW:.*]] = arith.addi %[[OLD]]
// CHECK: memref.store %[[NEW]], %[[CTR]]
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[DONE_SEM]])
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[DONE_PTR]], %[[NEW]])
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
// CHECK: ttkernel.experimental.semaphore_wait_min
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

// A wait in a zero-trip loop has no dynamic transfer-handle origin and is
// removed without rejecting the enclosing function.
// CHECK-LABEL: func.func @zero_trip_receive_wait
// CHECK-NOT: ttl.wait
// CHECK-NOT: ttl.pipe_transfer
// CHECK-NOT: unrealized_conversion_cast
func.func @zero_trip_receive_wait() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %p, %recv
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  scf.for %iter = %zero to %zero step %one iter_args(%xf_arg = %xf)
      -> (!ttl.transfer_handle) {
    ttl.wait %xf_arg : !ttl.transfer_handle
    scf.yield %xf_arg : !ttl.transfer_handle
  }
  func.return
}

// -----

// Explicit Pipe Transfer IR lowers through the same receiver-authored
// address publication, sender-ready wait, payload write, and completion wait.
// CHECK-LABEL: func.func @explicit_pipe_transfer_ir
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.experimental.semaphore_wait_min
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
// CHECK: %[[P0_DST_X:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[P0_DST_Y:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[P0_ADDR:.*]] = ttkernel.get_write_ptr
// CHECK: %[[SCRATCH0:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.noc_inline_dw_write(core[%[[P0_DST_X]], %[[P0_DST_Y]]], %[[SCRATCH0]], %[[P0_ADDR]], {{.*}}, noc {{.*}})
// CHECK: %[[P0_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// Second receive post publishes to p1 table slot and increments p1 ready sem.
// CHECK: %[[P1_DST_X:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[P1_DST_Y:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[P1_ADDR:.*]] = ttkernel.get_write_ptr
// CHECK: %[[SCRATCH1:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[P1_TABLE_ADDR:.*]] = arith.addi %[[SCRATCH1]], %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write(core[%[[P1_DST_X]], %[[P1_DST_Y]]], %[[P1_TABLE_ADDR]], %[[P1_ADDR]], {{.*}}, noc {{.*}})
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

// Three same-source transfers with mutually overlapping intervals need three
// distinct ready semaphores and address-table slots. The third slot must land
// at byte offset 8, confirming the table grows monotonically rather than
// aliasing back onto an earlier slot.
// CHECK-LABEL: func.func @same_source_three_pipes_use_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P2_READY_IDX:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[P1_TABLE_OFF:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[P2_TABLE_OFF:.*]] = arith.constant 8 : i32
// First post publishes to p0 table slot and increments p0 ready sem.
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// Second post publishes to p1 table slot (offset 4) and increments p1 ready sem.
// CHECK: arith.addi {{.*}}, %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// Third post publishes to p2 table slot (offset 8) and increments p2 ready sem.
// CHECK: arith.addi {{.*}}, %[[P2_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P2_READY_IDX]])
func.func @same_source_three_pipes_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv2 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send2 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.wait %post2 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Two same-source transfers with non-overlapping post-to-send intervals reuse
// the same ready semaphore and SRAM address-table slot.
// CHECK-LABEL: func.func @same_source_sequential_transfers_reuse_sync_state
// CHECK-NOT: arith.constant 4 : i32
// CHECK: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK-NOT: arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: arith.constant 4 : i32
// CHECK: return
func.func @same_source_sequential_transfers_reuse_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Repeated create_pipe ops with the same PipeKey share one transfer-allocation
// unit so repeated uses preserve the current per-pipe protocol state.
// CHECK-LABEL: func.func @same_pipe_key_transfer_creates_share_sync_state
// CHECK-NOT: arith.constant 4 : i32
// CHECK: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK-NOT: arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: arith.constant 4 : i32
// CHECK: return
func.func @same_pipe_key_transfer_creates_share_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Transfer intervals that are not bounded by dominance conservatively conflict
// with other same-source intervals, so they receive distinct ready state.
// CHECK-LABEL: func.func @same_source_control_flow_interval_uses_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P1_TABLE_OFF:.*]] = arith.constant 4 : i32
// CHECK: scf.if
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: arith.addi {{.*}}, %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: arith.addi {{.*}}, %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_async_write
func.func @same_source_control_flow_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  scf.if %cond {
    %then_token = ttl.pipe_transfer.post %transfer0, %recv0
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  } else {
    %else_token = ttl.pipe_transfer.post %transfer0, %recv0
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  }
  %send0 = ttl.pipe_transfer.send %transfer0, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %transfer1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Send operations in different control-flow regions do not define a bounded
// transfer interval, so later same-source transfers must not reuse the same
// sender-ready counter or SRAM address-table slot.
// CHECK-LABEL: func.func @same_source_control_flow_send_interval_uses_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P1_TABLE_OFF:.*]] = arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: scf.if
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: arith.addi {{.*}}, %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
func.func @same_source_control_flow_send_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer0, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  scf.if %cond {
    %then_send = ttl.pipe_transfer.send %transfer0, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %then_send : !ttl.transfer_handle<write>
  } else {
    %else_send = ttl.pipe_transfer.send %transfer0, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %else_send : !ttl.transfer_handle<write>
  }
  ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
  %transfer1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// A transfer with a receive post but no send has no bounded post-to-send
// interval, so it conservatively conflicts with later same-source transfers.
// CHECK-LABEL: func.func @same_source_missing_send_interval_uses_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P1_TABLE_OFF:.*]] = arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: arith.addi {{.*}}, %[[P1_TABLE_OFF]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
func.func @same_source_missing_send_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer0, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %transfer1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
  ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Overlapping transfers from different source nodes can use the same local
// ready semaphore id and the same SRAM address-table offset because both
// resources are physically local to each source node.
// CHECK-LABEL: func.func @different_sources_overlap_reuse_source_local_sync_state
// CHECK-NOT: arith.constant 4 : i32
// CHECK: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK-NOT: arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: arith.constant 4 : i32
// CHECK: return
func.func @different_sources_overlap_reuse_source_local_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Unused same-source pipe declarations do not allocate ready counters; the
// active transfer keeps its ready counter in a local hardware semaphore.
// CHECK-LABEL: func.func @same_source_pipes_keep_local_ready_counters_below_limit
// CHECK-DAG: %[[READY_IDX_BELOW:.*]] = arith.constant 1 : index
// CHECK: ttkernel.get_semaphore(%[[READY_IDX_BELOW]])
// CHECK: ttkernel.experimental.semaphore_wait
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

// PipeNet ids do not determine semaphore ids. A high id still receives compact
// completion and sender-ready allocations.
// CHECK-LABEL: func.func @high_pipe_net_id_uses_compact_semaphore_ids
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_PTR]]
func.func @high_pipe_net_id_uses_compact_semaphore_ids() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 14 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// A high PipeNet id does not force GlobalSemaphore-backed sender-ready state.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// CHECK-NOT: ttl.pipe_global_semaphore_count
// CHECK-LABEL: func.func @high_pipe_net_id_keeps_local_ready_counter
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[SCRATCH_ARG_IDX:.*]] = arith.constant 0 : index
// CHECK: %[[SCRATCH_POST:.*]] = ttkernel.get_common_arg_val(%[[SCRATCH_ARG_IDX]])
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set
// CHECK: %[[SCRATCH_SEND:.*]] = ttkernel.get_common_arg_val(%[[SCRATCH_ARG_IDX]])
// CHECK: ttkernel.reinterpret_cast{{.*}}(%[[SCRATCH_SEND]])
// CHECK: ttkernel.load_from_l1
func.func @high_pipe_net_id_keeps_local_ready_counter() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Disjoint receiver sets reuse one completion counter. Each source uses the
// next local semaphore id for its sender-ready counter.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// CHECK-NOT: ttl.pipe_global_semaphore_count
// CHECK-LABEL: func.func @disjoint_receivers_reuse_completion_counter
// CHECK-DAG: %[[ZERO_INDEX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.get_common_arg_val(%[[ZERO_INDEX]])
// CHECK: ttkernel.load_from_l1
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore(%[[ZERO_INDEX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
func.func @disjoint_receivers_reuse_completion_counter() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
  %side = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>) -> !ttl.transfer_handle<write>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %side, %recv1 : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %side : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
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
// CHECK: ttkernel.experimental.semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_multicast_addr({{.*}}, %[[DST_ADDR]]
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_async_write_barrier
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
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_END]], %[[DST_Y_END]]], end_xy[%[[DST_X_START]], %[[DST_Y_START]]], %[[DST_ADDR]], noc %[[NOC]])
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
// CHECK: ttkernel.experimental.semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_multicast_addr({{.*}}, %[[DST_ADDR]]
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[REMOTE_DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[REMOTE_DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: %[[LOCAL_DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[LOCAL_DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_async_write_barrier
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
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], noc %[[NOC]])
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
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], noc %[[NOC]])
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
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1
// CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}}, noc {{.*}} : (i32, index, index, i32, i32, i8) -> ()
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
// the SRAM address table, then wait on the transfer's completion counter.
// CHECK-LABEL: func.func @copy_pipe_to_cb_multicast
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.noc_inline_dw_write(core[{{.*}}, {{.*}}], %[[SCRATCH]], %[[DST_ADDR]], {{.*}}, noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]], {{.*}}, %[[NOC]])
// CHECK: %[[V:.*]] = memref.load %[[CTR]]
// CHECK: %[[NEW:.*]] = arith.addi %[[V]]
// CHECK: memref.store %[[NEW]], %[[CTR]]
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[DONE_SEM]])
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[DONE_PTR]], %[[NEW]])
// CHECK: ttkernel.cb_push_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.experimental.semaphore_wait(
func.func @copy_pipe_to_cb_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Tokens retain the dynamic post sequence when stored in a tensor and consumed
// in reverse order.
// CHECK-LABEL: func.func @reversed_pipe_tokens
// CHECK: %[[INITIAL:.*]] = tensor.empty() : tensor<2xi32>
// CHECK: %[[TOKENS:.*]] = scf.for {{.*}} iter_args(%[[CARRIED:.*]] = %[[INITIAL]]) -> (tensor<2xi32>) {
// CHECK: %[[OLD_SEQUENCE:.*]] = memref.load
// CHECK: %[[SEQUENCE:.*]] = arith.addi %[[OLD_SEQUENCE]]
// CHECK: %[[NEXT:.*]] = tensor.insert %[[SEQUENCE]] into %[[CARRIED]]
// CHECK: scf.yield %[[NEXT]] : tensor<2xi32>
// CHECK: scf.for
// CHECK: %[[REVERSE_INDEX:.*]] = arith.subi
// CHECK: %[[SELECTED_SEQUENCE:.*]] = tensor.extract %[[TOKENS]][%[[REVERSE_INDEX]]] : tensor<2xi32>
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[SELECTED_SEQUENCE]])
func.func @reversed_pipe_tokens()
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %initial = tensor.empty() : tensor<2x!ttl.pipe_token<net 0>>
  %tokens = scf.for %write_index = %zero to %two step %one
      iter_args(%carried = %initial) -> tensor<2x!ttl.pipe_token<net 0>> {
    %dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token = ttl.pipe_transfer.post %transfer, %dst
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    %send = ttl.pipe_transfer.send %transfer, %src_cb
        : (!ttl.pipe_transfer,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    %next = tensor.insert %token into %carried[%write_index]
        : tensor<2x!ttl.pipe_token<net 0>>
    scf.yield %next : tensor<2x!ttl.pipe_token<net 0>>
  }
  scf.for %read_index = %zero to %two step %one {
    %reverse_index = arith.subi %one, %read_index : index
    %token = tensor.extract %tokens[%reverse_index]
        : tensor<2x!ttl.pipe_token<net 0>>
    ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  }
  func.return
}

// -----

// A control-flow merge preserves the sequence produced by the executed post.
// CHECK-LABEL: func.func @selected_pipe_token
// CHECK: %[[SELECTED:.*]] = scf.if {{.*}} -> (i32) {
// CHECK: %[[THEN_SEQUENCE:.*]] = arith.addi
// CHECK: scf.yield %[[THEN_SEQUENCE]] : i32
// CHECK: } else {
// CHECK: %[[ELSE_SEQUENCE:.*]] = arith.addi
// CHECK: scf.yield %[[ELSE_SEQUENCE]] : i32
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[SELECTED]])
func.func @selected_pipe_token(%condition: i1)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %token = scf.if %condition -> (!ttl.pipe_token<net 0>) {
    %dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %then_token = ttl.pipe_transfer.post %transfer, %dst
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    scf.yield %then_token : !ttl.pipe_token<net 0>
  } else {
    %dst = ttl.cb_reserve %dst_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %else_token = ttl.pipe_transfer.post %transfer, %dst
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    scf.yield %else_token : !ttl.pipe_token<net 0>
  }
  %send = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// Reusing a token waits for the same completion threshold.
// CHECK-LABEL: func.func @wait_twice_for_pipe_token
// CHECK: %[[SEQUENCE:.*]] = arith.addi
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[SEQUENCE]])
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[SEQUENCE]])
func.func @wait_twice_for_pipe_token()
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      expectedReceivers = 1 : i64,
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %dst = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token = ttl.pipe_transfer.post %transfer, %dst
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  %send = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}

// -----

// A pipe function argument supplies the point-to-point transfer contract.
// CHECK-LABEL: func.func @pipe_block_argument
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NOT: ttl.pipe_transfer
func.func @pipe_block_argument(
    %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %dst_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %handle = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %handle : !ttl.transfer_handle
  func.return
}

// -----

// A receive wait completes its exact posted phase. The receiver can post the
// next phase even though matching sends execute in another kernel thread.

// CHECK-LABEL: func.func @two_sequential_receiver_phases
// CHECK-DAG: %[[COMPLETION_INDEX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[SEQUENCE:.*]] = memref.alloca() : memref<1xi32>
// CHECK-NOT: memref.alloca
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[PREVIOUS0:.*]] = memref.load %[[SEQUENCE]][%[[COMPLETION_INDEX]]]
// CHECK-NEXT: %[[TOKEN0:.*]] = arith.addi %[[PREVIOUS0]], %[[ONE]]
// CHECK-NEXT: memref.store %[[TOKEN0]], %[[SEQUENCE]][%[[COMPLETION_INDEX]]]
// CHECK: %[[COMPLETION0:.*]] = ttkernel.get_semaphore(%[[COMPLETION_INDEX]])
// CHECK-NEXT: %[[COMPLETION_PTR0:.*]] = ttkernel.reinterpret_cast(%[[COMPLETION0]])
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min(%[[COMPLETION_PTR0]], %[[TOKEN0]])
// CHECK-NOT: memref.alloca
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[PREVIOUS1:.*]] = memref.load %[[SEQUENCE]][%[[COMPLETION_INDEX]]]
// CHECK-NEXT: %[[TOKEN1:.*]] = arith.addi %[[PREVIOUS1]], %[[ONE]]
// CHECK-NEXT: memref.store %[[TOKEN1]], %[[SEQUENCE]][%[[COMPLETION_INDEX]]]
// CHECK: %[[COMPLETION1:.*]] = ttkernel.get_semaphore(%[[COMPLETION_INDEX]])
// CHECK-NEXT: %[[COMPLETION_PTR1:.*]] = ttkernel.reinterpret_cast(%[[COMPLETION1]])
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min(%[[COMPLETION_PTR1]], %[[TOKEN1]])
// CHECK-NOT: memref.alloca
// CHECK-NOT: ttl.pipe_transfer
func.func @two_sequential_receiver_phases()
    attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %recv_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %reserve0 = ttl.cb_reserve %recv_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %receive0 = ttl.copy %pipe, %reserve0
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %receive0 : !ttl.transfer_handle
  ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %reserve1 = ttl.cb_reserve %recv_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %receive1 = ttl.copy %pipe, %reserve1
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %receive1 : !ttl.transfer_handle
  ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// CHECK-LABEL: func.func @two_sequential_sender_phases
// CHECK-DAG: %[[SEND_COMPLETION_INDEX:.*]] = arith.constant 0 : index
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: %[[SEND_COMPLETION0:.*]] = ttkernel.get_semaphore(%[[SEND_COMPLETION_INDEX]])
// CHECK: %[[SEND_COMPLETION_NOC0:.*]] = ttkernel.get_noc_addr({{.*}}, %[[SEND_COMPLETION0]]
// CHECK-NEXT: ttkernel.noc_semaphore_inc(%[[SEND_COMPLETION_NOC0]]
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: %[[SEND_COMPLETION1:.*]] = ttkernel.get_semaphore(%[[SEND_COMPLETION_INDEX]])
// CHECK: %[[SEND_COMPLETION_NOC1:.*]] = ttkernel.get_noc_addr({{.*}}, %[[SEND_COMPLETION1]]
// CHECK-NEXT: ttkernel.noc_semaphore_inc(%[[SEND_COMPLETION_NOC1]]
// CHECK-NOT: ttl.pipe_transfer
func.func @two_sequential_sender_phases()
    attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %send_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %send0 = ttl.copy %send_cb, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %send_cb, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  func.return
}
