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

// A pipe function argument without call sites retains the point-to-point
// transfer contract encoded by its type.
// CHECK-LABEL: func.func @pipe_block_argument
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NOT: ttl.pipe_transfer
func.func @pipe_block_argument(
    %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %dst_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.cb_reserve %dst_cb
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %handle = ttl.copy %pipe, %dst
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  ttl.wait %handle : !ttl.receive_request
  func.return
}

// Define the sender half so the block-argument receiver belongs to a complete
// transfer.
func.func @pipe_block_argument_sender(
    %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %src_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %handle = ttl.copy %src_cb, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %handle : !ttl.transfer_handle<write>
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
// CHECK-DAG: ttkernel.noc_async_write %[[SRC_ADDR]], core[%[[DST_X]], %[[DST_Y]]], %[[DST_ADDR]], {{.*}}, noc %[[NOC]] : (i32, index, index, i32, i32, i8) -> ()
// CHECK-DAG: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_async_write_barrier
// CHECK: return
func.func @copy_cb_to_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// A non-pipe producer makes the receiver address dynamic, preserving the
// receiver-published protocol tested above.
func.func @copy_cb_to_pipe_receiver() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %local = ttl.cb_reserve %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv = ttl.cb_reserve %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  ttl.wait %post : !ttl.receive_request
  ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // Preserve the receiver-published protocol tested above.
  %local = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  ttl.wait %xf : !ttl.receive_request
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// Define the sender half so the test contains one complete transfer.
func.func @copy_pipe_to_cb_sender() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %send = ttl.copy %src, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  // Preserve the receiver-published protocol tested above.
  %local = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %loop_pipe = scf.for %iter = %zero to %one step %one iter_args(%pipe_arg = %p)
      -> (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    scf.yield %pipe_arg : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  }
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %loop_pipe, %recv
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  ttl.wait %xf : !ttl.receive_request
  ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// Define the sender half so the test contains one complete transfer.
func.func @copy_loop_carried_pipe_to_cb_sender() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %send = ttl.copy %src, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %p, %recv
      : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.receive_request
  scf.for %iter = %zero to %zero step %one iter_args(%xf_arg = %xf)
      -> (!ttl.receive_request) {
    ttl.wait %xf_arg : !ttl.receive_request
    scf.yield %xf_arg : !ttl.receive_request
  }
  func.return
}

// Define the sender half so the post belongs to a complete transfer even
// though its wait is unreachable.
func.func @zero_trip_receive_wait_sender()
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %cb, %p
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// A full-block point-to-point Pipe Transfer computes the receiver DFB slot
// address while preserving the sender-ready wait.
// CHECK-LABEL: func.func @explicit_pipe_transfer_ir
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: memref.alloca
// CHECK: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttl.pipe_transfer
// CHECK-NOT: unrealized_conversion_cast
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
func.func @explicit_pipe_transfer_ir() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer_init = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %transfer = scf.for %iter = %zero to %one step %one iter_args(%transfer_arg = %transfer_init)
      -> (!ttl.pipe_transfer) {
    scf.yield %transfer_arg : !ttl.pipe_transfer
  }
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token = ttl.pipe_transfer.post %transfer, %recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send = ttl.pipe_transfer.send %transfer, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// A proven single-writer receiver DFB stream replaces sender-ready rendezvous
// with sender-local capacity released by the receiver pop. Receiver
// block_count=2 verifies that a one-shot transfer uses its initial slot without
// allocating a sender-local slot counter.
// CHECK-LABEL: func.func @computed_address_capacity_protocol
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[CAPACITY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[INITIAL_CAPACITY:.*]] = arith.constant 2 : i32
// CHECK: %[[CAPACITY_SEM:.*]] = ttkernel.get_semaphore(%[[CAPACITY_IDX]])
// CHECK: %[[CAPACITY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[CAPACITY_SEM]])
// CHECK: ttkernel.noc_semaphore_set(%[[CAPACITY_PTR]], %[[INITIAL_CAPACITY]])
// CHECK: %[[RELEASE_ADDR:.*]] = ttkernel.get_noc_addr
// CHECK: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_wait_front
// CHECK: ttkernel.cb_pop_front
// CHECK: ttkernel.noc_semaphore_inc(%[[RELEASE_ADDR]]
// CHECK: ttkernel.noc_async_atomic_barrier
// CHECK: %[[ACQUIRE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%{{.*}})
// CHECK: %[[OLD_ACQUIRED:.*]] = memref.load %[[CAP_CTR:.*]][%{{.*}}]
// CHECK: %[[NEW_ACQUIRED:.*]] = arith.addi %[[OLD_ACQUIRED]]
// CHECK: memref.store %[[NEW_ACQUIRED]], %[[CAP_CTR]][%{{.*}}]
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[ACQUIRE_PTR]], %[[NEW_ACQUIRED]])
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: ttkernel.store_to_l1
// CHECK-NOT: ttkernel.experimental.semaphore_wait(
// CHECK-NOT: ttkernel.noc_semaphore_set
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.remui
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.experimental.semaphore_wait(
// CHECK-NOT: ttkernel.noc_semaphore_set
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @computed_address_capacity_protocol() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Capacity-proven transfers do not reserve sender-ready counters, while
// receiver-post transfers in the same module still do. Transfers to disjoint
// receivers reuse completion counter 0, so the remaining sender-ready and
// capacity counters use semaphore ids 1 and 2.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 3 : i64
// CHECK-LABEL: func.func @mixed_capacity_and_sender_ready_compact_resources
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1, 2>
// CHECK: %[[READY_SEM:.*]] = arith.constant 1 : index
// CHECK: %[[CAPACITY_IDX:.*]] = arith.constant 2 : index
// CHECK: %[[CAPACITY_SEM:.*]] = ttkernel.get_semaphore(%[[CAPACITY_IDX]])
// CHECK: %[[CAPACITY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[CAPACITY_SEM]])
// CHECK: ttkernel.noc_semaphore_set(%[[CAPACITY_PTR]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[MIXED_OLD:.*]] = memref.load %[[MIXED_CTR:.*]][%{{.*}}]
// CHECK: %[[MIXED_NEW:.*]] = arith.addi %[[MIXED_OLD]]
// CHECK: memref.store %[[MIXED_NEW]], %[[MIXED_CTR]][%{{.*}}]
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.get_semaphore(%[[READY_SEM]]
// CHECK: ttkernel.experimental.semaphore_wait
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
  func.func @mixed_capacity_and_sender_ready_compact_resources()
      attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %capacity_dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %ready_dst = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %capacity_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %ready_pipe = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 1
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1>
    %capacity_transfer = ttl.pipe_transfer.create %capacity_pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    %ready_transfer = ttl.pipe_transfer.create %ready_pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> -> !ttl.pipe_transfer
    ttl.if_dst %capacity_pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %capacity_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %capacity_transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %capacity_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready = ttl.cb_wait %capacity_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %capacity_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %capacity_pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %capacity_transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    ttl.if_dst %ready_pipe : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> {
      %recv = ttl.cb_reserve %ready_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %ready_transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 1>
      ttl.cb_push %ready_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %ready_pipe : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> {
      %send = ttl.pipe_transfer.send %ready_transfer, %src_cb
          : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Static receiver subviews compute the DFB block address and add the static
// byte offset inside the block.
// CHECK-LABEL: func.func @static_subview_pipe_transfer_computed_address
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[STATIC_OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: %[[BASE:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DST_ADDR:.*]] = arith.addi %[[BASE]], %[[STATIC_OFFSET]]
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
func.func @static_subview_pipe_transfer_computed_address() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %recv_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv = tensor.extract_slice %recv_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %token = ttl.pipe_transfer.post %transfer, %recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send = ttl.pipe_transfer.send %transfer, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Multiple incoming transfers to one receiver DFB compute distinct receiver
// slots but use receiver-post synchronization because a pop does not identify
// which sender's capacity to release.
// CHECK-LABEL: func.func @two_incoming_edges_one_dfb_compute_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK-DAG: %[[SLOT1_OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[BASE0:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[BASE0]]
// CHECK: %[[BASE1:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DST_ADDR1:.*]] = arith.addi %[[BASE1]], %[[SLOT1_OFFSET]]
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR1]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
func.func @two_incoming_edges_one_dfb_compute_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pA = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %pB = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
  %tA = ttl.pipe_transfer.create %pA {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %tB = ttl.pipe_transfer.create %pB {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> -> !ttl.pipe_transfer
  ttl.if_dst %pA : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %recvA = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %tokA = ttl.pipe_transfer.post %tA, %recvA
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %tokA : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_dst %pB : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
    %recvB = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %tokB = ttl.pipe_transfer.post %tB, %recvB
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
    ttl.pipe_transfer.wait %tokB : !ttl.pipe_token<net 1>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %pA : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %sendA = ttl.pipe_transfer.send %tA, %src_cb0
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %sendA : !ttl.transfer_handle<write>
  }
  ttl.if_src %pB : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> {
    %sendB = ttl.pipe_transfer.send %tB, %src_cb1
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %sendB : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Receiver-published-address transfers use a compact SRAM address table even
// when a computed-address transfer is live between them in the same source
// stream. The second published-address slot is byte offset 4, not offset 8.
// CHECK-LABEL: func.func @receiver_published_address_slots_ignore_computed_colors
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK-DAG: %[[SECOND_SLOT_OFFSET:.*]] = arith.constant 4 : i32
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: %[[SECOND_TABLE_ADDR:.*]] = arith.addi {{.*}}, %[[SECOND_SLOT_OFFSET]]
// CHECK: ttkernel.noc_inline_dw_write(core{{.*}}, %[[SECOND_TABLE_ADDR]]
module attributes {ttl.launch_grid = array<i64: 4, 1>} {
func.func @receiver_published_address_slots_ignore_computed_colors(%dynamic_idx: index)
    attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %published_a_dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %computed_dst = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %published_b_dst = ttl.bind_cb {cb_index = 3, block_count = 1} {dfb_id = 3 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %published_a_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %computed_pipe = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 1
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1>
  %published_b_pipe = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 2
      : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 2>
  %published_a_transfer = ttl.pipe_transfer.create %published_a_pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %computed_transfer = ttl.pipe_transfer.create %computed_pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> -> !ttl.pipe_transfer
  %published_b_transfer = ttl.pipe_transfer.create %published_b_pipe {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 2> -> !ttl.pipe_transfer
  ttl.if_dst %published_a_pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %published_a_full = ttl.cb_reserve %published_a_dst
        : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %published_a_recv = tensor.extract_slice %published_a_full[%dynamic_idx, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %published_a_token = ttl.pipe_transfer.post %published_a_transfer, %published_a_recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %published_a_token : !ttl.pipe_token<net 0>
    ttl.cb_push %published_a_dst : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %computed_pipe : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 1> {
    %computed_recv = ttl.cb_reserve %computed_dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %computed_token = ttl.pipe_transfer.post %computed_transfer, %computed_recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
    ttl.pipe_transfer.wait %computed_token : !ttl.pipe_token<net 1>
    ttl.cb_push %computed_dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %published_b_pipe : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 2> {
    %published_b_full = ttl.cb_reserve %published_b_dst
        : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %published_b_recv = tensor.extract_slice %published_b_full[%dynamic_idx, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %published_b_token = ttl.pipe_transfer.post %published_b_transfer, %published_b_recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 2>
    ttl.pipe_transfer.wait %published_b_token : !ttl.pipe_token<net 2>
    ttl.cb_push %published_b_dst : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %published_a_pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %published_a_send = ttl.pipe_transfer.send %published_a_transfer, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    %computed_send = ttl.pipe_transfer.send %computed_transfer, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    %published_b_send = ttl.pipe_transfer.send %published_b_transfer, %src_cb
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %published_a_send : !ttl.transfer_handle<write>
    ttl.wait %computed_send : !ttl.transfer_handle<write>
    ttl.wait %published_b_send : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Two collective senders into the same receiver DFB compute distinct multicast
// receiver slots from the reservation order proven by PipeGraph.
// CHECK-LABEL: func.func @two_multicast_edges_one_dfb_compute_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK-DAG: %[[SLOT1_OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[BASE0:.*]] = ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write_multicast_loopback_src({{.*}}, %[[BASE0]], {{.*}})
// CHECK: %[[BASE1:.*]] = ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: %[[DST_ADDR1:.*]] = arith.addi %[[BASE1]], %[[SLOT1_OFFSET]]
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write_multicast_loopback_src({{.*}}, %[[DST_ADDR1]], {{.*}})
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
func.func @two_multicast_edges_one_dfb_compute_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(1, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(1, 0) dst(0, 0) to(1, 0) net 0>
  %t0 = ttl.pipe_transfer.create %p0 {kind = #ttl.pipe_transfer_kind<collective>}
      : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %t1 = ttl.pipe_transfer.create %p1 {kind = #ttl.pipe_transfer_kind<collective>}
      : !ttl.pipe<src(1, 0) dst(0, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
    %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %tok0 = ttl.pipe_transfer.post %t0, %recv0
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %tok0 : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_dst %p1 : !ttl.pipe<src(1, 0) dst(0, 0) to(1, 0) net 0> {
    %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %tok1 = ttl.pipe_transfer.post %t1, %recv1
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %tok1 : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
    %send0 = ttl.pipe_transfer.send %t0, %src_cb0
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p1 : !ttl.pipe<src(1, 0) dst(0, 0) to(1, 0) net 0> {
    %send1 = ttl.pipe_transfer.send %t1, %src_cb1
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Reusing the same DFB index on independent receiver cores does not conflict.
// The receiver-core coordinate remains part of the incoming-edge identity, so
// both point-to-point transfers compute their receiver DFB slot addresses.
// CHECK-LABEL: func.func @independent_receivers_same_dfb_compute_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK: ttkernel.get_common_arg_val
// CHECK: ttkernel.get_common_arg_val
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 4, 1>} {
func.func @independent_receivers_same_dfb_compute_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %pipe1 = ttl.create_pipe src(1, 0) dst(3, 0) to(3, 0) net 1 : !ttl.pipe<src(1, 0) dst(3, 0) to(3, 0) net 1>
  %transfer0 = ttl.pipe_transfer.create %pipe0 {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %transfer1 = ttl.pipe_transfer.create %pipe1 {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(1, 0) dst(3, 0) to(3, 0) net 1> -> !ttl.pipe_transfer
  ttl.if_dst %pipe0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token0 = ttl.pipe_transfer.post %transfer0, %recv0
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %pipe1 : !ttl.pipe<src(1, 0) dst(3, 0) to(3, 0) net 1> {
    %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token1 = ttl.pipe_transfer.post %transfer1, %recv1
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
    ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 1>
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %pipe0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %send0 = ttl.pipe_transfer.send %transfer0, %src_cb0
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %pipe1 : !ttl.pipe<src(1, 0) dst(3, 0) to(3, 0) net 1> {
    %send1 = ttl.pipe_transfer.send %transfer1, %src_cb1
        : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
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
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  // Preserve receiver-published lowering while testing an unused post token.
  %local = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.pipe_token<net 0>
  func.return
}

// Define the sender half so the explicit post belongs to a complete transfer.
func.func @explicit_pipe_transfer_receive_only_sender() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %send = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
      -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  func.return
}

// -----

// Two pipes in the same PipeNet with the same source need distinct ready
// semaphores, otherwise posts for one pipe can satisfy the other pipe's send.
// CHECK-LABEL: func.func @same_source_two_pipes_use_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// First receive post increments p0 ready sem.
// CHECK: %[[P0_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// Second receive post increments p1 ready sem.
// CHECK: %[[P1_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// First send waits on p0 ready sem and computes the destination address.
// CHECK: %[[P0_SEND_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: %[[P0_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[P0_SEND_READY]])
// CHECK: ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait(%[[P0_READY_PTR]]
// CHECK: ttkernel.noc_async_write
// Second send waits on p1 ready sem and computes the destination address.
// CHECK: %[[P1_SEND_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: %[[P1_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[P1_SEND_READY]])
// CHECK: ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait(%[[P1_READY_PTR]]
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
func.func @same_source_two_pipes_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post0 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post1 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Three same-source transfers with mutually overlapping intervals need three
// distinct ready semaphores.
// CHECK-LABEL: func.func @same_source_three_pipes_use_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[P2_READY_IDX:.*]] = arith.constant 3 : index
// First post increments p0 ready sem.
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// Second post increments p1 ready sem.
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// Third post increments p2 ready sem.
// CHECK: ttkernel.get_semaphore(%[[P2_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 4, 1>} {
func.func @same_source_three_pipes_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post0 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post1 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %p2 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0> {
    %recv2_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv2 = tensor.extract_slice %recv2_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post2 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p2 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0> {
    %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send2 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Guarded same-source transfers with separate receiver regions use distinct
// ready semaphores.
// CHECK-LABEL: func.func @same_source_sequential_transfers_use_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
func.func @same_source_sequential_transfers_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post0 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_dst %p1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post1 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p1 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> {
    %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Distinct transfer definitions for one PipeKey have separate completion
// state. Since the first transfer completes before the second starts, interval
// coloring reuses one ready semaphore. The receiver sequence proves both DFB
// addresses without publication.
// CHECK-LABEL: func.func @same_pipe_key_nonoverlapping_transfers_reuse_storage
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[READY_IDX:.*]] = arith.constant 2 : index
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: return
func.func @same_pipe_key_nonoverlapping_transfers_reuse_storage() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.receive_request
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.receive_request
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Transfer intervals that are not bounded by dominance conservatively conflict
// with other same-source intervals, so they receive distinct ready state.
// CHECK-LABEL: func.func @same_source_control_flow_interval_uses_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK: %[[P0_POST_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: scf.if
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[P0_SEND_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: %[[P0_SEND_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[P0_SEND_READY]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[P0_SEND_PTR]]
// CHECK: %[[P1_POST_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[P1_SEND_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: %[[P1_SEND_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[P1_SEND_READY]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[P1_SEND_PTR]]
// CHECK: ttkernel.noc_async_write
func.func @same_source_control_flow_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
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
  %transfer1 = ttl.pipe_transfer.create %p1 {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Send operations in different control-flow regions do not define a bounded
// transfer interval, so later same-source transfers must not reuse the same
// sender-ready counter.
// CHECK-LABEL: func.func @same_source_control_flow_send_interval_uses_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[P0_SEND_READY:.*]] = ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: %[[P0_SEND_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[P0_SEND_READY]])
// CHECK: scf.if
// CHECK: ttkernel.experimental.semaphore_wait(%[[P0_SEND_PTR]]
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[P1_SEND_READY:.*]] = ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: %[[P1_SEND_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[P1_SEND_READY]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[P1_SEND_PTR]]
func.func @same_source_control_flow_send_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
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
  %transfer1 = ttl.pipe_transfer.create %p1 {kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Overlapping transfers from different source nodes can use the same local
// ready semaphore id because the resource is physically local to each source
// node.
// CHECK-LABEL: func.func @different_sources_overlap_reuse_source_local_sync_state
// CHECK: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
func.func @different_sources_overlap_reuse_source_local_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post0 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %p1 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0> {
    %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post1 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %p1 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0> {
    %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Unused same-source pipe declarations do not allocate ready counters; the
// active transfer keeps its ready counter in a local hardware semaphore.
// CHECK-LABEL: func.func @same_source_pipes_keep_local_ready_counters_below_limit
// CHECK-DAG: %[[READY_IDX_BELOW:.*]] = arith.constant 1 : index
// CHECK: ttkernel.get_semaphore(%[[READY_IDX_BELOW]])
// CHECK: ttkernel.experimental.semaphore_wait
func.func @same_source_pipes_keep_local_ready_counters_below_limit() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
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
  %post = ttl.copy %p0, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  %send = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.receive_request
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Sparse PipeNet ids do not reserve unused completion semaphores. One active
// transfer uses completion semaphore 0 and sender-ready semaphore 1.
// CHECK-LABEL: func.func @sparse_pipe_net_id_uses_compact_resources
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_PTR]]
func.func @sparse_pipe_net_id_uses_compact_resources() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 14 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 14>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.receive_request
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// A high PipeNet id does not force GlobalSemaphore allocation. Resource ids
// depend on active transfers rather than PipeNet numbering.
// CHECK-LABEL: module attributes
// CHECK-NOT: ttl.pipe_global_semaphore_count
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// CHECK-LABEL: func.func @high_pipe_net_id_uses_local_ready_counter
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
func.func @high_pipe_net_id_uses_local_ready_counter() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15> {
    %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15> {
    %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Transfers on disjoint receiver nodes reuse completion semaphore 0. Ready
// semaphore 1 is also reusable because each source core has independent local
// semaphore storage.
// CHECK-LABEL: module attributes
// CHECK-NOT: ttl.pipe_global_semaphore_count
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 2 : i64
// CHECK-LABEL: func.func @interleaved_pipenets_reuse_local_resources
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[SECOND_READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SECOND_READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore(%{{.*}})
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], {{.*}})
// CHECK: ttkernel.experimental.semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 3, 1>} {
func.func @interleaved_pipenets_reuse_local_resources() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
  %side = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  ttl.if_dst %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15> {
    %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post0 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_dst %side : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0> {
    %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
    %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
        : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
    %post1 = ttl.copy %side, %recv1 : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post1 : !ttl.receive_request
    ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15> {
    %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>) -> !ttl.transfer_handle<write>
    ttl.wait %send0 : !ttl.transfer_handle<write>
  }
  ttl.if_src %side : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0> {
    %send1 = ttl.copy %src_cb, %side : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send1 : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Sender and receiver callbacks create separate pipe values for one transfer.
// Their lowering must use the same sender-ready semaphore on every loop
// iteration.
// CHECK-LABEL: func.func @separate_pipe_values_share_ready_state
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[SENDER_READY:.*]] = ttkernel.get_semaphore(%[[READY_INDEX:.*]])
// CHECK: %[[SENDER_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[SENDER_READY]])
// CHECK: %[[RECEIVER_READY:.*]] = ttkernel.get_semaphore(%[[READY_INDEX]])
// CHECK: %[[RECEIVER_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[RECEIVER_READY]], {{.*}})
// CHECK: scf.for
// CHECK: ttkernel.experimental.semaphore_wait(%[[SENDER_READY_PTR]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc(%[[RECEIVER_READY_NOC]], {{.*}}, {{.*}})
module attributes {ttl.launch_grid = array<i64: 4, 1>} {
func.func @separate_pipe_values_share_ready_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 3>
  scf.for %iteration = %c0 to %c8 step %c1 {
    %sender_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0 {isCollective = true} : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    ttl.if_src %sender_pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0> {
      %send = ttl.copy %src_cb, %sender_pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    %receiver_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0 {isCollective = true} : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    ttl.if_dst %receiver_pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 3> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %receiver_pipe, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 3>
      %consumed = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 3> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 3>
    }
  }
  func.return
}
}

// -----

// Fabric completion and sender readiness use GlobalSemaphore addresses. The
// semantic PipeNet id does not determine their runtime-argument indices.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-LABEL: func.func @fabric_sender
// CHECK-DAG: %[[SENDER_GLOBAL_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SENDER_READY_IDX:.*]] = arith.constant 2 : index
// CHECK: %[[SENDER_READY_ADDR:.*]] = ttkernel.get_common_arg_val(%[[SENDER_READY_IDX]])
// CHECK-NEXT: %[[SENDER_READY_PTR:.*]] = ttkernel.reinterpret_cast(%[[SENDER_READY_ADDR]])
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min(%[[SENDER_READY_PTR]]
// CHECK: %[[SENDER_DONE_ADDR:.*]] = ttkernel.get_common_arg_val(%[[SENDER_GLOBAL_IDX]])
// CHECK: %[[REMOTE_DONE_ADDR:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[SENDER_DONE_ADDR]], {{.*}})
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc({{.*}}, %[[REMOTE_DONE_ADDR]], {{.*}})
// CHECK-NOT: ttkernel.get_semaphore
// CHECK-LABEL: func.func @fabric_receiver
// CHECK-DAG: %[[RECEIVER_GLOBAL_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[RECEIVER_READY_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[RECEIVER_READY_ADDR:.*]] = ttkernel.get_common_arg_val(%[[RECEIVER_READY_IDX]])
// CHECK: %[[READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[RECEIVER_READY_ADDR]], {{.*}})
// CHECK-NEXT: ttkernel.routing_plane.atomic_inc({{.*}}, %[[READY_NOC]], {{.*}})
// CHECK: %[[RECEIVER_DONE_ADDR:.*]] = ttkernel.get_common_arg_val(%[[RECEIVER_GLOBAL_IDX]])
// CHECK: %[[RECEIVER_DONE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[RECEIVER_DONE_ADDR]])
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[RECEIVER_DONE_PTR]]
// CHECK-NOT: ttkernel.get_semaphore
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @fabric_sender() attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 37 {deviceTransfer = #ttl.device_transfer<domain = <components = <name = "device", extent = [1, 4]>>, edge = <source = <coordinates = [0, 2]>, destination = <coordinates = [0, 0]>>>} : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>
    %send = ttl.copy %cb, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @fabric_receiver() attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 37 {deviceTransfer = #ttl.device_transfer<domain = <components = <name = "device", extent = [1, 4]>>, edge = <source = <coordinates = [0, 2]>, destination = <coordinates = [0, 0]>>>} : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>
    %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %pipe, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 37>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}

// -----

// CB -> Pipe (multicast, non-loopback): the sender uses the proven common
// receiver DFB address and signals every receiver after the multicast write.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.experimental.semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_async_write_barrier
// CHECK-NOT: ttkernel.noc_semaphore_set_multicast
module attributes {ttl.launch_grid = array<i64: 2, 4>} {
func.func @copy_cb_to_pipe_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %recv = ttl.cb_reserve %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// NOC1 multicast lowering reverses the translated destination rectangle before
// constructing tt-metal multicast transactions and semaphore addresses.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast_noc1
// CHECK: %[[NOC:.*]] = arith.constant 1 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_END]], %[[DST_Y_END]], %[[DST_X_START]], %[[DST_Y_START]], %[[DONE_SEM]], %[[NOC]])
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.experimental.semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_END]], %[[DST_Y_END]]], end_xy[%[[DST_X_START]], %[[DST_Y_START]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
module attributes {ttl.launch_grid = array<i64: 2, 4>} {
func.func @copy_cb_to_pipe_multicast_noc1() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc>, "ttl.noc_index" = 1 : i64 } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %recv = ttl.cb_reserve %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// CB -> Pipe (multicast loopback): payload writes use multicast with the
// computed common destination address. Signaling uses a multicast increment
// for remote receivers and a point-to-point increment for the local receiver.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast_loopback
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[REMOTE_DONE_NOC:.*]] = ttkernel.get_noc_multicast_addr(%[[DST_X_START]], %[[DST_Y_START]], %[[DST_X_END]], %[[DST_Y_END]], %[[DONE_SEM]], %[[NOC]])
// CHECK: %[[LOCAL_DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], %[[NOC]])
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: ttkernel.experimental.semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_ADDR]], noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast(%[[REMOTE_DONE_NOC]], {{.*}}, {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc(%[[LOCAL_DONE_NOC]], {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.noc_async_write_barrier
// CHECK-NOT: ttkernel.noc_semaphore_set_multicast
module attributes {ttl.launch_grid = array<i64: 1, 4>} {
func.func @copy_cb_to_pipe_multicast_loopback() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
    %recv = ttl.cb_reserve %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0> {
    %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Source-in-destination multicast computes the receiver DFB address and keeps
// aggregate ready counting.
// CHECK-LABEL: func.func @loopback_multicast_aggregate_ready_counting
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}})
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
func.func @loopback_multicast_aggregate_ready_counting() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
    %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
    %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// One-shot non-loopback multicast uses its initial destination address and one
// aggregate ready count.
// CHECK-LABEL: func.func @non_loopback_multicast_computed_address
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[BASE:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.remui
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[BASE]], {{.*}})
// CHECK-NOT: ttkernel.noc_async_write_multicast_loopback_src
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 4, 1>} {
func.func @non_loopback_multicast_computed_address() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0> {
    %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %post : !ttl.receive_request
    ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0> {
    %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
}

// -----

// Degenerate multicast metadata preserves aggregate ready counting when a
// slice-origin multicast covers one destination.
// CHECK-LABEL: func.func @degenerate_multicast_aggregate_ready_counting
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_common_arg_val
// CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}}, noc {{.*}} : (i32, index, index, i32, i32, i8) -> ()
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
func.func @degenerate_multicast_aggregate_ready_counting() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {isCollective = true} : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.receive_request
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}
}

// -----

// Pipe -> DFB (multicast receiver): reserve the proven DFB slot, notify the
// sender, and wait on the transfer-specific completion counter.
// CHECK-LABEL: func.func @copy_pipe_to_cb_multicast
// CHECK: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]], %[[NOC]])
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[DONE_SEM]])
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]], {{.*}}, %[[NOC]])
// CHECK: %[[V:.*]] = memref.load %[[CTR]]
// CHECK: %[[NEW:.*]] = arith.addi %[[V]]
// CHECK: memref.store %[[NEW]], %[[CTR]]
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[DONE_PTR]], %[[NEW]])
// CHECK: ttkernel.cb_push_back(%[[DST_DFB]]
// CHECK-NOT: ttkernel.experimental.semaphore_wait(
// CHECK: return
module attributes {ttl.launch_grid = array<i64: 2, 4>} {
func.func @copy_pipe_to_cb_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %recv = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.receive_request
    ttl.wait %xf : !ttl.receive_request
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  func.return
}

// Define the sender half so the receiver belongs to a complete transfer.
func.func @copy_pipe_to_cb_multicast_sender() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0> {
    %send = ttl.copy %src, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  func.return
}
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
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
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

// Reusing a token waits for the same completion threshold.
// CHECK-LABEL: func.func @wait_twice_for_pipe_token
// CHECK: %[[SEQUENCE:.*]] = arith.addi
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[SEQUENCE]])
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[SEQUENCE]])
func.func @wait_twice_for_pipe_token()
    attributes {"ttl.kernel_thread" = #ttkernel.thread<noc>} {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
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
