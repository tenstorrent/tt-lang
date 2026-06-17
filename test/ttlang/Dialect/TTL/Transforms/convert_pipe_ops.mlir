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

// A full-block point-to-point Pipe Transfer computes the receiver DFB slot
// address while preserving the sender-ready wait.
// CHECK-LABEL: func.func @explicit_pipe_transfer_ir
// CHECK-SAME: ttl.base_cta_index = 3 : i32
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: memref.alloca
// CHECK: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.noc_semaphore_set
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK: ttkernel.experimental::semaphore_wait_min
// CHECK-NOT: ttl.pipe_transfer
// CHECK-NOT: unrealized_conversion_cast
func.func @explicit_pipe_transfer_ir() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer_init = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %transfer = scf.for %iter = %zero to %one step %one iter_args(%transfer_arg = %transfer_init)
      -> (!ttl.pipe_transfer) {
    scf.yield %transfer_arg : !ttl.pipe_transfer
  }
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// A proven single-edge receiver DFB stream replaces sender-ready rendezvous
// with sender-local capacity released by the receiver pop.
// CHECK-LABEL: func.func @computed_address_capacity_protocol
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[CAPACITY_SEM:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[INITIAL_CAPACITY:.*]] = arith.constant 1 : i32
// CHECK: ttkernel.semaphore_set(%[[CAPACITY_SEM]], %[[INITIAL_CAPACITY]])
// CHECK: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_wait_front
// CHECK: ttkernel.cb_pop_front
// CHECK: ttkernel.semaphore_up_remote(%[[CAPACITY_SEM]]
// CHECK: ttkernel.noc_async_atomic_barrier
// CHECK: ttkernel.semaphore_down(%[[CAPACITY_SEM]]
// CHECK-NOT: ttkernel.experimental::semaphore_wait(
// CHECK-NOT: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.experimental::semaphore_wait(
// CHECK-NOT: ttkernel.noc_semaphore_set
module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @computed_address_capacity_protocol() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
    %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
    ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
      ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready = ttl.cb_wait %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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

// Static receiver subviews compute the DFB block address and add the static
// byte offset inside the block.
// CHECK-LABEL: func.func @static_subview_pipe_transfer_computed_address
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[STATIC_OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: %[[BASE:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: %[[DST_ADDR:.*]] = arith.addi %[[BASE]], %[[STATIC_OFFSET]]
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
func.func @static_subview_pipe_transfer_computed_address() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %p {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv = tensor.extract_slice %recv_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %token = ttl.pipe_transfer.post %transfer, %recv
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send = ttl.pipe_transfer.send %transfer, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Multiple incoming pipe edges to one receiver DFB compute distinct receiver
// slots and keep sender-ready counters for back-pressure.
// CHECK-LABEL: func.func @two_incoming_edges_one_dfb_compute_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK-DAG: %[[SLOT1_OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[BASE0:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[BASE0]]
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[BASE1:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK: %[[DST_ADDR1:.*]] = arith.addi %[[BASE1]], %[[SLOT1_OFFSET]]
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR1]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
func.func @two_incoming_edges_one_dfb_compute_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pA = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %pB = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 1 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1>
  %tA = ttl.pipe_transfer.create %pA {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %tB = ttl.pipe_transfer.create %pB {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 1> -> !ttl.pipe_transfer
  %recvA = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %tokA = ttl.pipe_transfer.post %tA, %recvA
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %recvB = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %tokB = ttl.pipe_transfer.post %tB, %recvB
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
  %sendA = ttl.pipe_transfer.send %tA, %src_cb0
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %sendA : !ttl.transfer_handle<write>
  %sendB = ttl.pipe_transfer.send %tB, %src_cb1
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %sendB : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %tokA : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipe_transfer.wait %tokB : !ttl.pipe_token<net 1>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Two collective senders into the same receiver DFB compute distinct multicast
// receiver slots from the graph-owned receiver-slot period.
// CHECK-LABEL: func.func @two_multicast_edges_one_dfb_compute_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK-DAG: %[[SLOT1_OFFSET:.*]] = arith.constant 4096 : i32
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[BASE0:.*]] = ttkernel.get_compile_time_arg_val({{[0-9]+}}) : () -> i32
// CHECK: ttkernel.noc_async_write_multicast_loopback_src({{.*}}, %[[BASE0]], {{.*}})
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[BASE1:.*]] = ttkernel.get_compile_time_arg_val({{[0-9]+}}) : () -> i32
// CHECK: %[[DST_ADDR1:.*]] = arith.addi %[[BASE1]], %[[SLOT1_OFFSET]]
// CHECK: ttkernel.noc_async_write_multicast_loopback_src({{.*}}, %[[DST_ADDR1]], {{.*}})
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
func.func @two_multicast_edges_one_dfb_compute_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(1, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(1, 0) dst(0, 0) to(1, 0) net 0>
  %t0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
      : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %t1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 2 : i64, kind = #ttl.pipe_transfer_kind<collective>}
      : !ttl.pipe<src(1, 0) dst(0, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %tok0 = ttl.pipe_transfer.post %t0, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %tok1 = ttl.pipe_transfer.post %t1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send0 = ttl.pipe_transfer.send %t0, %src_cb0
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.pipe_transfer.send %t1, %src_cb1
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %tok0 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipe_transfer.wait %tok1 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Reusing the same DFB index on independent receiver cores does not conflict.
// The receiver-core coordinate remains part of the incoming-edge identity, so
// both point-to-point transfers compute their receiver DFB slot addresses.
// CHECK-LABEL: func.func @independent_receivers_same_dfb_compute_addresses
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 2>
// CHECK: ttkernel.get_compile_time_arg_val(3)
// CHECK: ttkernel.get_compile_time_arg_val(3)
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
func.func @independent_receivers_same_dfb_compute_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %src_cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %pipe0 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %pipe1 = ttl.create_pipe src(1, 0) dst(3, 0) to(3, 0) net 1 : !ttl.pipe<src(1, 0) dst(3, 0) to(3, 0) net 1>
  %transfer0 = ttl.pipe_transfer.create %pipe0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %transfer1 = ttl.pipe_transfer.create %pipe1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(1, 0) dst(3, 0) to(3, 0) net 1> -> !ttl.pipe_transfer
  %recv0 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer0, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %recv1 = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 1>
  %send0 = ttl.pipe_transfer.send %transfer0, %src_cb0
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb1
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipe_transfer.wait %token1 : !ttl.pipe_token<net 1>
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.reinterpret_cast{{.*}}(%{{.*}})
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.noc_async_write
// Second send waits on p1 ready sem and computes the destination address.
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
func.func @same_source_two_pipes_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
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
func.func @same_source_three_pipes_use_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %p2 = ttl.create_pipe src(0, 0) dst(3, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv2_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv2 = tensor.extract_slice %recv2_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post2 = ttl.copy %p2, %recv2 : (!ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  %send2 = ttl.copy %src_cb, %p2 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(3, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send2 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.wait %post2 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Two same-source transfers with non-overlapping post-to-send intervals reuse
// the same ready semaphore.
// CHECK-LABEL: func.func @same_source_sequential_transfers_reuse_sync_state
// CHECK: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
func.func @same_source_sequential_transfers_reuse_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Repeated create_pipe ops with the same PipeKey share one transfer-allocation
// unit so repeated uses preserve the current per-pipe protocol state.
// CHECK-LABEL: func.func @same_pipe_key_transfer_creates_share_sync_state
// CHECK: %[[READY_IDX:.*]] = arith.constant 1 : index
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.load_from_l1
// CHECK: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.load_from_l1
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
// CHECK: scf.if
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.noc_async_write
func.func @same_source_control_flow_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
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
  %transfer1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
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
// CHECK: scf.if
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
func.func @same_source_control_flow_send_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cond = arith.constant true
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
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
  %transfer1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
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

// A transfer with a receive post but no send has no bounded post-to-send
// interval, so it conservatively conflicts with later same-source transfers.
// CHECK-LABEL: func.func @same_source_missing_send_interval_uses_distinct_sync_state
// CHECK-DAG: %[[P0_READY_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[P1_READY_IDX:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_semaphore(%[[P0_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.get_semaphore(%[[P1_READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
func.func @same_source_missing_send_interval_uses_distinct_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(0, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0>
  %transfer0 = ttl.pipe_transfer.create %p0 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> -> !ttl.pipe_transfer
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %token0 = ttl.pipe_transfer.post %transfer0, %recv0
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %transfer1 = ttl.pipe_transfer.create %p1 {expectedReceivers = 1 : i64, kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(2, 0) to(2, 0) net 0> -> !ttl.pipe_transfer
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %token1 = ttl.pipe_transfer.post %transfer1, %recv1
      : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.pipe_token<net 0>
  %send1 = ttl.pipe_transfer.send %transfer1, %src_cb
      : (!ttl.pipe_transfer, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.pipe_transfer.wait %token0 : !ttl.pipe_token<net 0>
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
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: return
func.func @different_sources_overlap_reuse_source_local_sync_state() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %p1 = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %p1, %recv1 : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  %send1 = ttl.copy %src_cb, %p1 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Unused same-source pipe declarations do not allocate ready counters; the
// active transfer keeps its ready counter in a local hardware semaphore.
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

// A transfer whose receiver completion counter consumes semaphore id 14 can
// still use local semaphore id 15 for sender-ready state.
// CHECK-LABEL: func.func @pipe_ready_counter_at_local_limit
// CHECK-DAG: %[[READY_IDX:.*]] = arith.constant 15 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_semaphore(%[[READY_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[READY_PTR]]
func.func @pipe_ready_counter_at_local_limit() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
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

// A high PipeNet id can consume the last local semaphore id for receiver
// completion, so computed addressing still uses a GlobalSemaphore-backed
// sender-ready counter.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 1 : i64
// CHECK-LABEL: func.func @high_pipe_net_uses_global_ready_counter
// CHECK-SAME: ttl.base_cta_index = 3 : i32
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[READY_ARG_IDX:.*]] = arith.constant 0 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_common_arg_val(%[[READY_ARG_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_common_arg_val(%[[READY_ARG_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.noc_async_write {{.*}}, core{{.*}}, %[[DST_ADDR]]
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_semaphore(%[[READY_ARG_IDX]])
func.func @high_pipe_net_uses_global_ready_counter() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// When one transfer forces GlobalSemaphore-backed ready counters, all
// source-local ready counters use the same compact runtime-arg layout while
// receiver completion remains per PipeNet.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-LABEL: func.func @interleaved_pipenets_use_global_ready_and_local_completion
// CHECK-DAG: %[[FIRST_READY_ARG_IDX:.*]] = arith.constant 1 : index
// CHECK: %[[READY_POST:.*]] = ttkernel.get_common_arg_val(%[[FIRST_READY_ARG_IDX]])
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[READY_POST]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[READY_SEND:.*]] = ttkernel.get_common_arg_val(%[[FIRST_READY_ARG_IDX]])
// CHECK: %[[READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[READY_SEND]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[READY_PTR]]
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.noc_async_write
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]], {{.*}})
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.load_from_l1
func.func @interleaved_pipenets_use_global_ready_and_local_completion() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 1>
  %p0 = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 15 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>
  %side = ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0) net 0 : !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>
  %recv0_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv0 = tensor.extract_slice %recv0_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post0 = ttl.copy %p0, %recv0 : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send0 = ttl.copy %src_cb, %p0 : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 15>) -> !ttl.transfer_handle<write>
  %recv1_full = ttl.cb_reserve %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %recv1 = tensor.extract_slice %recv1_full[1, 0] [1, 1] [1, 1]
      : tensor<2x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
  %post1 = ttl.copy %side, %recv1 : (!ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send1 = ttl.copy %src_cb, %side : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(2, 0) to(2, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send0 : !ttl.transfer_handle<write>
  ttl.wait %post0 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.wait %send1 : !ttl.transfer_handle<write>
  ttl.wait %post1 : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[2, 1], !ttcore.tile<32x32, f32>, 1>
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
// CHECK: %[[DST_X_START:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.*]] = ttkernel.experimental::convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.*]] = ttkernel.experimental::convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.*]] = ttkernel.experimental::convert_logical_y_to_translated
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
// selected common destination address. Signaling splits into
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

// Source-in-destination multicast computes the receiver DFB address and keeps
// aggregate ready counting.
// CHECK-LABEL: func.func @loopback_multicast_aggregate_ready_counting
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}})
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
func.func @loopback_multicast_aggregate_ready_counting() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}

// -----

// Non-loopback multicast computes one destination address and uses one
// aggregate ready count.
// CHECK-LABEL: func.func @non_loopback_multicast_computed_address
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[POST_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: ttkernel.cb_reserve_back(%[[POST_DFB]]
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], {{.*}}, {{.*}}, start_xy[{{.*}}, {{.*}}], end_xy[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}})
// CHECK-NOT: ttkernel.noc_async_write_multicast_loopback_src
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
func.func @non_loopback_multicast_computed_address() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
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
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[{{.*}}, {{.*}}], %[[DST_ADDR]], {{.*}} : (i32, index, index, i32, i32) -> ()
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.muli
// CHECK-NOT: ttkernel.load_from_l1
func.func @degenerate_multicast_aggregate_ready_counting() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %dst_cb = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {isCollective = true} : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
  %recv = ttl.cb_reserve %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %post = ttl.copy %p, %recv : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttl.transfer_handle
  %send = ttl.copy %src_cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %send : !ttl.transfer_handle<write>
  ttl.wait %post : !ttl.transfer_handle
  ttl.cb_push %dst_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
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
