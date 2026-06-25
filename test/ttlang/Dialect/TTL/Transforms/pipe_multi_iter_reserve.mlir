// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies the receiver-posted pipe protocol. Senders consume the
// destination address published by the receiver and never advance the receiver
// DFB. Receivers reserve the destination DFB slot, publish its address, wait
// for completion, and then push the slot.

// Sender-side unicast lowering waits for a receiver-published destination
// address and uses that address for the NoC write. It must not reserve or push
// the receiver DFB.
// CHECK-LABEL: func.func @sender_uses_published_unicast_address
// CHECK: %[[NOC:.+]] = arith.constant 0 : i8
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: ttkernel.cb_reserve_back
// CHECK: %[[SRC_WP:.+]] = ttkernel.get_write_ptr
// CHECK: %[[DST_X:.+]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y:.+]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_WP:.+]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DST_WP]])
// CHECK: ttkernel.noc_async_write %[[SRC_WP]], core[%[[DST_X]], %[[DST_Y]]], %[[DST_WP]], {{.*}}, noc %[[NOC]] : (i32, index, index, i32, i32, i8) -> ()
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NOT: ttkernel.cb_push_back
// CHECK: return
func.func @sender_uses_published_unicast_address() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %src, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Sender-side multicast lowering also consumes receiver-published addresses.
// The sender waits for all destinations to publish before issuing the multicast
// write and must not advance any receiver DFB.
// CHECK-LABEL: func.func @sender_uses_published_multicast_addresses
// CHECK: %[[NOC:.+]] = arith.constant 0 : i8
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK-NOT: ttkernel.cb_reserve_back
// CHECK: %[[SRC_WP:.+]] = ttkernel.get_write_ptr
// CHECK: %[[DST_X_START:.+]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_START:.+]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_X_END:.+]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y_END:.+]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[DST_WP:.+]] = ttkernel.load_from_l1
// CHECK-NOT: ttkernel.get_noc_multicast_addr({{.*}}, %[[DST_WP]]
// CHECK: ttkernel.noc_async_write_multicast(%[[SRC_WP]], {{.*}}, {{.*}}, start_xy[%[[DST_X_START]], %[[DST_Y_START]]], end_xy[%[[DST_X_END]], %[[DST_Y_END]]], %[[DST_WP]], noc %[[NOC]])
// CHECK: ttkernel.noc_async_write_barrier(%[[NOC]])
// CHECK: ttkernel.noc_semaphore_inc_multicast({{.*}}, {{.*}}, {{.*}}, %[[NOC]])
// CHECK: ttkernel.noc_async_atomic_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.cb_push_back
// CHECK: return
func.func @sender_uses_published_multicast_addresses() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %src, %pipe
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
         !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>)
      -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Receiver lowering publishes the reserved DFB write pointer before waiting
// for the sender's completion signal.
// CHECK-LABEL: func.func @receiver_publishes_reserved_dfb_address
// CHECK: ttkernel.cb_reserve_back
// CHECK: %[[DST_WP:.+]] = ttkernel.get_write_ptr
// CHECK: ttkernel.noc_inline_dw_write({{.*}}, %[[DST_WP]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[OLD:.+]] = memref.load
// CHECK: %[[NEW:.+]] = arith.addi %[[OLD]]
// CHECK: memref.store %[[NEW]]
// CHECK: ttkernel.experimental.semaphore_wait_min({{.*}}, %[[NEW]])
// CHECK: ttkernel.cb_push_back
func.func @receiver_publishes_reserved_dfb_address() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst = ttl.bind_cb {cb_index = 5, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
  %recv_dst = ttl.cb_reserve %dst
      : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %xf = ttl.copy %pipe, %recv_dst
      : (!ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>,
         tensor<1x1x!ttcore.tile<32x32, f32>>)
      -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  func.return
}

// -----

// Repeated receives use a cumulative per-PipeNet completion counter. The
// expected wait value is incremented inside the loop, so each iteration waits
// for the next transfer rather than reusing the first completion value.
// CHECK-LABEL: func.func @receiver_advances_wait_counter_inside_loop
// CHECK: %[[CTR:.+]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]
// CHECK: %[[DST_DFB:.+]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: scf.for
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.+]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: ttkernel.noc_inline_dw_write({{.*}}, %[[DST_ADDR]]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: %[[DONE_PTR:.+]] = ttkernel.reinterpret_cast
// CHECK: memref.load %[[CTR]]
// CHECK: %[[NEXT:.+]] = arith.addi
// CHECK: memref.store %[[NEXT]], %[[CTR]]
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[DONE_PTR]], %[[NEXT]])
// CHECK: ttkernel.cb_push_back(%[[DST_DFB]]
func.func @receiver_advances_wait_counter_inside_loop() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0
      : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %iter = %c0 to %c2 step %c1 {
    %recv_dst = ttl.cb_reserve %dst
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %pipe, %recv_dst
        : (!ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  }
  func.return
}
