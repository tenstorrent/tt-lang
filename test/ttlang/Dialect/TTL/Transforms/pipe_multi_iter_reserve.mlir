// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Sender-side unicast lowering must reserve and push the destination DFB on
// the sender's local view so its fifo_wr_ptr stays in lockstep with the
// receiver's across iterations. With no receiver in this module, the source
// and destination DFB index coincide (both cb_index = 0).
// CHECK-LABEL: func.func @sender_reserves_dest_dfb_unicast
// CHECK: %[[N:.+]] = arith.constant 1 : i32
// CHECK: %[[DFB:.+]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DFB]], %[[N]])
// CHECK: %[[WP:.+]] = ttkernel.get_write_ptr(%[[DFB]])
// CHECK: ttkernel.get_noc_addr({{.*}}, %[[WP]])
// CHECK: ttkernel.noc_async_write(
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.cb_push_back(%[[DFB]], %[[N]])
func.func @sender_reserves_dest_dfb_unicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Multicast lowering: same bracketing pattern on the destination DFB.
// CHECK-LABEL: func.func @sender_reserves_dest_dfb_multicast
// CHECK: %[[N:.+]] = arith.constant 1 : i32
// CHECK: %[[DFB:.+]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.experimental::semaphore_wait
// CHECK: ttkernel.cb_reserve_back(%[[DFB]], %[[N]])
// CHECK: %[[WP:.+]] = ttkernel.get_write_ptr(%[[DFB]])
// CHECK: ttkernel.experimental::get_noc_multicast_addr({{.*}}, %[[WP]])
// CHECK-NEXT: ttkernel.noc_async_write_multicast(
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_set_multicast
// CHECK: ttkernel.cb_push_back(%[[DFB]], %[[N]])
func.func @sender_reserves_dest_dfb_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Gather: a pipe whose receiver uses a different DFB index than the sender.
// The PipeGraph picks up the Pipe -> CB op in @receiver and the sender lowering
// must reserve/push the *destination* DFB (cb_index = 5), not its own source
// DFB (cb_index = 3).
// CHECK-LABEL: func.func @sender
// CHECK: %[[N:.+]] = arith.constant 1 : i32
// CHECK: %[[SRC:.+]] = ttkernel.get_compile_time_arg_val(3)
// CHECK: %[[DST:.+]] = ttkernel.get_compile_time_arg_val(5)
// CHECK: ttkernel.cb_reserve_back(%[[DST]], %[[N]])
// CHECK: %[[WP_DST:.+]] = ttkernel.get_write_ptr(%[[DST]])
// CHECK: %[[WP_SRC:.+]] = ttkernel.get_write_ptr(%[[SRC]])
// CHECK: ttkernel.get_noc_addr({{.*}}, %[[WP_DST]])
// CHECK: ttkernel.noc_async_write(%[[WP_SRC]],
// CHECK: ttkernel.cb_push_back(%[[DST]], %[[N]])
func.func @sender() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0 : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
  %xf = ttl.copy %src, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

func.func @receiver() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0) net 0 : !ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>
  %xf = ttl.copy %p, %dst : (!ttl.pipe<src(1, 0) dst(0, 0) to(0, 0) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// Multicast loopback (same-index): the sender is in the destination range
// and reads/writes the same DFB index. The receive callback on the sender
// core already issues reserve_back / push_back on the destination DFB, so
// the lowering must not double-advance. Both the multicast destination
// address and the source-data address come from `get_write_ptr` on the
// single CB handle (two SSA values, same backing CB), and the call must
// be `noc_async_write_multicast_loopback_src` to include the sender core.
// CHECK-LABEL: func.func @sender_skips_reserve_in_loopback
// CHECK-NOT: ttkernel.cb_reserve_back
// CHECK: %[[CB:.+]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[WP_DST:.+]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK: %[[WP_SRC:.+]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK: ttkernel.experimental::get_noc_multicast_addr({{.*}}, %[[WP_DST]])
// CHECK-NEXT: ttkernel.noc_async_write_multicast_loopback_src(%[[WP_SRC]],
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK-NOT: ttkernel.cb_push_back
// CHECK: return
func.func @sender_skips_reserve_in_loopback() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Cross-DFB multicast loopback: sender is in the destination range but
// reads from cb_index=3 (source DFB) while the receiver writes into
// cb_index=5 (destination DFB). The receive callback on the sender core
// advances the destination DFB; the sender path must NOT advance it again.
// At the same time the NOC write must still target the destination DFB
// (write_ptr of cb_index=5) and pull source data from the source DFB
// (write_ptr of cb_index=3).
// CHECK-LABEL: func.func @sender_skips_reserve_in_cross_dfb_loopback
// CHECK-NOT: ttkernel.cb_reserve_back
// CHECK: %[[SRC:.+]] = ttkernel.get_compile_time_arg_val(3)
// CHECK: %[[DST:.+]] = ttkernel.get_compile_time_arg_val(5)
// CHECK: %[[WP_DST:.+]] = ttkernel.get_write_ptr(%[[DST]])
// CHECK: %[[WP_SRC:.+]] = ttkernel.get_write_ptr(%[[SRC]])
// CHECK: ttkernel.experimental::get_noc_multicast_addr({{.*}}, %[[WP_DST]])
// CHECK-NEXT: ttkernel.noc_async_write_multicast_loopback_src(%[[WP_SRC]],
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK-NOT: ttkernel.cb_push_back
// CHECK: return
func.func @sender_skips_reserve_in_cross_dfb_loopback() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %src = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  %xf = ttl.copy %src, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

func.func @receiver_cross_dfb_loopback() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %dst = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  %xf = ttl.copy %p, %dst : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}
