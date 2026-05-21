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
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[MAILBOX_SEM]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1(%[[MAILBOX_PTR]]
// CHECK: %[[DST_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DST_ADDR]])
// CHECK: ttkernel.noc_async_write(%[[SRC_ADDR]], %[[DST_NOC]]
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]])
// CHECK: ttkernel.noc_semaphore_inc(%[[DONE_NOC]]
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
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[MAILBOX_SEM]])
// CHECK: ttkernel.store_to_l1(%[[DST_ADDR]], %[[MAILBOX_PTR]]
// CHECK: %[[MAILBOX_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[MAILBOX_SEM]])
// CHECK: ttkernel.remote_sram_write_u32(%[[MAILBOX_SEM]], %[[MAILBOX_NOC]])
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]])
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]]
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

// CB -> Pipe (multicast, non-loopback): sender wait, multicast write,
// inc_multicast on every receiver's recvSem.
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[MAILBOX_SEM]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1(%[[MAILBOX_PTR]]
// CHECK: %[[DST_NOC:.*]] = ttkernel.experimental::get_noc_multicast_addr({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[DST_ADDR]])
// CHECK-NEXT: ttkernel.noc_async_write_multicast(%[[SRC_ADDR]], %[[DST_NOC]]
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[DONE_NOC:.*]] = ttkernel.experimental::get_noc_multicast_addr({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[DONE_SEM]])
// CHECK-NEXT: ttkernel.noc_semaphore_inc_multicast(%[[DONE_NOC]]
// CHECK-NOT: ttkernel.noc_semaphore_set_multicast
func.func @copy_cb_to_pipe_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// CB -> Pipe (multicast loopback): data uses
// noc_async_write_multicast_loopback_src; signal splits into
// inc_multicast to remote receivers + local noc_semaphore_inc on self
// (no inc_multicast loopback in tt-metal).
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast_loopback
// CHECK: %[[SRC_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[ADDR_READY_SEM]])
// CHECK: ttkernel.experimental::semaphore_wait(%[[ADDR_READY_PTR]]
// CHECK: ttkernel.noc_semaphore_set(%[[ADDR_READY_PTR]]
// CHECK: %[[SRC_ADDR:.*]] = ttkernel.get_write_ptr(%[[SRC_DFB]])
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[MAILBOX_SEM]])
// CHECK: %[[DST_ADDR:.*]] = ttkernel.load_from_l1(%[[MAILBOX_PTR]]
// CHECK: %[[DST_NOC:.*]] = ttkernel.experimental::get_noc_multicast_addr({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[DST_ADDR]])
// CHECK-NEXT: ttkernel.noc_async_write_multicast_loopback_src(%[[SRC_ADDR]], %[[DST_NOC]]
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: %[[DONE_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[REMOTE_DONE_NOC:.*]] = ttkernel.experimental::get_noc_multicast_addr({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[DONE_SEM]])
// CHECK-NEXT: ttkernel.noc_semaphore_inc_multicast(%[[REMOTE_DONE_NOC]]
// CHECK: %[[LOCAL_DONE_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[DONE_SEM]])
// CHECK: ttkernel.noc_semaphore_inc(%[[LOCAL_DONE_NOC]]
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
// CHECK: %[[CTR:.*]] = memref.alloca() : memref<1xi32>
// CHECK: memref.store {{.*}}, %[[CTR]]
// CHECK: %[[DST_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: ttkernel.cb_reserve_back(%[[DST_DFB]]
// CHECK: %[[DST_ADDR:.*]] = ttkernel.get_write_ptr(%[[DST_DFB]])
// CHECK: %[[MAILBOX_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[MAILBOX_PTR:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[MAILBOX_SEM]])
// CHECK: ttkernel.store_to_l1(%[[DST_ADDR]], %[[MAILBOX_PTR]]
// CHECK: %[[MAILBOX_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[MAILBOX_SEM]])
// CHECK: ttkernel.remote_sram_write_u32(%[[MAILBOX_SEM]], %[[MAILBOX_NOC]])
// CHECK: %[[ADDR_READY_SEM:.*]] = ttkernel.get_semaphore
// CHECK: %[[ADDR_READY_NOC:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[ADDR_READY_SEM]])
// CHECK: ttkernel.noc_semaphore_inc(%[[ADDR_READY_NOC]]
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
