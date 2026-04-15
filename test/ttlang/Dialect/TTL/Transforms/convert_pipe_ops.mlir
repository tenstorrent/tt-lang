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
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.get_noc_addr
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_inc
func.func @copy_cb_to_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Pipe -> CB copy (unicast receiver): wait for sender semaphore, then reset
// CHECK-LABEL: func.func @copy_pipe_to_cb
// CHECK: ttkernel.get_semaphore
// CHECK-NEXT: ttkernel.reinterpret_cast
// CHECK-NEXT: ttkernel.experimental::semaphore_wait
// CHECK-NEXT: ttkernel.noc_semaphore_set
func.func @copy_pipe_to_cb() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %xf = ttl.copy %p, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}

// -----

// CB -> Pipe copy (multicast): lowers to semaphore handshake + noc_async_write_multicast
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast
// CHECK: ttkernel.get_semaphore
// CHECK-NEXT: ttkernel.reinterpret_cast
// CHECK-NEXT: ttkernel.experimental::semaphore_wait
// CHECK-NEXT: ttkernel.noc_semaphore_set
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.experimental::get_noc_multicast_addr
// CHECK-NEXT: ttkernel.noc_async_write_multicast
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.experimental::get_noc_multicast_addr
// CHECK-NEXT: ttkernel.noc_semaphore_set_multicast
func.func @copy_cb_to_pipe_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// CB -> Pipe copy (multicast loopback): sender is within dst range, uses loopback variant
// CHECK-LABEL: func.func @copy_cb_to_pipe_multicast_loopback
// CHECK: ttkernel.get_semaphore
// CHECK-NEXT: ttkernel.reinterpret_cast
// CHECK-NEXT: ttkernel.experimental::semaphore_wait
// CHECK-NEXT: ttkernel.noc_semaphore_set
// CHECK: ttkernel.get_write_ptr
// CHECK: ttkernel.experimental::get_noc_multicast_addr
// CHECK-NEXT: ttkernel.noc_async_write_multicast_loopback_src
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.experimental::get_noc_multicast_addr
// CHECK-NEXT: ttkernel.noc_semaphore_set_multicast_loopback
func.func @copy_cb_to_pipe_multicast_loopback() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3) net 0 : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(0, 0) to(0, 3) net 0>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Pipe -> CB copy (multicast receiver): handshake with sender then wait for data
// CHECK-LABEL: func.func @copy_pipe_to_cb_multicast
// CHECK: ttkernel.get_semaphore
// CHECK-NEXT: ttkernel.reinterpret_cast
// CHECK-NEXT: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_semaphore_inc
// CHECK-NEXT: ttkernel.experimental::semaphore_wait
func.func @copy_pipe_to_cb_multicast() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>
  %xf = ttl.copy %p, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 3) net 0>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle
  ttl.wait %xf : !ttl.transfer_handle
  func.return
}
