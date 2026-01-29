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
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  ttl.if_src %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)> {
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
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 3) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)>
  ttl.if_dst %p : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 3)> {
    "ttkernel.noc_async_read_barrier"() : () -> ()
  }
  func.return
}

// -----

// CB -> Pipe copy: lowers to multicast write ops
// CHECK-LABEL: func.func @copy_cb_to_pipe
// CHECK: ttkernel.get_compile_time_arg_val
// CHECK: ttkernel.get_read_ptr
// CHECK: ttkernel.get_noc_multicast_addr
// CHECK: ttkernel.noc_async_write_multicast
// CHECK: ttkernel.noc_async_write_barrier
func.func @copy_cb_to_pipe() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  %xf = ttl.copy %cb, %p : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>) -> !ttl.transfer_handle<write>
  ttl.wait %xf : !ttl.transfer_handle<write>
  func.return
}

// -----

// Pipe -> CB copy: destination side, no-op (data arrives via multicast)
// The CB may be optimized away if unused, but the wait becomes a read barrier.
// CHECK-LABEL: func.func @copy_pipe_to_cb
// CHECK: ttkernel.noc_async_read_barrier
// CHECK-NOT: ttkernel.noc_async_read_tile
func.func @copy_pipe_to_cb() attributes { "ttl.kernel_thread" = #ttkernel.thread<noc> } {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %p = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>
  %xf = ttl.copy %p, %cb : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0)>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> !ttl.transfer_handle<read>
  ttl.wait %xf : !ttl.transfer_handle<read>
  func.return
}
